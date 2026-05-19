#!/usr/bin/env python3
import os
import shutil
from typing import Sequence

import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util

from .config import Config
from .utils import require_fsl, pkg_file
from .nodes.data_onboard import CopyImagesToRaw, CopyExVivoSidecars, WritePathsFile
from .nodes.brainmasking import BrainMask
from .nodes.drift_wrap import Drift
from .nodes.Gibbs import Gibbs
from .nodes.spatial_drift import SpatialDrift
from .nodes.eddy import PrepareEddy, EddyRun, EddyQuad
from .nodes.difftensor import DiffTensor
from .nodes.reg import Reg2Std, MMORFReg
from .nodes.bedpostx_xtract import FibreCudimot, BedpostX, Xtract
from .nodes.qa_report import QAReport

from .nodes.denoise import Denoise

from nipype import config as nipype_config
nipype_config.update_config({"execution": {"hash_method": "content"}})


# -----------------------------
# Helpers
# -----------------------------
def _resolve_first_existing(cands):
    for p in cands:
        if p and os.path.isfile(p):
            return p
    return None


def _resolve_file(preferred, fallback_rel, files_dir):
    if preferred and os.path.isfile(preferred):
        return preferred
    if fallback_rel:
        fb = os.path.join(files_dir, fallback_rel)
        if os.path.isfile(fb):
            return fb
    return None


def _eddy_outputs_exist(base):
    must = [
        base + ".eddy_parameters",
        base + ".eddy_rotated_bvecs",
        base + ".eddy_movement_rms",
    ]
    has_txt = all(os.path.isfile(p) for p in must)
    has_img = any(
        os.path.isfile(p)
        for p in (
            base + ".nii.gz",
            base + "_corrected.nii.gz",
            base + "_unwarped_images.nii.gz",
        )
    )
    return has_txt and has_img


def _eddy_qc_done(base):
    qc_dir = base + ".qc"
    return os.path.isdir(qc_dir) and os.path.isfile(os.path.join(qc_dir, "qc.json"))


def _link_into_datadir(imain, outdir, name="data.nii.gz"):
    import os, shutil
    if isinstance(imain, (list, tuple)):
        imain = imain[0]
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    dst = os.path.join(outdir, name)
    src = os.path.abspath(imain)
    try:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return os.path.abspath(dst)


def _copy_or_link(src, dst):
    import os, shutil
    if not src:
        return ""
    src = os.path.abspath(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return os.path.abspath(dst)


def _sync_grads(bvals, bvecs, outdir):
    import os
    outdir = os.path.abspath(outdir)
    bvals_out = _copy_or_link(bvals, os.path.join(outdir, "bvals")) if bvals else ""
    bvecs_out = _copy_or_link(bvecs, os.path.join(outdir, "bvecs")) if bvecs else ""
    return bvals_out, bvecs_out


def _require_files(paths: Sequence[str], label: str):
    missing = [p for p in paths if not (p and os.path.isfile(p))]
    if missing:
        raise FileNotFoundError(
            f"[tractography] Missing required {label} file(s):\n  - " + "\n  - ".join(missing)
        )


# -----------------------------
# Main
# -----------------------------
def build_workflow(cfg: Config, verbose: bool = False, run_mode: str = "preproc_only") -> pe.Workflow:
    """
    run_mode:
      - 'preproc_only' : build preprocessing graph only
      - 'tract_only'   : assume preprocessed outputs on disk; run BedpostX/Fibre -> XTRACT -> QA
      - 'full'         : preprocessing + BedpostX/Fibre -> XTRACT -> QA
    """
    require_fsl()
    run_mode = (run_mode or "preproc_only").strip().lower()
    if run_mode not in {"preproc_only", "tract_only", "full"}:
        raise ValueError("run_mode must be 'preproc_only', 'tract_only', or 'full'.")

    # Package dirs for fallbacks
    PKG_DIR = os.path.dirname(__file__)
    FILES_DIR = os.path.join(PKG_DIR, "files")
    NODES_DIR = os.path.join(PKG_DIR, "nodes")

    # Workspace layout
    dmri_root = cfg.abs_dmri_root
    proc_dir = dmri_root

    data_rawdir = os.path.join(proc_dir, "data_raw")
    sidecarsdir = os.path.join(proc_dir, "sidecars")
    datadir = os.path.join(proc_dir, "data")
    denoise_dir = os.path.join(proc_dir, "denoise")

    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(data_rawdir, exist_ok=True)
    os.makedirs(sidecarsdir, exist_ok=True)
    os.makedirs(datadir, exist_ok=True)
    os.makedirs(denoise_dir, exist_ok=True)
    
    acqtype = (getattr(cfg, "acqtype", "ssfp") or "ssfp").strip().lower()
    mode = (getattr(cfg, "MODE", getattr(cfg, "mode", "nhp")) or "nhp").strip().lower()  # "nhp" | "hum"

    # Keep a copy of YAML used
    if getattr(cfg, "source_yaml", None) and os.path.isfile(cfg.source_yaml):
        try:
            shutil.copy2(cfg.source_yaml, os.path.join(proc_dir, "config_used.yaml"))
        except Exception as e:
            print(f"[pipeline] Warning: could not copy YAML: {e}")

    # Choose workflow/log dir name by run_mode
    wf_name = "nipype_logging" if run_mode in ("preproc_only", "full") else "nipype_logging_tract"

    # Nipype workflow + base dir
    wf = pe.Workflow(name=wf_name)
    wf.base_dir = proc_dir  # logs/working dirs live under: <proc_dir>/<wf_name>/

    # Convenience paths for QA
    log_dir = os.path.join(proc_dir, wf_name)
    graph_png = os.path.join(log_dir, "graph.png")

    # Common subdirs
    gibbs_dir = os.path.join(proc_dir, "gibbs")
    drift_dir = os.path.join(proc_dir, "drift_correction")
    spdrift_dir = os.path.join(proc_dir, "data_spatial_drift")
    eddy_root = os.path.join(proc_dir, "eddy")
    stdreg_dir = os.path.join(proc_dir, "stdreg")
    xtract_out = os.path.join(proc_dir, "xtract")
    qa_outdir = os.path.join(proc_dir, "QA")

    # Warp paths
    warp_anat2std_fn = os.path.join(stdreg_dir, "stdreg_anat_to_std_warp.nii.gz")
    warp_std2anat_fn = os.path.join(stdreg_dir, "stdreg_std_to_anat_warp.nii.gz")
    mmorf_combined = os.path.join(stdreg_dir, "combined_warp.nii.gz")
    mmorf_combined_inv = os.path.join(stdreg_dir, "combined_warp_inv.nii.gz")

    # ----------------------------------------------------------
    # TRACT-ONLY MODE: assume preproc files exist on disk
    # ----------------------------------------------------------
    if run_mode == "tract_only":
        print("[pipeline] Building tract-only workflow (BedpostX/Fibre -> XTRACT -> QA).")

        data_nii = os.path.join(datadir, "data.nii.gz")
        mask_file = os.path.join(datadir, "nodif_brain_mask.nii.gz")

        # Always require data + mask
        _require_files([data_nii, mask_file], "preproc outputs (data/mask)")

        # For SE, also require bvals/bvecs (BedpostX needs them)
        if acqtype == "se":
            bvals_file = os.path.join(datadir, "bvals")
            bvecs_file = os.path.join(datadir, "bvecs")
            _require_files([bvals_file, bvecs_file], "preproc outputs (bvals/bvecs)")

        # Registration method
        reg_method = (getattr(cfg, "reg_method", None) or getattr(cfg, "REG_METHOD", "FNIRT")).strip().lower()
        if reg_method == "fnirt":
            _require_files([warp_std2anat_fn, warp_anat2std_fn], "FNIRT warps")
            warp_std2anat = warp_std2anat_fn
            warp_anat2std = warp_anat2std_fn
        elif reg_method == "mmorf":
            _require_files([mmorf_combined, mmorf_combined_inv], "MMORF warps")
            warp_std2anat = mmorf_combined_inv
            warp_anat2std = mmorf_combined
        else:
            raise ValueError(f"Unknown reg_method: {reg_method}")

        # SSFP or SE tractography path
        if acqtype == "ssfp":
            fibre_deps = pe.Node(util.Merge(2), name="fibre_deps_tractonly")
            fibre_deps.inputs.in1 = mask_file
            fa_std_guess = os.path.join(stdreg_dir, "FA_in_STD.nii.gz")
            fibre_deps.inputs.in2 = fa_std_guess if os.path.isfile(fa_std_guess) else ""

            fibre_node = pe.Node(FibreCudimot(), name="fibre_cudimot")
            fibre_node.inputs.subject_dir = datadir
            fibre_node.inputs.pipeline_cmd = getattr(cfg, "fibre_pipeline_cmd", getattr(cfg, "FIBRE_PIPELINE_CMD", "Pipeline_Ball_3_Sticks_1Shell_SSFP.sh"))
            fibre_node.inputs.init_method = getattr(cfg, "fibre_init_method", getattr(cfg, "FIBRE_INIT_METHOD", ""))
            fibre_node.inputs.queue = getattr(cfg, "fibre_queue", getattr(cfg, "FIBRE_QUEUE", ""))
            fibre_node.inputs.njobs = int(getattr(cfg, "fibre_njobs", getattr(cfg, "FIBRE_NJOBS", 4)))
            fibre_node.inputs.run_mcmc = bool(getattr(cfg, "fibre_run_mcmc", getattr(cfg, "FIBRE_RUN_MCMC", True)))
            fibre_node.inputs.no_levmar = bool(getattr(cfg, "fibre_no_levmar", getattr(cfg, "FIBRE_NO_LEVMAR", True)))
            fibre_node.inputs.burnin = int(getattr(cfg, "fibre_burnin", getattr(cfg, "FIBRE_BURNIN", 10000)))
            fibre_node.inputs.jumps = int(getattr(cfg, "fibre_jumps", getattr(cfg, "FIBRE_JUMPS", 1250)))
            fibre_node.inputs.sample_every = int(getattr(cfg, "fibre_sample_every", getattr(cfg, "FIBRE_SAMPLE_EVERY", 25)))
            wf.connect(fibre_deps, "out", fibre_node, "deps")

            xtract_node = pe.Node(Xtract(), name="xtract")
            xtract_node.inputs.outdir = xtract_out
            xtract_node.inputs.mode = mode
            if mode == "hum":
                xtract_node.inputs.stdref = getattr(cfg, "xtract_stdref_hum", None)
            else:
                xtract_node.inputs.stdref = getattr(cfg, "std_ref_file", None)
            xtract_node.inputs.species = getattr(cfg, "xtract_species", getattr(cfg, "XTRACT_SPECIES", "CUSTOM"),)
            xtract_node.inputs.mac_ext = getattr(cfg, "xtract_mac_ext", getattr(cfg, "xtract_mac_ext_default", "_NMT"),)
            xtract_node.inputs.no_gpu = bool(getattr(cfg, "xtract_no_gpu", getattr(cfg, "XTRACT_NO_GPU", False)))
            xtract_node.inputs.ptx_steplength = float(getattr(cfg, "xtract_ptx_steplength", getattr(cfg, "XTRACT_PTX_STEPLENGTH", 0.15)))
            xtract_node.inputs.profiles_dir = getattr(cfg, "xtract_profiles_dir", getattr(cfg, "XTRACT_PROFILES_DIR", ""),)
            # Viewer
            xtract_node.inputs.viewer_script = pkg_file("nodes", "tract_viewer.py")
            xtract_node.inputs.do_viewer = cfg.DO_VIEWER
            wf.connect(fibre_node, "cudimot_dir", xtract_node, "cudimot_dir")

        else:
            bpx_node = pe.Node(BedpostX(), name="bedpostx")
            bpx_node.inputs.data_dir = datadir

            xtract_node = pe.Node(Xtract(), name="xtract")
            xtract_node.inputs.outdir = xtract_out
            xtract_node.inputs.mode = mode
            if mode == "hum":
                xtract_node.inputs.stdref = getattr(cfg, "xtract_stdref_hum", None)
            else:
                xtract_node.inputs.stdref = getattr(cfg, "std_ref_file", None)
            xtract_node.inputs.species = getattr(cfg, "xtract_species", getattr(cfg, "XTRACT_SPECIES", "CUSTOM"),)
            xtract_node.inputs.mac_ext = getattr(cfg, "xtract_mac_ext", getattr(cfg, "xtract_mac_ext_default", "_NMT"),)
            xtract_node.inputs.no_gpu = bool(getattr(cfg, "xtract_no_gpu", getattr(cfg, "XTRACT_NO_GPU", False)))
            xtract_node.inputs.ptx_steplength = float(getattr(cfg, "xtract_ptx_steplength", getattr(cfg, "XTRACT_PTX_STEPLENGTH", 0.15)))
            xtract_node.inputs.profiles_dir = getattr(cfg, "xtract_profiles_dir", getattr(cfg, "XTRACT_PROFILES_DIR", ""),)
            # Viewer
            xtract_node.inputs.viewer_script = pkg_file("nodes", "tract_viewer.py")
            xtract_node.inputs.do_viewer = cfg.DO_VIEWER
            wf.connect(bpx_node, "bpx_dir", xtract_node, "bpx_dir")

        # Warps into XTRACT
        xtract_node.inputs.warp_std2anat = warp_std2anat
        xtract_node.inputs.warp_anat2std = warp_anat2std

        # Viewer
        xtract_node.inputs.viewer_script = pkg_file("nodes", "tract_viewer.py")
        xtract_node.inputs.do_viewer = bool(getattr(cfg, "do_viewer", True))

        # QA for tract-only
        qa = pe.Node(QAReport(), name="qa_report")
        qa.inputs.dmri_root = dmri_root
        qa.inputs.outdir = qa_outdir
        qa.inputs.pipeline_yaml = os.path.join(proc_dir, "config_used.yaml")
        qa.inputs.run_mode = "tract_only"

        qa.inputs.nipype_dir_tract = log_dir
        qa.inputs.nipype_graph_tract = graph_png
        qa.inputs.nipype_dir_preproc = os.path.join(proc_dir, "nipype_logging")
        qa.inputs.nipype_graph_preproc = os.path.join(proc_dir, "nipype_logging", "graph.png")

        qa.inputs.eddy_qc_json = os.path.join(eddy_root, "eddy.qc", "qc.json")
        # standard-space outline (generic, optional)
        qa.inputs.std_mask_img = getattr(cfg, "std_mask_file", None)

        # Core inputs
        qa.inputs.data_4d = os.path.join(eddy_root, "eddy.nii.gz")
        qa.inputs.bvals = os.path.join(eddy_root, "bvals")
        qa.inputs.bvecs = os.path.join(eddy_root, "eddy.eddy_rotated_bvecs")
        qa.inputs.mask_file = os.path.join(datadir, "nodif_brain_mask.nii.gz")

        # add native FA/V1 if present
        fa_native = os.path.join(datadir, "dti", "dti_FA.nii.gz")
        v1_native = os.path.join(datadir, "dti", "dti_V1.nii.gz")
        if os.path.isfile(fa_native):
            qa.inputs.fa_img = fa_native
        if os.path.isfile(v1_native):
            qa.inputs.v1_img = v1_native

        wf.connect(xtract_node, "outdir", qa, "xtract_dir")

        qa_deps = pe.Node(util.Merge(2), name="qa_deps_tractonly")
        qa_deps.inputs.in1 = os.path.join(datadir, "nodif_brain_mask.nii.gz")
        qa_deps.inputs.in2 = warp_anat2std
        wf.connect(qa_deps, "out", qa, "deps")

        return wf

    # ----------------------------------------------------------
    # PREPROC GRAPH (used for both preproc_only and full)
    # ----------------------------------------------------------
    print("[pipeline] Building preprocessing workflow (up to DTIFIT and registration).")

    # 1) Sidecars / grads
    if acqtype == "ssfp":
        sc_node = pe.Node(CopyExVivoSidecars(), name="copy_sidecars")
        sc_node.inputs.bvals = cfg.data_paths.get("bvals", "")
        sc_node.inputs.bvecs = cfg.data_paths.get("bvecs", "")
        sc_node.inputs.TRs          = cfg.sidecars.get("trs", "")
        sc_node.inputs.diffGradAmps = cfg.sidecars.get("diffgradamps", "")
        sc_node.inputs.diffGradDurs = cfg.sidecars.get("diffgraddurs", "")
        sc_node.inputs.flipAngles   = cfg.sidecars.get("flipangles", "")
        sc_node.inputs.flipMask     = cfg.sidecars.get("flipmask", "")
        sc_node.inputs.noisefloor   = cfg.sidecars.get("noisefloor", "")
        sc_node.inputs.whichshell   = cfg.sidecars.get("whichshell", "")
        sc_node.inputs.outdir = sidecarsdir
        grads_src = sc_node
    else:
        def _copy_grads(bvals, bvecs, outdir):
            import os, shutil
            outdir = os.path.abspath(outdir)
            os.makedirs(outdir, exist_ok=True)
            bvals_out = os.path.join(outdir, "bvals")
            bvecs_out = os.path.join(outdir, "bvecs")
            if bvals and os.path.isfile(bvals):
                shutil.copy2(os.path.abspath(bvals), bvals_out)
            if bvecs and os.path.isfile(bvecs):
                shutil.copy2(os.path.abspath(bvecs), bvecs_out)
            return (
                bvals_out if os.path.isfile(bvals_out) else "",
                bvecs_out if os.path.isfile(bvecs_out) else "",
            )

        sc_node = pe.Node(
            util.Function(
                input_names=["bvals", "bvecs", "outdir"],
                output_names=["bvals_file", "bvecs_file"],
                function=_copy_grads,
            ),
            name="copy_grads_only",
        )
        sc_node.inputs.bvals = cfg.data_paths.get("bvals", "")
        sc_node.inputs.bvecs = cfg.data_paths.get("bvecs", "")
        sc_node.inputs.outdir = sidecarsdir
        grads_src = sc_node

    # 2) Data onboarding (SSFP only copies into data_raw/)
    if acqtype == "ssfp":
        img_node = pe.Node(CopyImagesToRaw(), name="copy_images")
        img_node.inputs.data_nii = cfg.data_paths["data_nii"]
        img_node.inputs.T1map = cfg.maps["t1map"]
        img_node.inputs.T2map = cfg.maps["t2map"]
        img_node.inputs.B1map = cfg.maps["b1map"]
        img_node.inputs.outdir = data_rawdir
        if getattr(cfg, "brain_mask", None) and os.path.isfile(cfg.brain_mask):
            img_node.inputs.brain_mask = cfg.brain_mask
    else:
        img_node = None  # SE uses provided data_nii directly

    # 2b) Optional denoising (immediately before Gibbs)
    denoise_in_mag = cfg.data_paths["data_nii"]
    denoise_in_ph  = cfg.data_paths.get("data_phase", None)

    if getattr(cfg, "run_denoise", getattr(cfg, "RUN_DENOISE", False)):
        dn = pe.Node(Denoise(), name="denoise")
        dn.inputs.denoise_sh = getattr(cfg, "denoise_sh", getattr(cfg, "DENOISE_SH", None))
        dn.inputs.meth = getattr(cfg, "denoise_method", getattr(cfg, "DENOISE_METHOD", "NORDIC"))
        dn.inputs.name = getattr(cfg, "denoise_name", getattr(cfg, "DENOISE_NAME", "data"))
        dn.inputs.oPath = denoise_dir

        # Complex if we actually have a phase file, else magnitude
        if denoise_in_ph and os.path.isfile(denoise_in_ph):
            dn.inputs.domain = "complex"
        else:
            dn.inputs.domain = "mag"

        # Wire inputs: for SSFP use the on-boarded copy, else direct from config
        if acqtype == "ssfp":
            wf.connect(img_node, "data_file", dn, "magn")
            if denoise_in_ph and os.path.isfile(denoise_in_ph):
                dn.inputs.phase = denoise_in_ph
        else:
            dn.inputs.magn = denoise_in_mag
            if denoise_in_ph and os.path.isfile(denoise_in_ph):
                dn.inputs.phase = denoise_in_ph

        # 3) Gibbs unringing — feed from denoise output
        gibbs_node = pe.Node(Gibbs(), name="gibbs")
        gibbs_node.inputs.outdir = gibbs_dir
        gibbs_node.inputs.run_gibbs = bool(getattr(cfg, "run_gibbs", getattr(cfg, "RUN_GIBBS", True)))
        gibbs_node.inputs.use_docker = bool(getattr(cfg, "use_docker", getattr(cfg, "USE_DOCKER", True)))
        gibbs_node.inputs.container_runtime = getattr(cfg, "container_runtime", getattr(cfg, "CONTAINER_RUNTIME", "docker"))
        gibbs_node.inputs.interactive_tty = bool(getattr(cfg, "interactive_tty", getattr(cfg, "INTERACTIVE_TTY", True)))
        gibbs_node.inputs.gibbs_image = getattr(cfg, "gibbs_image", getattr(cfg, "GIBBS_IMAGE", "docker.io/mrtrix3/mrtrix3:3.0.7"))
        wf.connect(dn, "out_nii", gibbs_node, "data_nii")
    else:
        # 3) Gibbs unringing — feed from raw (SSFP copy or SE direct)
        gibbs_node = pe.Node(Gibbs(), name="gibbs")
        gibbs_node.inputs.outdir = gibbs_dir
        gibbs_node.inputs.run_gibbs = bool(getattr(cfg, "run_gibbs", getattr(cfg, "RUN_GIBBS", True)))
        gibbs_node.inputs.use_docker = bool(getattr(cfg, "use_docker", getattr(cfg, "USE_DOCKER", True)))
        gibbs_node.inputs.container_runtime = getattr(cfg, "container_runtime", getattr(cfg, "CONTAINER_RUNTIME", "docker"))
        gibbs_node.inputs.interactive_tty = bool(getattr(cfg, "interactive_tty", getattr(cfg, "INTERACTIVE_TTY", True)))
        gibbs_node.inputs.gibbs_image = getattr(cfg, "gibbs_image", getattr(cfg, "GIBBS_IMAGE", "docker.io/mrtrix3/mrtrix3:3.0.7"))
        if acqtype == "ssfp":
            wf.connect(img_node, "data_file", gibbs_node, "data_nii")
        else:
            gibbs_node.inputs.data_nii = cfg.data_paths["data_nii"]
    
    wf.connect(grads_src, "bvals_file", gibbs_node, "bvals")
    wf.connect(grads_src, "bvecs_file", gibbs_node, "bvecs")

    # 4) Brain mask
    mean_b0_path = os.path.join(datadir, "mean_b0.nii.gz")
    mask_path = os.path.join(datadir, "nodif_brain_mask.nii.gz")

    if getattr(cfg, "brain_mask", None) and os.path.isfile(cfg.brain_mask):
        def _copy_mask_and_meanb0(input_mask, out_mask, imain, out_meanb0, fslbin):
            import os, shutil, subprocess
            os.makedirs(os.path.dirname(out_mask), exist_ok=True)
            shutil.copy2(input_mask, out_mask)
            if isinstance(imain, (list, tuple)):
                imain = imain[0]
            subprocess.run(
                [
                    os.path.join(fslbin, "fslmaths"),
                    os.path.abspath(imain),
                    "-Tmean",
                    os.path.abspath(out_meanb0),
                ],
                check=True,
            )
            return os.path.abspath(out_meanb0), os.path.abspath(out_mask)

        mask_node = pe.Node(
            util.Function(
                input_names=["input_mask", "out_mask", "imain", "out_meanb0", "fslbin"],
                output_names=["mean_b0", "mask"],
                function=_copy_mask_and_meanb0,
            ),
            name="brain_mask_passthrough",
        )
        mask_node.inputs.input_mask = os.path.abspath(cfg.brain_mask)
        mask_node.inputs.out_mask = mask_path
        mask_node.inputs.out_meanb0 = mean_b0_path
        mask_node.inputs.fslbin = os.path.join(os.environ.get("FSLDIR", "/usr/local/fsl"), "bin")
        wf.connect(gibbs_node, "out_file", mask_node, "imain")
    else:
        mask_node = pe.Node(
            BrainMask(threshold=float(getattr(cfg, "brain_mask_thr", getattr(cfg, "BRAIN_MASK_THR", 50.0)))),
            name="brain_mask",
        )
        wf.connect(gibbs_node, "out_file", mask_node, "data_nii")
        wf.connect(grads_src, "bvals_file", mask_node, "bvals")
        mask_node.inputs.out_meanb0 = mean_b0_path
        mask_node.inputs.out_mask = mask_path

    # 5) Temporal drift correction
    drift_node = pe.Node(Drift(), name="drift")
    drift_node.inputs.script = getattr(cfg, "drift_py", getattr(cfg, "DRIFT_PY", None))
    drift_node.inputs.model = "single"
    drift_node.inputs.outdir = os.path.join(proc_dir, "drift_correction")
    drift_node.inputs.output = ["data_corrected.nii.gz"]
    std_mask = getattr(cfg, "std_mask_file", None)
    if std_mask and os.path.isfile(std_mask):
        drift_node.inputs.mask = std_mask

    std_vent = getattr(cfg, "std_ventricles_file", None)
    if std_vent and os.path.isfile(std_vent):
        drift_node.inputs.vent = std_vent

    std_ref = getattr(cfg, "std_ref_file", None)
    if std_ref and os.path.isfile(std_ref):
        drift_node.inputs.std = std_ref

    wf.connect(gibbs_node, "out_file", drift_node, "input")
    wf.connect(grads_src, "bvals_file", drift_node, "bvals")

    # 6) Spatial drift
    spdrift_dir = os.path.join(proc_dir, "data_spatial_drift")
    spdrift_out = os.path.join(spdrift_dir, "data_reg.nii.gz")
    if os.path.isfile(spdrift_out):
        print("[pipeline] Detected existing spatial-drift output, reusing and skipping spatial drift.")
        # Passthrough node so downstream nodes still see an "out_file"
        spdrift_node = pe.Node(util.IdentityInterface(fields=["out_file"]),name="spatial_drift_passthrough")
        spdrift_node.inputs.out_file = spdrift_out
    else:
        spdrift_node = pe.Node(SpatialDrift(), name="spatial_drift")
        spdrift_node.inputs.outdir = spdrift_dir
        spdrift_node.inputs.enable = bool(getattr(cfg, "run_spatial_drift", getattr(cfg, "RUN_SPATIAL_DRIFT", True)))
        sel_drift_out = pe.Node(util.Select(index=0), name="select_drift_out")
        wf.connect(drift_node, "out_files", sel_drift_out, "inlist")
        wf.connect(sel_drift_out, "out", spdrift_node, "imain")

    # 6b) Link spatial-drift output into <proc>/data/data.nii.gz
    link_data = pe.Node(
        util.Function(
            input_names=["imain", "outdir", "name"],
            output_names=["out_path"],
            function=_link_into_datadir,
        ),
        name="link_data_to_datadir",
    )
    link_data.inputs.outdir = datadir
    link_data.inputs.name = "data.nii.gz"
    wf.connect(spdrift_node, "out_file", link_data, "imain")

    # 6c) Ensure <proc>/data/bvals & bvecs exist
    def _sync_grads_local(bvals, bvecs, outdir):
        import os, shutil
        outdir = os.path.abspath(outdir)
        os.makedirs(outdir, exist_ok=True)

        def _copy_or_link_local(src, dst):
            if not src:
                return ""
            src = os.path.abspath(src)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                if os.path.islink(dst) or os.path.exists(dst):
                    os.remove(dst)
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            return os.path.abspath(dst)

        bvals_out = _copy_or_link_local(bvals, os.path.join(outdir, "bvals")) if bvals else ""
        bvecs_out = _copy_or_link_local(bvecs, os.path.join(outdir, "bvecs")) if bvecs else ""
        return bvals_out, bvecs_out

    sync_grads = pe.Node(
        util.Function(
            input_names=["bvals", "bvecs", "outdir"],
            output_names=["bvals_out", "bvecs_out"],
            function=_sync_grads_local,
        ),
        name="sync_grads_to_datadir",
    )
    sync_grads.inputs.outdir = datadir
    wf.connect(grads_src, "bvals_file", sync_grads, "bvals")
    wf.connect(grads_src, "bvecs_file", sync_grads, "bvecs")

    # 7) EDDY prep and run
    eddy_prep = pe.Node(PrepareEddy(), name="eddy_prepare")
    eddy_prep.inputs.out_root = eddy_root
    eddy_prep.inputs.echo_spacing_sec = float(getattr(cfg, "eddy_echo_sec", getattr(cfg, "EDDY_ECHO_SEC", 0.0)))
    wf.connect(spdrift_node, "out_file", eddy_prep, "imain")
    wf.connect(grads_src, "bvals_file", eddy_prep, "bvals")
    wf.connect(grads_src, "bvecs_file", eddy_prep, "bvecs")

    eddy_run = pe.Node(EddyRun(), name="eddy_run")
    eddy_run.inputs.no_gpu = bool(getattr(cfg, "no_gpu", getattr(cfg, "NO_GPU", False)))
    eddy_run.inputs.enable = bool(getattr(cfg, "run_eddy", getattr(cfg, "RUN_EDDY", True)))
    eddy_run.inputs.out = os.path.join(eddy_root, "eddy")
    wf.connect(eddy_prep, "imain_out", eddy_run, "imain")
    wf.connect(mask_node, "mask", eddy_run, "mask")
    wf.connect(eddy_prep, "index", eddy_run, "index")
    wf.connect(eddy_prep, "acqparams", eddy_run, "acqp")
    wf.connect(eddy_prep, "bvals_out", eddy_run, "bvals")
    wf.connect(eddy_prep, "bvecs_out", eddy_run, "bvecs")

    eddy_qc = pe.Node(EddyQuad(), name="eddy_qc")
    eddy_qc.inputs.enable = bool(getattr(cfg, "run_eddy_qc", getattr(cfg, "RUN_EDDY_QC", True)))
    eddy_qc.inputs.base = os.path.join(eddy_root, "eddy")
    wf.connect(eddy_prep, "index", eddy_qc, "idx")
    wf.connect(eddy_prep, "acqparams", eddy_qc, "acqp")
    wf.connect(mask_node, "mask", eddy_qc, "mask")
    wf.connect(eddy_prep, "bvals_out", eddy_qc, "bvals")

    # If EDDY/EDDY_QC are already present, skip them
    eddy_base = os.path.join(eddy_root, "eddy")
    if _eddy_outputs_exist(eddy_base):
        print("[pipeline] Detected existing EDDY outputs, skipping EDDY run")
        eddy_run.inputs.enable = False
    if _eddy_qc_done(eddy_base):
        print("[pipeline] Detected existing EDDY_QC outputs, skipping EDDY_QC")
        eddy_qc.inputs.enable = False

    # 8) CFP/FixP manifests for SSFP
    if acqtype == "ssfp":
        paths_node = pe.Node(WritePathsFile(), name="write_CFP_FixP")
        ls_dir = os.path.join(proc_dir, "data_lowershell")
        os.makedirs(ls_dir, exist_ok=True)
        paths_node.inputs.lowershell_dir = ls_dir
        paths_node.inputs.data_dir = datadir
        paths_node.inputs.sidecars_dir = sidecarsdir
        wf.connect(mask_node, "mean_b0", paths_node, "mean_b0")
        wf.connect(img_node, "t1_file", paths_node, "t1_map")
        wf.connect(img_node, "t2_file", paths_node, "t2_map")
        wf.connect(img_node, "b1_file", paths_node, "b1_map")
    else:
        paths_node = None

    # 9) Diffusion tensor modelling
    dti_node = pe.Node(DiffTensor(), name="difftensor")
    dti_node.inputs.subject_dir = datadir
    dti_node.inputs.approx_b = int(getattr(cfg, "dti_approx_b", getattr(cfg, "DTI_APPROX_B", 1000)))
    dti_node.inputs.db = int(getattr(cfg, "dti_db", getattr(cfg, "DTI_DB", 60)))
    dti_node.inputs.b0 = int(getattr(cfg, "dti_b0", getattr(cfg, "DTI_B0", 0)))
    if acqtype == "ssfp":
        dti_node.inputs.sidecars_dir = sidecarsdir
        dti_node.inputs.njobs = int(getattr(cfg, "cudimot_njobs", getattr(cfg, "CUDIMOT_NJOBS", 4)))
        dti_node.inputs.run_mcmc = bool(getattr(cfg, "cudimot_run_mcmc", getattr(cfg, "CUDIMOT_RUN_MCMC", True)))
        dti_node.inputs.no_levmar = bool(getattr(cfg, "cudimot_no_levmar", getattr(cfg, "CUDIMOT_NO_LEVMAR", True)))
        dti_node.inputs.burnin = int(getattr(cfg, "cudimot_burnin", getattr(cfg, "CUDIMOT_BURNIN", 10000)))
        dti_node.inputs.jumps = int(getattr(cfg, "cudimot_jumps", getattr(cfg, "CUDIMOT_JUMPS", 1250)))
        dti_node.inputs.sample_every = int(getattr(cfg, "cudimot_sample_every", getattr(cfg, "CUDIMOT_SAMPLE_EVERY", 25)))
    wf.connect(spdrift_node, "out_file", dti_node, "imain")
    wf.connect(grads_src, "bvals_file", dti_node, "bvals")
    wf.connect(grads_src, "bvecs_file", dti_node, "bvecs")
    wf.connect(mask_node, "mask", dti_node, "mask")
    if acqtype == "ssfp" and paths_node is not None:
        wf.connect(paths_node, "FixP", dti_node, "fixp_path")

    # 10) Registration to STD (FNIRT or MMORF)
    reg_method = (getattr(cfg, "reg_method", None) or getattr(cfg, "REG_METHOD", "FNIRT")).strip().lower()
    if reg_method == "fnirt":
        std_atl_fa = getattr(cfg, "std_fa_file", None)
        if not (std_atl_fa and os.path.isfile(std_atl_fa)):
            raise FileNotFoundError(
                f"[FNIRT] Missing standard FA template (std_fa_file): {std_atl_fa}"
            )
        stdreg = pe.Node(Reg2Std(), name="reg2std")
        stdreg.inputs.atl    = std_atl_fa
        stdreg.inputs.outdir = stdreg_dir
        wf.connect(dti_node, "fa_file", stdreg, "fa")
    elif reg_method == "mmorf":
        # MMORF registration targets: use cfg.std_* targets (mode-aware defaults set in Config)
        std_fa = getattr(cfg, "std_fa_file", None)
        std_tn = getattr(cfg, "std_tensor_file", None)
        cfg_tpl = _resolve_file(
            getattr(cfg, "mmorf_config_template", None) or getattr(cfg, "MMORF_CONFIG_TEMPLATE", None),
            "mmorf_config_template.ini",
            FILES_DIR,
        )
        stdreg = pe.Node(MMORFReg(), name="mmorf_reg")
        stdreg.inputs.outdir = stdreg_dir
        stdreg.inputs.std_ref_fa = std_fa
        stdreg.inputs.std_ref_tensor = std_tn
        stdreg.inputs.config_template = cfg_tpl
        stdreg.inputs.mode = mode
        wf.connect(dti_node, "fa_file", stdreg, "fa")
        wf.connect(dti_node, "tensor_file", stdreg, "tensor")
    else:
        raise ValueError(f"Unknown reg_method: {reg_method}")

    # 11) Fibre + XTRACT (only in FULL mode)
    xtract_node = None
    if run_mode == "full":
        if acqtype == "ssfp":
            fibre_deps = pe.Node(util.Merge(5), name="fibre_deps")
            wf.connect(link_data, "out_path", fibre_deps, "in1")
            wf.connect(paths_node, "CFP", fibre_deps, "in2")
            wf.connect(paths_node, "FixP", fibre_deps, "in3")
            wf.connect(mask_node, "mask", fibre_deps, "in4")
            wf.connect(dti_node, "fa_file", fibre_deps, "in5")

            fibre_node = pe.Node(FibreCudimot(), name="fibre_cudimot")
            fibre_node.inputs.subject_dir = datadir
            fibre_node.inputs.pipeline_cmd = getattr(cfg, "fibre_pipeline_cmd", getattr(cfg, "FIBRE_PIPELINE_CMD", "Pipeline_Ball_3_Sticks_1Shell_SSFP.sh"))
            fibre_node.inputs.init_method = getattr(cfg, "fibre_init_method", getattr(cfg, "FIBRE_INIT_METHOD", ""))
            fibre_node.inputs.queue = getattr(cfg, "fibre_queue", getattr(cfg, "FIBRE_QUEUE", ""))
            fibre_node.inputs.njobs = int(getattr(cfg, "fibre_njobs", getattr(cfg, "FIBRE_NJOBS", 4)))
            fibre_node.inputs.run_mcmc = bool(getattr(cfg, "fibre_run_mcmc", getattr(cfg, "FIBRE_RUN_MCMC", True)))
            fibre_node.inputs.no_levmar = bool(getattr(cfg, "fibre_no_levmar", getattr(cfg, "FIBRE_NO_LEVMAR", True)))
            fibre_node.inputs.burnin = int(getattr(cfg, "fibre_burnin", getattr(cfg, "FIBRE_BURNIN", 10000)))
            fibre_node.inputs.jumps = int(getattr(cfg, "fibre_jumps", getattr(cfg, "FIBRE_JUMPS", 1250)))
            fibre_node.inputs.sample_every = int(getattr(cfg, "fibre_sample_every", getattr(cfg, "FIBRE_SAMPLE_EVERY", 25)))
            wf.connect(fibre_deps, "out", fibre_node, "deps")

            xtract_node = pe.Node(Xtract(), name="xtract")
            xtract_node.inputs.outdir = xtract_out
            xtract_node.inputs.mode = mode
            if mode == "hum":
                xtract_node.inputs.stdref = getattr(cfg, "xtract_stdref_hum", None)
            else:
                xtract_node.inputs.stdref = getattr(cfg, "std_ref_file", None)
            xtract_node.inputs.species = getattr(cfg, "xtract_species", getattr(cfg, "XTRACT_SPECIES", "CUSTOM"),)
            xtract_node.inputs.mac_ext = getattr(cfg, "xtract_mac_ext", getattr(cfg, "xtract_mac_ext_default", "_NMT"),)
            xtract_node.inputs.no_gpu = bool(getattr(cfg, "xtract_no_gpu", getattr(cfg, "XTRACT_NO_GPU", False)))
            xtract_node.inputs.ptx_steplength = float(getattr(cfg, "xtract_ptx_steplength", getattr(cfg, "XTRACT_PTX_STEPLENGTH", 0.15)))
            xtract_node.inputs.profiles_dir = getattr(cfg, "xtract_profiles_dir", getattr(cfg, "XTRACT_PROFILES_DIR", ""),)
            # Viewer
            xtract_node.inputs.viewer_script = pkg_file("nodes", "tract_viewer.py")
            xtract_node.inputs.do_viewer = cfg.DO_VIEWER
            wf.connect(fibre_node, "cudimot_dir", xtract_node, "cudimot_dir")

        else:
            bpx_node = pe.Node(BedpostX(), name="bedpostx")
            bpx_node.inputs.data_dir = datadir
            wf.connect(link_data, "out_path", bpx_node, "trigger")

            xtract_node = pe.Node(Xtract(), name="xtract")
            xtract_node.inputs.outdir = xtract_out
            xtract_node.inputs.mode = mode
            if mode == "hum":
                xtract_node.inputs.stdref = getattr(cfg, "xtract_stdref_hum", None)
            else:
                xtract_node.inputs.stdref = getattr(cfg, "std_ref_file", None)
            xtract_node.inputs.species = getattr(cfg, "xtract_species", getattr(cfg, "XTRACT_SPECIES", "CUSTOM"),)
            xtract_node.inputs.mac_ext = getattr(cfg, "xtract_mac_ext", getattr(cfg, "xtract_mac_ext_default", "_NMT"),)
            xtract_node.inputs.no_gpu = bool(getattr(cfg, "xtract_no_gpu", getattr(cfg, "XTRACT_NO_GPU", False)))
            xtract_node.inputs.ptx_steplength = float(getattr(cfg, "xtract_ptx_steplength", getattr(cfg, "XTRACT_PTX_STEPLENGTH", 0.15)))
            xtract_node.inputs.profiles_dir = getattr(cfg, "xtract_profiles_dir", getattr(cfg, "XTRACT_PROFILES_DIR", ""),)
            # Viewer
            xtract_node.inputs.viewer_script = pkg_file("nodes", "tract_viewer.py")
            xtract_node.inputs.do_viewer = cfg.DO_VIEWER
            wf.connect(bpx_node, "bpx_dir", xtract_node, "bpx_dir")

        # Warps into XTRACT in FULL mode
        if reg_method == "fnirt":
            wf.connect(stdreg, "warp_std2anat", xtract_node, "warp_std2anat")
            wf.connect(stdreg, "warp_anat2std", xtract_node, "warp_anat2std")
        else:
            wf.connect(stdreg, "combined_warp_inv", xtract_node, "warp_std2anat")
            wf.connect(stdreg, "combined_warp", xtract_node, "warp_anat2std")

        # XTRACT viewer hook
        xtract_node.inputs.viewer_script = pkg_file("nodes", "tract_viewer.py")
        xtract_node.inputs.do_viewer = bool(getattr(cfg, "do_viewer", True))

    # 12) QA report (always; tolerant of missing files)
    qa = pe.Node(QAReport(), name="qa_report")
    qa.inputs.dmri_root = dmri_root
    qa.inputs.outdir = qa_outdir
    qa.inputs.pipeline_yaml = os.path.join(proc_dir, "config_used.yaml")
    qa.inputs.run_mode = run_mode  # 'preproc_only' | 'tract_only' | 'full'

    qa.inputs.nipype_dir_preproc = log_dir
    qa.inputs.nipype_graph_preproc = graph_png

    qa.inputs.eddy_qc_json = os.path.join(eddy_root, "eddy.qc", "qc.json")
    qa.inputs.std_mask_img = getattr(cfg, "std_mask_file", None)  # generic standard outline mask

    # Diffusion inputs to QA
    qa.inputs.data_4d = os.path.join(eddy_root, "eddy.nii.gz")
    qa.inputs.bvals = os.path.join(eddy_root, "bvals")
    qa.inputs.bvecs = os.path.join(eddy_root, "eddy.eddy_rotated_bvecs")
    qa.inputs.mask_file = os.path.join(datadir, "nodif_brain_mask.nii.gz")

    # Native FA / V1
    wf.connect(dti_node, "fa_file", qa, "fa_img")
    v1_native = os.path.join(datadir, "dti", "dti_V1.nii.gz")
    qa.inputs.v1_img = v1_native

    # XTRACT dir if built
    if xtract_node is not None:
        wf.connect(xtract_node, "outdir", qa, "xtract_dir")

    # Force QA late using robust sockets
    qa_deps = pe.Node(util.Merge(3), name="qa_deps")
    wf.connect(dti_node, "fa_file", qa_deps, "in1")

    # Use known-present warp outputs just to enforce ordering
    if reg_method == "fnirt":
        wf.connect(stdreg, "warp_anat2std", qa_deps, "in2")
    else:  # MMORF
        wf.connect(stdreg, "combined_warp", qa_deps, "in2")

    # Use eddy_prep's bvals_out (always there) to avoid fragile EddyRun outputs
    wf.connect(eddy_prep, "bvals_out", qa_deps, "in3")
    wf.connect(qa_deps, "out", qa, "deps")

    return wf
