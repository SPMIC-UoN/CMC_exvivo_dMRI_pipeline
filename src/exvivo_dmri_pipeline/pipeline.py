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
    # proc_dir = os.path.join(dmri_root, "proc")
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

    # Keep a copy of YAML used
    if cfg.source_yaml and os.path.isfile(cfg.source_yaml):
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
    nmt_outdir = os.path.join(proc_dir, "NMTreg")
    xtract_out = os.path.join(proc_dir, "xtract")
    qa_outdir = os.path.join(proc_dir, "QA")

    # Warp paths (if relying on on-disk)
    warp_std2anat_fn = os.path.join(nmt_outdir, "warp_std2anat.nii.gz")
    warp_anat2std_fn = os.path.join(nmt_outdir, "warp_anat2std.nii.gz")
    mmorf_combined = os.path.join(nmt_outdir, "combined_warp.nii.gz")
    mmorf_combined_inv = os.path.join(nmt_outdir, "combined_warp_inv.nii.gz")

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

        # Warps
        reg_method = getattr(cfg, "REG_METHOD", "FNIRT").upper().strip()
        if reg_method == "FNIRT":
            _require_files([warp_std2anat_fn, warp_anat2std_fn], "FNIRT warps")
            warp_std2anat = warp_std2anat_fn
            warp_anat2std = warp_anat2std_fn
        elif reg_method == "MMORF":
            _require_files([mmorf_combined, mmorf_combined_inv], "MMORF warps")
            warp_std2anat = mmorf_combined_inv
            warp_anat2std = mmorf_combined
        else:
            raise ValueError(f"Unknown REG_METHOD: {reg_method}")

        # SSFP or SE tractography path
        if acqtype == "ssfp":
            fibre_deps = pe.Node(util.Merge(2), name="fibre_deps_tractonly")
            fibre_deps.inputs.in1 = mask_file
            fa_std_guess = os.path.join(nmt_outdir, "dti_FA_in_NMT.nii.gz")
            fibre_deps.inputs.in2 = fa_std_guess if os.path.isfile(fa_std_guess) else ""

            fibre_node = pe.Node(FibreCudimot(), name="fibre_cudimot")
            fibre_node.inputs.subject_dir = datadir
            fibre_node.inputs.pipeline_cmd = cfg.FIBRE_PIPELINE_CMD
            fibre_node.inputs.init_method = cfg.FIBRE_INIT_METHOD
            fibre_node.inputs.queue = cfg.FIBRE_QUEUE
            fibre_node.inputs.njobs = cfg.FIBRE_NJOBS
            fibre_node.inputs.run_mcmc = cfg.FIBRE_RUN_MCMC
            fibre_node.inputs.no_levmar = cfg.FIBRE_NO_LEVMAR
            fibre_node.inputs.burnin = cfg.FIBRE_BURNIN
            fibre_node.inputs.jumps = cfg.FIBRE_JUMPS
            fibre_node.inputs.sample_every = cfg.FIBRE_SAMPLE_EVERY
            wf.connect(fibre_deps, "out", fibre_node, "deps")

            xtract_node = pe.Node(Xtract(), name="xtract")
            xtract_node.inputs.outdir = xtract_out
            xtract_node.inputs.stdref = cfg.std_ref_file
            xtract_node.inputs.species = cfg.XTRACT_SPECIES
            xtract_node.inputs.no_gpu = cfg.XTRACT_NO_GPU
            xtract_node.inputs.ptx_steplength = cfg.XTRACT_PTX_STEPLEN
            xtract_node.inputs.profiles_dir = cfg.XTRACT_PROFILES_DIR
            wf.connect(fibre_node, "cudimot_dir", xtract_node, "cudimot_dir")

        else:
            bpx_node = pe.Node(BedpostX(), name="bedpostx")
            bpx_node.inputs.data_dir = datadir

            xtract_node = pe.Node(Xtract(), name="xtract")
            xtract_node.inputs.outdir = xtract_out
            xtract_node.inputs.stdref = cfg.std_ref_file
            xtract_node.inputs.species = cfg.XTRACT_SPECIES
            xtract_node.inputs.no_gpu = cfg.XTRACT_NO_GPU
            xtract_node.inputs.ptx_steplength = cfg.XTRACT_PTX_STEPLEN
            xtract_node.inputs.profiles_dir = cfg.XTRACT_PROFILES_DIR

            wf.connect(bpx_node, "bpx_dir", xtract_node, "bpx_dir")

        # Warps into XTRACT
        xtract_node.inputs.warp_std2anat = warp_std2anat
        xtract_node.inputs.warp_anat2std = warp_anat2std

        # Viewer
        xtract_node.inputs.viewer_script = pkg_file("nodes", "tract_viewer.py")
        xtract_node.inputs.do_viewer = bool(getattr(cfg, "DO_VIEWER", True))

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
        qa.inputs.nmt_mask_img = pkg_file("files", "NMT_v2.0_sym_05mm_LR_brainmask.nii.gz")
        qa.inputs.nmt_fa_img = pkg_file("files", "NMT_v2.0_sym_05mm_FA.nii.gz")

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
        sc_node.inputs.TRs = cfg.sidecars.get("TRs", "")
        sc_node.inputs.diffGradAmps = cfg.sidecars.get("diffGradAmps", "")
        sc_node.inputs.diffGradDurs = cfg.sidecars.get("diffGradDurs", "")
        sc_node.inputs.flipAngles = cfg.sidecars.get("flipAngles", "")
        sc_node.inputs.flipMask = cfg.sidecars.get("flipMask", "")
        sc_node.inputs.noisefloor = cfg.sidecars.get("noisefloor", "")
        sc_node.inputs.whichshell = cfg.sidecars.get("whichshell", "")
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
        img_node.inputs.T1map = cfg.maps["T1map"]
        img_node.inputs.T2map = cfg.maps["T2map"]
        img_node.inputs.B1map = cfg.maps["B1map"]
        img_node.inputs.outdir = data_rawdir
        if cfg.brain_mask and os.path.isfile(cfg.brain_mask):
            img_node.inputs.brain_mask = cfg.brain_mask
    else:
        img_node = None  # SE uses provided data_nii directly

    # 2b) Optional denoising (immediately before Gibbs)
    # Decide source magnitude + phase and domain
    denoise_in_mag = cfg.data_paths["data_nii"]
    denoise_in_ph  = cfg.data_paths.get("data_phase", None)
    if acqtype == "ssfp":
        # Use the copy under data_raw as source to keep everything in <proc>
        # We connect to the *output* of copy_images when available.
        pass

    if cfg.RUN_DENOISE:
        dn = pe.Node(Denoise(), name="denoise")
        dn.inputs.denoise_sh = cfg.DENOISE_SH
        dn.inputs.meth = cfg.DENOISE_METHOD
        dn.inputs.name = cfg.DENOISE_NAME
        dn.inputs.oPath = denoise_dir

        # Complex if we actually have a phase file, else magnitude
        if denoise_in_ph and os.path.isfile(denoise_in_ph):
            dn.inputs.domain = "complex"
        else:
            dn.inputs.domain = "mag"

        # Wire inputs: for SSFP use the on-boarded copy, else direct from config
        if acqtype == "ssfp":
            # magn from CopyImagesToRaw output; phase from cfg (optional)
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
        gibbs_node.inputs.run_gibbs = cfg.RUN_GIBBS
        gibbs_node.inputs.use_docker = cfg.USE_DOCKER
        gibbs_node.inputs.container_runtime = cfg.CONTAINER_RUNTIME
        gibbs_node.inputs.interactive_tty = cfg.INTERACTIVE_TTY
        gibbs_node.inputs.gibbs_image = cfg.GIBBS_IMAGE
        wf.connect(dn, "out_nii", gibbs_node, "data_nii")
    else:
        # 3) Gibbs unringing — feed from raw (SSFP copy or SE direct)
        gibbs_node = pe.Node(Gibbs(), name="gibbs")
        gibbs_node.inputs.outdir = gibbs_dir
        gibbs_node.inputs.run_gibbs = cfg.RUN_GIBBS
        gibbs_node.inputs.use_docker = cfg.USE_DOCKER
        gibbs_node.inputs.container_runtime = cfg.CONTAINER_RUNTIME
        gibbs_node.inputs.interactive_tty = cfg.INTERACTIVE_TTY
        gibbs_node.inputs.gibbs_image = cfg.GIBBS_IMAGE
        if acqtype == "ssfp":
            wf.connect(img_node, "data_file", gibbs_node, "data_nii")
        else:
            gibbs_node.inputs.data_nii = cfg.data_paths["data_nii"]
    
    wf.connect(grads_src, "bvals_file", gibbs_node, "bvals")
    wf.connect(grads_src, "bvecs_file", gibbs_node, "bvecs")

    # 4) Brain mask
    mean_b0_path = os.path.join(datadir, "mean_b0.nii.gz")
    mask_path = os.path.join(datadir, "nodif_brain_mask.nii.gz")

    if cfg.brain_mask and os.path.isfile(cfg.brain_mask):
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
            BrainMask(threshold=getattr(cfg, "brain_mask_thr", 50.0)),
            name="brain_mask",
        )
        wf.connect(gibbs_node, "out_file", mask_node, "data_nii")
        wf.connect(grads_src, "bvals_file", mask_node, "bvals")
        mask_node.inputs.out_meanb0 = mean_b0_path
        mask_node.inputs.out_mask = mask_path

    # 5) Temporal drift correction
    drift_node = pe.Node(Drift(), name="drift")
    drift_node.inputs.script = cfg.drift_py
    drift_node.inputs.model = "single"
    drift_node.inputs.outdir = os.path.join(proc_dir, "drift_correction")
    drift_node.inputs.output = ["data_corrected.nii.gz"]
    drift_node.inputs.mask = cfg.std_mask_file
    drift_node.inputs.vent = cfg.std_ventricles_file
    drift_node.inputs.std = cfg.std_ref_file
    wf.connect(gibbs_node, "out_file", drift_node, "input")
    wf.connect(grads_src, "bvals_file", drift_node, "bvals")

    # 6) Spatial drift
    spdrift_node = pe.Node(SpatialDrift(), name="spatial_drift")
    spdrift_node.inputs.outdir = os.path.join(proc_dir, "data_spatial_drift")
    spdrift_node.inputs.enable = cfg.RUN_SPATIAL_DRIFT
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
    eddy_prep.inputs.echo_spacing_sec = float(cfg.EDDY_ECHO_SEC)
    wf.connect(spdrift_node, "out_file", eddy_prep, "imain")
    wf.connect(grads_src, "bvals_file", eddy_prep, "bvals")
    wf.connect(grads_src, "bvecs_file", eddy_prep, "bvecs")

    eddy_run = pe.Node(EddyRun(), name="eddy_run")
    eddy_run.inputs.no_gpu = cfg.NO_GPU
    eddy_run.inputs.enable = cfg.RUN_EDDY
    eddy_run.inputs.out = os.path.join(eddy_root, "eddy")
    wf.connect(eddy_prep, "imain_out", eddy_run, "imain")
    wf.connect(mask_node, "mask", eddy_run, "mask")
    wf.connect(eddy_prep, "index", eddy_run, "index")
    wf.connect(eddy_prep, "acqparams", eddy_run, "acqp")
    wf.connect(eddy_prep, "bvals_out", eddy_run, "bvals")
    wf.connect(eddy_prep, "bvecs_out", eddy_run, "bvecs")

    eddy_qc = pe.Node(EddyQuad(), name="eddy_qc")
    eddy_qc.inputs.enable = cfg.RUN_EDDY_QC
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
    dti_node.inputs.approx_b = cfg.DTI_APPROX_B
    dti_node.inputs.db = cfg.DTI_DB
    dti_node.inputs.b0 = cfg.DTI_B0
    if acqtype == "ssfp":
        dti_node.inputs.sidecars_dir = sidecarsdir
        dti_node.inputs.njobs = cfg.CUDIMOT_NJOBS
        dti_node.inputs.run_mcmc = cfg.CUDIMOT_RUN_MCMC
        dti_node.inputs.no_levmar = cfg.CUDIMOT_NO_LEVMAR
        dti_node.inputs.burnin = cfg.CUDIMOT_BURNIN
        dti_node.inputs.jumps = cfg.CUDIMOT_JUMPS
        dti_node.inputs.sample_every = cfg.CUDIMOT_SAMPLE_EVERY
    wf.connect(spdrift_node, "out_file", dti_node, "imain")
    wf.connect(grads_src, "bvals_file", dti_node, "bvals")
    wf.connect(grads_src, "bvecs_file", dti_node, "bvecs")
    wf.connect(mask_node, "mask", dti_node, "mask")
    if acqtype == "ssfp" and paths_node is not None:
        wf.connect(paths_node, "FixP", dti_node, "fixp_path")

    # 10) Registration to NMT
    reg_method = getattr(cfg, "REG_METHOD", "FNIRT").upper().strip()
    if reg_method == "FNIRT":
        reg2std_py = _resolve_first_existing(
            [
                getattr(cfg, "REG2STD_PY", None),
                os.path.join(NODES_DIR, "reg2std.py"),
                os.path.join(FILES_DIR, "reg2std.py"),
            ]
        )
        if not reg2std_py:
            raise FileNotFoundError("[FNIRT] Could not locate reg2std.py.")
        std_atl_fa = _resolve_first_existing(
            [
                getattr(cfg, "std_ref_file", None),
                os.path.join(FILES_DIR, "NMT_FA.nii.gz"),
            ]
        )
        if not std_atl_fa:
            raise FileNotFoundError("[FNIRT] Could not locate NMT FA template.")

        nmt_reg = pe.Node(Reg2Std(), name="reg2std")
        nmt_reg.inputs.script = reg2std_py
        nmt_reg.inputs.atl = std_atl_fa
        nmt_reg.inputs.outdir = nmt_outdir
        if getattr(cfg, "REG2STD_CONFIG", None) and os.path.isfile(cfg.REG2STD_CONFIG):
            nmt_reg.inputs.config = cfg.REG2STD_CONFIG
        wf.connect(dti_node, "fa_file", nmt_reg, "fa")

    elif reg_method == "MMORF":
        std_fa = _resolve_file(getattr(cfg, "NMT_FA_FILE", None), "NMT_FA.nii.gz", FILES_DIR)
        std_tn = _resolve_file(getattr(cfg, "NMT_TENSOR_FILE", None), "NMT_tensor.nii.gz", FILES_DIR)
        cfg_tpl = _resolve_file(
            getattr(cfg, "MMORF_CONFIG_TEMPLATE", None),
            "mmorf_config_template.ini",
            FILES_DIR,
        )
        missing = []
        if not std_fa:
            missing.append("NMT FA")
        if not std_tn:
            missing.append("NMT tensor")
        if not cfg_tpl:
            missing.append("MMORF config template")
        if missing:
            raise FileNotFoundError("[MMORF] Missing: " + ", ".join(missing))

        nmt_reg = pe.Node(MMORFReg(), name="mmorf_reg")
        nmt_reg.inputs.outdir = nmt_outdir
        nmt_reg.inputs.std_ref_fa = std_fa
        nmt_reg.inputs.std_ref_tensor = std_tn
        nmt_reg.inputs.config_template = cfg_tpl
        wf.connect(dti_node, "fa_file", nmt_reg, "fa")
        wf.connect(dti_node, "tensor_file", nmt_reg, "tensor")
    else:
        raise ValueError(f"Unknown REG_METHOD: {reg_method}")

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
            fibre_node.inputs.pipeline_cmd = cfg.FIBRE_PIPELINE_CMD
            fibre_node.inputs.init_method = cfg.FIBRE_INIT_METHOD
            fibre_node.inputs.queue = cfg.FIBRE_QUEUE
            fibre_node.inputs.njobs = cfg.FIBRE_NJOBS
            fibre_node.inputs.run_mcmc = cfg.FIBRE_RUN_MCMC
            fibre_node.inputs.no_levmar = cfg.FIBRE_NO_LEVMAR
            fibre_node.inputs.burnin = cfg.FIBRE_BURNIN
            fibre_node.inputs.jumps = cfg.FIBRE_JUMPS
            fibre_node.inputs.sample_every = cfg.FIBRE_SAMPLE_EVERY
            wf.connect(fibre_deps, "out", fibre_node, "deps")

            xtract_node = pe.Node(Xtract(), name="xtract")
            xtract_node.inputs.outdir = xtract_out
            xtract_node.inputs.stdref = cfg.std_ref_file
            xtract_node.inputs.species = cfg.XTRACT_SPECIES
            xtract_node.inputs.no_gpu = cfg.XTRACT_NO_GPU
            xtract_node.inputs.ptx_steplength = cfg.XTRACT_PTX_STEPLEN
            xtract_node.inputs.profiles_dir = cfg.XTRACT_PROFILES_DIR
            wf.connect(fibre_node, "cudimot_dir", xtract_node, "cudimot_dir")

        else:
            bpx_node = pe.Node(BedpostX(), name="bedpostx")
            bpx_node.inputs.data_dir = datadir
            wf.connect(link_data, "out_path", bpx_node, "trigger")

            xtract_node = pe.Node(Xtract(), name="xtract")
            xtract_node.inputs.outdir = xtract_out
            xtract_node.inputs.stdref = cfg.std_ref_file
            xtract_node.inputs.species = cfg.XTRACT_SPECIES
            xtract_node.inputs.no_gpu = cfg.XTRACT_NO_GPU
            xtract_node.inputs.ptx_steplength = cfg.XTRACT_PTX_STEPLEN
            xtract_node.inputs.profiles_dir = cfg.XTRACT_PROFILES_DIR

            wf.connect(bpx_node, "bpx_dir", xtract_node, "bpx_dir")

        # Warps into XTRACT in FULL mode
        if reg_method == "FNIRT":
            wf.connect(nmt_reg, "warp_std2anat", xtract_node, "warp_std2anat")
            wf.connect(nmt_reg, "warp_anat2std", xtract_node, "warp_anat2std")
        else:
            wf.connect(nmt_reg, "combined_warp_inv", xtract_node, "warp_std2anat")
            wf.connect(nmt_reg, "combined_warp", xtract_node, "warp_anat2std")

        # XTRACT viewer hook
        xtract_node.inputs.viewer_script = pkg_file("nodes", "tract_viewer.py")
        xtract_node.inputs.do_viewer = bool(getattr(cfg, "DO_VIEWER", True))

    # 12) QA report (always; tolerant of missing files)
    qa = pe.Node(QAReport(), name="qa_report")
    qa.inputs.dmri_root = dmri_root
    qa.inputs.outdir = qa_outdir
    qa.inputs.pipeline_yaml = os.path.join(proc_dir, "config_used.yaml")
    qa.inputs.run_mode = run_mode  # 'preproc_only' | 'tract_only' | 'full'

    qa.inputs.nipype_dir_preproc = log_dir
    qa.inputs.nipype_graph_preproc = graph_png

    qa.inputs.eddy_qc_json = os.path.join(eddy_root, "eddy.qc", "qc.json")
    qa.inputs.nmt_mask_img = pkg_file("files", "NMT_v2.0_sym_05mm_LR_brainmask.nii.gz")
    qa.inputs.nmt_fa_img = pkg_file("files", "NMT_v2.0_sym_05mm_FA.nii.gz")

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
    if reg_method == "FNIRT":
        wf.connect(nmt_reg, "warp_anat2std", qa_deps, "in2")
    else:  # MMORF
        wf.connect(nmt_reg, "combined_warp", qa_deps, "in2")

    # Use eddy_prep's bvals_out (always there) to avoid fragile EddyRun outputs
    wf.connect(eddy_prep, "bvals_out", qa_deps, "in3")
    wf.connect(qa_deps, "out", qa, "deps")

    return wf
