# src/exvivo_dmri_pipeline/config.py
import os
from typing import Any, Dict, Optional

import yaml

from .utils import pkg_file


# -----------------------------
# helpers
# -----------------------------
def _abspath(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    p = str(p)
    if not p:
        return None
    return os.path.abspath(os.path.expanduser(p))


def _abspath_dict(d: Optional[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in (d or {}).items():
        if v:
            out[k] = _abspath(str(v))
    return out


def _lower_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k).lower(): v for k, v in d.items()}


# -----------------------------
# config
# -----------------------------
class Config:
    """
    Central configuration for the ex vivo pipeline, loaded from YAML.

    Modes:
      acqtype = "ssfp":
        - Requires: data_nii, bvals, bvecs
        - Requires sidecars: TRs, diffGradAmps, diffGradDurs, flipAngles, flipMask, noisefloor
          (optionally whichshell)
        - Requires maps: T1map, T2map, B1map

      acqtype = "se":
        - Requires: data_nii, bvals, bvecs
        - Sidecars/maps optional
    """

    def __init__(self, y: Dict[str, Any]) -> None:
        # Normalise top-level keys
        y = _lower_keys(y or {})

        # -----------------------------
        # core mode fields
        # -----------------------------
        acq_raw = str(y.get("acqtype", "ssfp") or "ssfp").strip().lower()
        if acq_raw not in {"ssfp", "se"}:
            raise RuntimeError("acqtype must be 'ssfp' or 'se'.")
        self.acqtype: str = acq_raw

        mode_raw = str(y.get("mode", "nhp") or "nhp").strip().lower()
        if mode_raw not in {"nhp", "hum"}:
            raise RuntimeError("mode must be 'nhp' or 'hum'.")
        self.MODE: str = mode_raw
        # also keep a lower-case alias for convenience / backwards compatibility
        self.mode: str = mode_raw

        # -----------------------------
        # workspace roots
        # -----------------------------
        self.dmri_root: str = y.get("dmri_root", "./proc")
        self.abs_dmri_root: str = _abspath(self.dmri_root) or os.path.abspath("./proc")
        self.proc_root: str = self.abs_dmri_root

        self.input_dir: Optional[str] = _abspath(y.get("input_dir"))

        # brain mask (optional; special strings treated as "none")
        brain_mask_raw = y.get("brain_mask")
        if isinstance(brain_mask_raw, str) and brain_mask_raw.strip().lower() in {
            "", "none", "null", "na", "nil"
        }:
            brain_mask_raw = None
        self.brain_mask: Optional[str] = _abspath(brain_mask_raw)

        # -----------------------------
        # data / maps / sidecars
        # -----------------------------
        data_paths_raw = y.get("data_paths") or {}
        maps_raw = y.get("maps") or {}
        sidecars_raw = y.get("sidecars") or {}

        if isinstance(data_paths_raw, dict):
            data_paths_raw = _lower_keys(data_paths_raw)
        if isinstance(maps_raw, dict):
            maps_raw = _lower_keys(maps_raw)
        if isinstance(sidecars_raw, dict):
            sidecars_raw = _lower_keys(sidecars_raw)

        # If a single input_dir is provided and no explicit data/maps/sidecars,
        # assume standard filenames in that directory.
        if self.input_dir and not data_paths_raw and not maps_raw and not sidecars_raw:
            root = self.input_dir

            def P(name: str) -> str:
                return os.path.join(root, name)

            data_paths_raw = {
                "data_nii": P("data.nii.gz"),
                "bvals": P("bvals"),
                "bvecs": P("bvecs"),
            }
            phase_candidate = P("data_ph.nii.gz")
            if os.path.isfile(phase_candidate):
                data_paths_raw["data_phase"] = phase_candidate

            if self.acqtype == "ssfp":
                maps_raw = {
                    "t1map": P("T1map.nii.gz"),
                    "t2map": P("T2map.nii.gz"),
                    "b1map": P("B1map.nii.gz"),
                }
                sidecars_raw = {
                    "trs": P("TRs"),
                    "diffgradamps": P("diffGradAmps"),
                    "diffgraddurs": P("diffGradDurs"),
                    "flipangles": P("flipAngles"),
                    "flipmask": P("flipMask"),
                    "noisefloor": P("noisefloor"),
                    "whichshell": P("whichshell"),
                }

        self.data_paths: Dict[str, str] = _abspath_dict(data_paths_raw)
        self.maps: Dict[str, str] = _abspath_dict(maps_raw)
        self.sidecars: Dict[str, str] = _abspath_dict(sidecars_raw)

        # Validate core diffusion inputs
        def _need(d: Dict[str, str], key: str, label: str) -> None:
            p = d.get(key)
            if not p or not os.path.isfile(p):
                raise FileNotFoundError(f"Missing required {label} '{key}': {p}")

        for k in ("data_nii", "bvals", "bvecs"):
            _need(self.data_paths, k, "data_paths")

        if self.acqtype == "ssfp":
            # SSFP maps
            for k in ("t1map", "t2map", "b1map"):
                _need(self.maps, k, "maps")

            # Required sidecars (whichshell is optional but must exist if provided)
            for k in ("trs", "diffgradamps", "diffgraddurs", "flipangles", "flipmask", "noisefloor"):
                _need(self.sidecars, k, "sidecars")

            if "whichshell" in self.sidecars:
                _need(self.sidecars, "whichshell", "sidecars")

        # -----------------------------
        # runtime toggles / eddy
        # -----------------------------
        self.no_gpu: bool = bool(y.get("no_gpu", False))
        self.NO_GPU: bool = self.no_gpu  # compatibility

        self.run_gibbs: bool = bool(y.get("run_gibbs", True))
        self.use_docker: bool = bool(y.get("use_docker", True))
        self.container_runtime: str = str(y.get("container_runtime", "docker"))
        self.interactive_tty: bool = bool(y.get("interactive_tty", True))
        self.gibbs_image: str = str(y.get("gibbs_image", "docker.io/mrtrix3/mrtrix3:3.0.7"))

        self.run_eddy: bool = bool(y.get("run_eddy", True))
        self.run_eddy_qc: bool = bool(y.get("run_eddy_qc", True))
        self.eddy_echo_sec = y.get("eddy_echo_sec")

        if (self.run_eddy or self.run_eddy_qc) and self.eddy_echo_sec is None:
            raise RuntimeError(
                "eddy_echo_sec must be set (seconds) when run_eddy or run_eddy_qc is True."
            )
        if self.eddy_echo_sec is not None:
            self.eddy_echo_sec = float(self.eddy_echo_sec)

        self.run_spatial_drift: bool = bool(y.get("run_spatial_drift", True))
        self.brain_mask_thr: float = float(y.get("brain_mask_thr", 50.0))

        # -----------------------------
        # DTI / CUDIMOT settings
        # -----------------------------
        self.dti_approx_b: int = int(y.get("dti_approx_b", 1000))
        self.dti_db: int = int(y.get("dti_db", 60))
        self.dti_b0: int = int(y.get("dti_b0", 0))

        # Script names for CUDIMOT DTI and fibre models
        # These are the ones you asked to keep explicitly in config.
        self.dti_pipeline_cmd: str = str(y.get("dti_pipeline_cmd", y.get("cudimot_cmd", "Pipeline_DTI_SSFP.sh")))
        self.cudimot_cmd: str = self.dti_pipeline_cmd  # alias

        self.cudimot_queue: Optional[str] = y.get("cudimot_queue")
        self.cudimot_njobs: int = int(y.get("cudimot_njobs", 4))
        self.cudimot_run_mcmc: bool = bool(y.get("cudimot_run_mcmc", True))
        self.cudimot_no_levmar: bool = bool(y.get("cudimot_no_levmar", True))
        self.cudimot_burnin: int = int(y.get("cudimot_burnin", 10000))
        self.cudimot_jumps: int = int(y.get("cudimot_jumps", 1250))
        self.cudimot_sample_every: int = int(y.get("cudimot_sample_every", 25))

        self.fibre_pipeline_cmd: str = str(y.get("fibre_pipeline_cmd", "Pipeline_Ball_3_Sticks_2Shells_SSFP.sh",))
        self.fibre_init_method: str = str(y.get("fibre_init_method", ""))
        self.fibre_queue: str = str(y.get("fibre_queue", ""))
        self.fibre_njobs: int = int(y.get("fibre_njobs", 4))
        self.fibre_run_mcmc: bool = bool(y.get("fibre_run_mcmc", True))
        self.fibre_no_levmar: bool = bool(y.get("fibre_no_levmar", True))
        self.fibre_burnin: int = int(y.get("fibre_burnin", 10000))
        self.fibre_jumps: int = int(y.get("fibre_jumps", 1250))
        self.fibre_sample_every: int = int(y.get("fibre_sample_every", 25))
        self.fibre_subject_dir: Optional[str] = _abspath(y.get("fibre_subject_dir"))

        # -----------------------------
        # registration
        # -----------------------------
        self.reg_method: str = str(y.get("reg_method", "mmorf") or "mmorf").strip().lower()
        if self.reg_method not in {"fnirt", "mmorf"}:
            raise RuntimeError("reg_method must be 'fnirt' or 'mmorf'.")
        self.REG_METHOD: str = self.reg_method.upper()  # for older checks

        self.reg2std_py: Optional[str] = _abspath(y.get("reg2std_py"))
        self.reg2std_config: Optional[str] = _abspath(y.get("reg2std_config"))

        # MMORF uses an INI template, but warp_res_init / FWHM will be set in the reg node
        self.mmorf_config_template: Optional[str] = _abspath(
            y.get("mmorf_config_template") or pkg_file("files", "mmorf_config_template.ini")
        )

        # -----------------------------
        # denoising (NORDIC default)
        # -----------------------------
        self.run_denoise: bool = bool(y.get("run_denoise", False))
        self.denoise_method: str = str(y.get("denoise_method", "NORDIC")).upper()
        self.denoise_sh: Optional[str] = _abspath(y.get("denoise_sh"))
        self.denoise_name: str = str(y.get("denoise_name", "data"))

        # -----------------------------
        # drift
        # -----------------------------
        self.drift_py: Optional[str] = _abspath(
            y.get("drift_py") or pkg_file("nodes", "drift.py")
        )

        # -----------------------------
        # FSL standard references
        # -----------------------------
        self.FSLDIR: str = os.environ.get("FSLDIR", "")
        if not self.FSLDIR or not os.path.isdir(self.FSLDIR):
            raise RuntimeError(
                "FSLDIR environment variable not set or invalid. Please source FSL before running."
            )

        self.fsl_std_dir: str = os.path.join(self.FSLDIR, "data", "standard")
        self.fsl_mni_t1_1mm: str = os.path.join(self.fsl_std_dir, "MNI152_T1_1mm.nii.gz")
        self.fsl_mni_vent: str = os.path.join(self.fsl_std_dir, "MNI152_T1_2mm_VentricleMask.nii.gz")
        self.fsl_hcp1065_fa_1mm: str = os.path.join(self.fsl_std_dir, "FSL_HCP1065_FA_1mm.nii.gz")
        self.fsl_hcp1065_ten_1mm: str = os.path.join(self.fsl_std_dir, "FSL_HCP1065_tensor_1mm.nii.gz")

        # NMT assets (shipped with package) for NHP
        self.nmt_fa_file: Optional[str] = _abspath(y.get("nmt_fa_file") or pkg_file("files", "NMT_FA.nii.gz"))
        self.nmt_tensor_file: Optional[str] = _abspath(y.get("nmt_tensor_file") or pkg_file("files", "NMT_tensor.nii.gz"))
        self.nmt_ventricles_file: Optional[str] = _abspath(y.get("nmt_ventricles_file") or pkg_file("files", "NMT_v2.0_sym_05mm_ventricles.nii.gz"))

        # -----------------------------
        # mode specific: registration, drift, XTRACT
        # -----------------------------
        if self.MODE == "hum":
            # Registration targets
            self.std_fa_file: Optional[str] = _abspath(y.get("std_fa_file") or self.fsl_hcp1065_fa_1mm)
            self.std_tensor_file: Optional[str] = _abspath(y.get("std_tensor_file") or self.fsl_hcp1065_ten_1mm)
            # Drift masks - default ventricles is the MNI 2 mm mask that you specified
            self.std_mask_file = _abspath(y.get("std_mask_file") or os.path.join(self.fsl_std_dir, "MNI152_T1_1mm_brain_mask.nii.gz"))
            self.std_ventricles_file: Optional[str] = _abspath(y.get("std_ventricles_file") or self.fsl_mni_vent)
            self.std_ref_file: Optional[str] = _abspath(y.get("std_ref_file") or self.std_fa_file)

            # XTRACT defaults
            self.xtract_species_default: str = "HUMAN"
            self.xtract_mac_ext_default: str = ""
            self.xtract_stdref_hum: Optional[str] = self.fsl_mni_t1_1mm
        else:
            # NHP: NMT references
            self.std_fa_file = _abspath(y.get("std_fa_file") or self.nmt_fa_file)
            self.std_tensor_file = _abspath(y.get("std_tensor_file") or self.nmt_tensor_file)
            self.std_mask_file = _abspath(y.get("std_mask_file") or pkg_file("files", "NMT_v2.0_sym_brainmask_dil.nii.gz"))
            self.std_ventricles_file = _abspath(y.get("std_ventricles_file") or self.nmt_ventricles_file)
            self.std_ref_file = _abspath(y.get("std_ref_file") or self.std_fa_file)

            self.xtract_species_default = "MACAQUE"
            self.xtract_mac_ext_default = "_NMT"
            self.xtract_stdref_hum = None

        # -----------------------------
        # XTRACT config
        # -----------------------------
        self.xtract_species: str = str(y.get("xtract_species", self.xtract_species_default) or self.xtract_species_default).upper()
        self.xtract_profiles_dir: str = (_abspath(y.get("xtract_profiles_dir")) if y.get("xtract_profiles_dir") else "")
        self.xtract_no_gpu: bool = bool(y.get("xtract_no_gpu", False))
        self.xtract_ptx_steplength: float = float(y.get("xtract_ptx_steplength", 0.15))
        # an optional override of the NMT mac_ext for NHP
        self.xtract_mac_ext: str = str(y.get("xtract_mac_ext", self.xtract_mac_ext_default))

        # -----------------------------
        # streamlines / viewer (xtract QC)
        # -----------------------------
        self.STREAMLINES_DO: bool = bool(y.get("streamlines_do", True))
        self.STREAMLINES_DENSITY_THRESHOLD: float = float(y.get("streamlines_density_threshold", 1e-3))
        self.STREAMLINES_FORMAT: str = str(y.get("streamlines_format", "trk"))
        self.STREAMLINES_PTX2_PREFIX: str = str(y.get("streamlines_ptx2_prefix", "densityNorm"))
        self.STREAMLINES_NUM_JOBS: int = int(y.get("streamlines_num_jobs", 1))
        self.DO_VIEWER: bool = bool(y.get("do_viewer", True))

        # -----------------------------
        # path to YAML (set in from_yaml)
        # -----------------------------
        self.abs_yaml_path: Optional[str] = None
        self.source_yaml: Optional[str] = None

    # -----------------------------
    # load from YAML
    # -----------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls(data)
        cfg.abs_yaml_path = os.path.abspath(path)
        cfg.source_yaml = cfg.abs_yaml_path
        return cfg

    # -----------------------------
    # repr for debugging
    # -----------------------------
    def __repr__(self) -> str:
        fields = {
            "MODE": self.MODE,
            "acqtype": self.acqtype,
            "dmri_root": self.abs_dmri_root,
            "run_eddy": self.run_eddy,
            "run_spatial_drift": self.run_spatial_drift,
            "reg_method": self.reg_method,
            "std_fa_file": self.std_fa_file,
        }
        kv = ", ".join(f"{k}={v}" for k, v in fields.items())
        return f"Config({kv})"