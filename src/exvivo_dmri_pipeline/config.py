# src/exvivo_dmri_pipeline/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Dict

import yaml

from .utils import pkg_file


# -----------------------------
# helpers
# -----------------------------
def _abs(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    return os.path.abspath(os.path.expanduser(p))


def _absify(d: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in (d or {}).items():
        if v:
            out[k] = _abs(v)
    return out


def _anykey(d: dict, key: str, default=None):
    if d is None:
        return default
    lc = key.lower()
    for k, v in d.items():
        if isinstance(k, str) and k.lower() == lc:
            return v
    return default


def _abs_or_none(p: Optional[str]) -> Optional[str]:
    if p and os.path.isfile(p):
        return os.path.abspath(p)
    return _abs(p)


# -----------------------------
# config
# -----------------------------
@dataclass
class Config:
    """
    Central configuration for the ex vivo dMRI pipeline.

    Modes:
      ssfp:
        Requires data_nii, bvals, bvecs
        Requires sidecars: TRs, diffGradAmps, diffGradDurs, flipAngles, flipMask, noisefloor
        Optional sidecar: whichshell
        Requires maps: T1map, T2map, B1map

      se:
        Requires data_nii, bvals, bvecs
        Optional brain_mask

    Denoising:
      Optional (run_denoise = False by default).
      If enabled and both magnitude + phase are present -> complex-domain denoise.
      If enabled and only magnitude present -> magnitude denoise.
      Expects a shell wrapper (denoise_sh) compatible with nodes/denoise.py.
    """

    # acquisition type
    acqtype: str = "ssfp"  # "ssfp" or "se"

    # optional single folder with standard filenames
    input_dir: Optional[str] = None

    # explicit paths
    data_paths: Dict[str, str] = field(default_factory=dict)   # keys: data_nii, bvals, bvecs, optional data_phase
    maps: Dict[str, str]       = field(default_factory=dict)   # keys: T1map, T2map, B1map
    sidecars: Dict[str, str]   = field(default_factory=dict)   # TRs, diffGradAmps, diffGradDurs, flipAngles, flipMask, noisefloor, optional whichshell

    brain_mask: Optional[str] = None
    dmri_root: Optional[str]  = None

    # runtime toggles
    no_gpu: bool = False
    brain_mask_thr: float = 50.0
    run_gibbs: bool = True
    use_docker: bool = True
    container_runtime: str = "docker"
    interactive_tty: bool = True
    gibbs_image: str = "docker.io/mrtrix3/mrtrix3:3.0.7"

    run_eddy: bool = True
    run_eddy_qc: bool = True
    eddy_echo_sec: Optional[float] = None

    run_spatial_drift: bool = True

    # drift + standard references
    drift_py: Optional[str] = None
    std_mask_file: Optional[str] = None
    std_ventricles_file: Optional[str] = None
    std_ref_file: Optional[str] = None

    # source yaml (filled in from loader)
    source_yaml: Optional[str] = None

    # DTI / CUDIMOT (QA tensor)
    dti_approx_b: int = 1000
    dti_db: int = 60
    dti_b0: int = 0

    # CUDIMOT DTI
    cudimot_cmd: str = "Pipeline_DTI_SSFP.sh"
    cudimot_queue: Optional[str] = None
    cudimot_njobs: int = 4
    cudimot_run_mcmc: bool = True
    cudimot_no_levmar: bool = True
    cudimot_burnin: int = 10000
    cudimot_jumps: int = 1250
    cudimot_sample_every: int = 25

    # registration (node chooses FNIRT or MMORF)
    reg_method: str = "FNIRT"
    reg2std_py: Optional[str] = None
    reg2std_config: Optional[str] = None

    # standard-space references for registration
    # for nhp we default to packaged NMT_* files; for hum we default to HCP1065 FA/tensor
    std_fa_file: Optional[str] = None
    std_tensor_file: Optional[str] = None

    # packaged NMT fallbacks (still exposed to allow override)
    nmt_fa_file: Optional[str] = None
    nmt_tensor_file: Optional[str] = None

    # mmorf ini template and multi-scale knobs
    mmorf_config_template: Optional[str] = None
    mmorf_warp_res_init: str = "16"
    mmorf_fwhm_set: str = "4.0 4.0 2.0 1.0 0.5 0.25 0.125"

    # fibre modelling
    fibre_pipeline_cmd: str = "Pipeline_Ball_3_Sticks_2Shells_SSFP.sh"
    fibre_init_method: str = ""
    fibre_queue: str = ""
    fibre_njobs: int = 4
    fibre_run_mcmc: bool = True
    fibre_no_levmar: bool = True
    fibre_burnin: int = 10000
    fibre_jumps: int = 1250
    fibre_sample_every: int = 25
    fibre_subject_dir: str = ""

    # XTRACT
    xtract_species: str = "CUSTOM"  # user override; falls back to *_default
    xtract_no_gpu: bool = False
    xtract_ptx_steplength: float = 0.15
    xtract_profiles_dir: str = ""

    # Default-by-mode knobs
    xtract_species_default: str = ""     # set in _finalize by mode
    xtract_mac_ext_default: str = ""     # set in _finalize by mode
    xtract_stdref_hum: Optional[str] = None  # MNI T1 1mm for human when profiles used

    # denoising
    run_denoise: bool = False
    denoise_sh: Optional[str] = None      # path to wrapper script (bash)
    denoise_method: str = "NORDIC"        # passed as -meth
    denoise_name: str = "data"            # output stem name

    # hum/nhp mode (new)
    mode: str = "nhp"  # "nhp" or "hum"
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        raw = { (k.lower() if isinstance(k, str) else k): v for k, v in raw.items() }
        import dataclasses
        allowed = {f.name for f in dataclasses.fields(cls)}
        unknown = [k for k in raw.keys() if isinstance(k, str) and k not in allowed]
        if unknown:
            print("[config] Warning: unknown config keys (ignored): " + ", ".join(sorted(unknown)))
        
        raw["acqtype"] = (raw.get("acqtype", "ssfp") or "ssfp").strip().lower()
        raw["mode"]    = (raw.get("mode", "nhp") or "nhp").strip().lower()

        cfg = cls(**raw)
        cfg.source_yaml = os.path.abspath(path)
        cfg._finalize()
        return cfg


    @property
    def abs_dmri_root(self) -> str:
        return self.dmri_root

    def _finalize(self) -> None:
        # workspace
        self.dmri_root = _abs(self.dmri_root or "./exvivo_proc")

        # acquisition type
        self.acqtype = (self.acqtype or "ssfp").strip().lower()
        if self.acqtype not in ("ssfp", "se"):
            raise ValueError("acqtype must be 'ssfp' or 'se'.")

        # optional brain mask normalization
        if isinstance(self.brain_mask, str) and self.brain_mask.strip().lower() in {"", "none", "null", "nil", "na"}:
            self.brain_mask = None

        # packaged defaults
        self.drift_py = _abs(self.drift_py or pkg_file("nodes", "drift.py"))
        # std_mask_file and std_ventricles_file will be set by mode below if not set

        # allow single-folder inference if explicit paths not provided
        if self.input_dir and not (self.data_paths or self.maps or self.sidecars):
            root = _abs(self.input_dir)

            def P(name: str) -> str:
                return os.path.join(root, name)

            self.data_paths = {
                "data_nii": P("data.nii.gz"),
                "bvals": P("bvals"),
                "bvecs": P("bvecs"),
            }
            # optional phase
            phase_cand = P("data_ph.nii.gz")
            if os.path.isfile(phase_cand):
                self.data_paths["data_phase"] = phase_cand

            # SSFP maps
            self.maps = {
                "T1map": P("T1map.nii.gz"),
                "T2map": P("T2map.nii.gz"),
                "B1map": P("B1map.nii.gz"),
            }

            # sidecars (no TE)
            self.sidecars = {
                "TRs": P("TRs"),
                "diffGradAmps": P("diffGradAmps"),
                "diffGradDurs": P("diffGradDurs"),
                "flipAngles": P("flipAngles"),
                "flipMask": P("flipMask"),
                "noisefloor": P("noisefloor"),
                "whichshell": P("whichshell"),
            }

        # absolutize
        self.data_paths = _absify(self.data_paths)
        self.maps       = _absify(self.maps)
        self.sidecars   = _absify(self.sidecars)
        self.brain_mask = _abs(self.brain_mask)

        # validate required inputs
        def _need(d: Dict[str, str], key: str):
            p = d.get(key)
            if not p or not os.path.isfile(p):
                raise FileNotFoundError(f"Missing required '{key}': {p}")

        for k in ("data_nii", "bvals", "bvecs"):
            _need(self.data_paths, k)

        if self.acqtype == "ssfp":
            for k in ("T1map", "T2map", "B1map"):
                _need(self.maps, k)

            for k in ("TRs", "diffGradAmps", "diffGradDurs", "flipAngles", "flipMask", "noisefloor"):
                _need(self.sidecars, k)

            if self.sidecars.get("whichshell"):
                _need(self.sidecars, "whichshell")
        else:
            # se: maps and sidecars not required
            self.maps = self.maps or {}
            self.sidecars = self.sidecars or {}

        # eddy echo spacing must be supplied if eddy or eddy_qc are enabled
        if (self.run_eddy or self.run_eddy_qc) and (self.eddy_echo_sec is None):
            raise ValueError("eddy_echo_sec must be set (seconds) when run_eddy or run_eddy_qc is True.")

        # normalize strings
        self.container_runtime = (self.container_runtime or "docker").strip()
        self.gibbs_image = self.gibbs_image or "docker.io/mrtrix3/mrtrix3:3.0.7"

        # registration method
        if self.reg_method.upper().strip() not in ("FNIRT", "MMORF"):
            raise ValueError("reg_method must be 'FNIRT' or 'MMORF'.")

        # reg2std script for fnirt path
        if self.reg2std_py:
            self.reg2std_py = _abs(self.reg2std_py)
        else:
            cand = pkg_file("nodes", "reg2std.py")
            self.reg2std_py = _abs(cand) if os.path.isfile(cand) else None
            if self.reg_method.upper().strip() == "FNIRT" and not self.reg2std_py:
                raise FileNotFoundError("reg2std_py must be provided for FNIRT registration.")

        if self.reg2std_config:
            self.reg2std_config = _abs(self.reg2std_config)

        # packaged NMT fallbacks (for nhp)
        self.nmt_fa_file = _abs_or_none(self.nmt_fa_file or pkg_file("files", "NMT_FA.nii.gz"))
        self.nmt_tensor_file = _abs_or_none(self.nmt_tensor_file or pkg_file("files", "NMT_tensor.nii.gz"))
        self.mmorf_config_template = _abs_or_none(self.mmorf_config_template or pkg_file("files", "mmorf_config_template.ini"))

        # -----------------------------
        # hum/nhp mode-specific setup
        # -----------------------------
        self.mode = (self.mode or "nhp").strip().lower()
        if self.mode not in {"nhp", "hum"}:
            raise ValueError("mode must be 'nhp' or 'hum'.")

        fsldir = os.environ.get("FSLDIR", "")
        std_dir = os.path.join(fsldir, "data", "standard") if fsldir else ""

        hcp_fa     = os.path.join(std_dir, "FSL_HCP1065_FA_1mm.nii.gz") if std_dir else None
        hcp_tensor = os.path.join(std_dir, "FSL_HCP1065_tensor_1mm.nii.gz") if std_dir else None
        mni_t1_1mm = os.path.join(std_dir, "MNI152_T1_1mm.nii.gz") if std_dir else None
        mni_vent   = os.path.join(std_dir, "MNI152_T1_2mm_VentricleMask.nii.gz") if std_dir else None

        if self.mode == "hum":
            # Registration targets
            self.std_fa_file     = _abs(self.std_fa_file     or hcp_fa)
            self.std_tensor_file = _abs(self.std_tensor_file or hcp_tensor)

            # Signal drift (ventricle) + reference
            self.std_ventricles_file = _abs(self.std_ventricles_file or mni_vent)
            self.std_ref_file        = _abs(self.std_ref_file or self.std_fa_file)

            # XTRACT defaults like in vivo
            self.xtract_species_default  = "HUMAN"
            self.xtract_mac_ext_default  = ""
            self.xtract_stdref_hum       = _abs(mni_t1_1mm)

            # MMORF multiscale defaults
            self.mmorf_warp_res_init = "64"
            self.mmorf_fwhm_set      = "16.0 16.0 8.0 4.0 2.0 1.0 0.5"

        else:  # nhp
            # Registration targets default to packaged NMT
            self.std_fa_file     = _abs(self.std_fa_file     or self.nmt_fa_file)
            self.std_tensor_file = _abs(self.std_tensor_file or self.nmt_tensor_file)

            # Signal drift ventricles + reference (packaged NMT)
            self.std_ventricles_file = _abs(
                self.std_ventricles_file or pkg_file("files", "NMT_v2.0_sym_05mm_ventricles.nii.gz")
            )
            self.std_ref_file = _abs(self.std_ref_file or self.std_fa_file)

            # XTRACT defaults like in vivo (macaque path)
            self.xtract_species_default  = "MACAQUE"
            self.xtract_mac_ext_default  = "_NMT"
            self.xtract_stdref_hum       = None  # not used for nhp

            # MMORF multiscale defaults
            self.mmorf_warp_res_init = "16"
            self.mmorf_fwhm_set      = "4.0 4.0 2.0 1.0 0.5 0.25 0.125"

        # --- XTRACT user override logic ---
        # If user supplied an explicit species, keep it; otherwise adopt mode default
        self.xtract_species = (self.xtract_species or self.xtract_species_default).strip().upper()

        # Profiles directory
        self.xtract_profiles_dir = _abs(self.xtract_profiles_dir or "")
        self.xtract_mac_ext_default = self.xtract_mac_ext_default
        
        # fibre defaults normalization
        self.fibre_pipeline_cmd = (self.fibre_pipeline_cmd or "Pipeline_Ball_3_Sticks_2Shells_SSFP.sh").strip()
        self.fibre_init_method  = (self.fibre_init_method or "").strip()
        self.fibre_queue        = (self.fibre_queue or "").strip()
        self.fibre_njobs        = int(self.fibre_njobs or 4)
        self.fibre_run_mcmc     = bool(self.fibre_run_mcmc)
        self.fibre_no_levmar    = bool(self.fibre_no_levmar)
        self.fibre_burnin       = int(self.fibre_burnin or 10000)
        self.fibre_jumps        = int(self.fibre_jumps or 1250)
        self.fibre_sample_every = int(self.fibre_sample_every or 25)
        self.fibre_subject_dir  = _abs(self.fibre_subject_dir or "")

        # xtract normalization
        self.xtract_species       = (self.xtract_species or "CUSTOM").strip()
        self.xtract_no_gpu        = bool(self.xtract_no_gpu)
        self.xtract_ptx_steplength = float(self.xtract_ptx_steplength or 0.15)
        self.xtract_profiles_dir  = _abs(self.xtract_profiles_dir or "")

        # denoising validation
        self.run_denoise = bool(self.run_denoise)
        self.denoise_method = (self.denoise_method or "NORDIC").strip()
        self.denoise_name = (self.denoise_name or "data").strip()

        if self.run_denoise:
            # locate wrapper if not explicitly set; try packaged default
            self.denoise_sh = _abs(self.denoise_sh or pkg_file("nodes", "denoise.sh"))
            if not (self.denoise_sh and os.path.isfile(self.denoise_sh)):
                raise FileNotFoundError("run_denoise=True, but denoise_sh not found.")
            # phase image is optional
            if "data_phase" in self.data_paths:
                self.data_paths["data_phase"] = _abs(self.data_paths.get("data_phase"))
