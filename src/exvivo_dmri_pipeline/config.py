# src/exvivo_dmri_pipeline/config.py
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional, Dict
import yaml
from .utils import pkg_file

def _abs(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    return os.path.abspath(os.path.expanduser(p))

def _absify(d: Dict[str, str]) -> Dict[str, str]:
    out = {}
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

def _AbsOrNone(p: Optional[str]) -> Optional[str]:
    if p and os.path.isfile(p):
        return os.path.abspath(p)
    return _abs(p)

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
      - Optional (RUN_DENOISE = False by default)
      - If enabled and both magnitude + phase are present -> complex-domain denoise
      - If enabled and only magnitude present -> magnitude denoise
      - Expects a shell wrapper (DENOISE_SH) compatible with args used by nodes/denoise.py
    """

    # Mode
    acqtype: str = "ssfp"   # "ssfp" or "se"

    # Option A: single folder with all expected filenames
    input_dir: Optional[str] = None

    # Option B: explicit paths
    # keys: data_nii, bvals, bvecs, optional data_phase
    data_paths: Dict[str, str] = field(default_factory=dict)
    maps: Dict[str, str]       = field(default_factory=dict)   # keys: T1map, T2map, B1map
    sidecars: Dict[str, str]   = field(default_factory=dict)   # TRs, diffGradAmps, diffGradDurs, flipAngles, flipMask, noisefloor, optional whichshell

    brain_mask: Optional[str] = None
    dmri_root: Optional[str]  = None

    # Runtime toggles
    NO_GPU: bool = False
    brain_mask_thr: float = 50.0
    RUN_GIBBS: bool = True
    USE_DOCKER: bool = True
    CONTAINER_RUNTIME: str = "docker"
    INTERACTIVE_TTY: bool = True
    GIBBS_IMAGE: str = "docker.io/mrtrix3/mrtrix3:3.0.7"

    RUN_EDDY: bool = True
    RUN_EDDY_QC: bool = True
    EDDY_ECHO_SEC: Optional[float] = None

    RUN_SPATIAL_DRIFT: bool = True

    drift_py: Optional[str] = None
    std_mask_file: Optional[str] = None
    std_ventricles_file: Optional[str] = None
    std_ref_file: Optional[str] = None

    source_yaml: Optional[str] = None

    # DTI / CUDIMOT (QA tensor)
    DTI_APPROX_B: int = 1000
    DTI_DB: int = 60
    DTI_B0: int = 0

    # CUDIMOT DTI
    CUDIMOT_CMD: str = "Pipeline_DTI_SSFP.sh"
    CUDIMOT_QUEUE: Optional[str] = None
    CUDIMOT_NJOBS: int = 4
    CUDIMOT_RUN_MCMC: bool = True
    CUDIMOT_NO_LEVMAR: bool = True
    CUDIMOT_BURNIN: int = 10000
    CUDIMOT_JUMPS: int = 1250
    CUDIMOT_SAMPLE_EVERY: int = 25

    # Registration
    REG_METHOD: str = "FNIRT"
    REG2STD_PY: Optional[str] = None
    REG2STD_CONFIG: Optional[str] = None
    NMT_FA_FILE: Optional[str] = None
    NMT_TENSOR_FILE: Optional[str] = None
    MMORF_CONFIG_TEMPLATE: Optional[str] = None

    # Fibre modelling
    FIBRE_PIPELINE_CMD: str = "Pipeline_Ball_3_Sticks_2Shells_SSFP.sh"
    FIBRE_INIT_METHOD: str = ""
    FIBRE_QUEUE: str = ""
    FIBRE_NJOBS: int = 4
    FIBRE_RUN_MCMC: bool = True
    FIBRE_NO_LEVMAR: bool = True
    FIBRE_BURNIN: int = 10000
    FIBRE_JUMPS: int = 1250
    FIBRE_SAMPLE_EVERY: int = 25
    FIBRE_SUBJECT_DIR: str = ""

    # XTRACT
    XTRACT_SPECIES: str = "CUSTOM"
    XTRACT_NO_GPU: bool = False
    XTRACT_PTX_STEPLEN: float = 0.15
    XTRACT_PROFILES_DIR: str = ""

    # --------------------
    # Denoising
    # --------------------
    RUN_DENOISE: bool = False
    DENOISE_SH: Optional[str] = None      # path to wrapper script (bash)
    DENOISE_METHOD: str = "NORDIC"        # passed as -meth
    DENOISE_NAME: str = "data"            # output stem name

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        raw["acqtype"] = (_anykey(raw, "acqtype", "ssfp") or "ssfp").strip().lower()

        cfg = cls(**raw)
        cfg.source_yaml = _abs(path)
        cfg._finalize()
        return cfg

    @property
    def abs_dmri_root(self) -> str:
        return self.dmri_root

    def _finalize(self) -> None:
        # Workspace
        self.dmri_root = _abs(self.dmri_root or "./exvivo_proc")

        # Mode
        self.acqtype = (self.acqtype or "ssfp").strip().lower()
        if self.acqtype not in ("ssfp", "se"):
            raise ValueError("acqtype must be 'ssfp' or 'se'.")

        # Optional brain mask normalisation
        if isinstance(self.brain_mask, str) and self.brain_mask.strip().lower() in {"", "none", "null", "nil", "na"}:
            self.brain_mask = None

        # Packaged defaults
        self.drift_py = _abs(self.drift_py or pkg_file("nodes", "drift.py"))
        self.std_mask_file = _abs(self.std_mask_file or pkg_file("files", "NMT_v2.0_sym_05mm_brainmask.nii.gz"))
        self.std_ventricles_file = _abs(self.std_ventricles_file or pkg_file("files", "NMT_v2.0_sym_05mm_ventricles.nii.gz"))
        self.std_ref_file = _abs(self.std_ref_file or pkg_file("files", "MACAQUE_NMT.nii.gz"))

        # Option A: infer expected filenames from a folder
        if self.input_dir and not (self.data_paths or self.maps or self.sidecars):
            root = _abs(self.input_dir)

            def P(name: str) -> str:
                return os.path.join(root, name)

            self.data_paths = {
                "data_nii": P("data.nii.gz"),
                "bvals": P("bvals"),
                "bvecs": P("bvecs"),
            }
            # Optional phase if present (do not enforce existence here)
            phase_cand = P("data_ph.nii.gz")
            if os.path.isfile(phase_cand):
                self.data_paths["data_phase"] = phase_cand

            # SSFP maps provided pre-coreg, named simply
            self.maps = {
                "T1map": P("T1map.nii.gz"),
                "T2map": P("T2map.nii.gz"),
                "B1map": P("B1map.nii.gz"),
            }

            # Sidecars (no TE)
            self.sidecars = {
                "TRs": P("TRs"),
                "diffGradAmps": P("diffGradAmps"),
                "diffGradDurs": P("diffGradDurs"),
                "flipAngles": P("flipAngles"),
                "flipMask": P("flipMask"),
                "noisefloor": P("noisefloor"),
                "whichshell": P("whichshell"),
            }

        # Absolutize paths
        self.data_paths = _absify(self.data_paths)
        self.maps       = _absify(self.maps)
        self.sidecars   = _absify(self.sidecars)
        self.brain_mask = _abs(self.brain_mask)

        # Validate per mode
        def _need(d: Dict[str, str], key: str):
            p = d.get(key)
            if not p or not os.path.isfile(p):
                raise FileNotFoundError(f"Missing required '{key}': {p}")

        # Always need data + grads
        for k in ("data_nii", "bvals", "bvecs"):
            _need(self.data_paths, k)

        if self.acqtype == "ssfp":
            # Require all three maps, including B1map
            for k in ("T1map", "T2map", "B1map"):
                _need(self.maps, k)

            # Required sidecars for SSFP (TE removed)
            for k in ("TRs", "diffGradAmps", "diffGradDurs", "flipAngles", "flipMask", "noisefloor"):
                _need(self.sidecars, k)
            if self.sidecars.get("whichshell"):
                _need(self.sidecars, "whichshell")
        else:
            # SE: maps and sidecars not required
            self.maps = self.maps or {}
            self.sidecars = self.sidecars or {}

        # EDDY echo spacing required if EDDY/QC enabled
        if (self.RUN_EDDY or self.RUN_EDDY_QC) and (self.EDDY_ECHO_SEC is None):
            raise ValueError("EDDY_ECHO_SEC must be set (seconds) when RUN_EDDY or RUN_EDDY_QC is True.")

        # Normalise strings/ints
        self.CONTAINER_RUNTIME = (self.CONTAINER_RUNTIME or "docker").strip()
        self.GIBBS_IMAGE = self.GIBBS_IMAGE or "docker.io/mrtrix3/mrtrix3:3.0.7"

        # Registration config
        if self.REG_METHOD.upper().strip() not in ("FNIRT", "MMORF"):
            raise ValueError("REG_METHOD must be 'FNIRT' or 'MMORF'.")

        if self.REG2STD_PY:
            self.REG2STD_PY = _abs(self.REG2STD_PY)
        else:
            cand = pkg_file("nodes", "reg2std.py")
            self.REG2STD_PY = _abs(cand) if os.path.isfile(cand) else None
            if self.REG_METHOD.upper().strip() == "FNIRT" and not self.REG2STD_PY:
                raise FileNotFoundError("REG2STD_PY must be provided for FNIRT registration.")

        if self.REG2STD_CONFIG:
            self.REG2STD_CONFIG = _abs(self.REG2STD_CONFIG)

        # MMORF packaged defaults
        self.NMT_FA_FILE = _AbsOrNone(pkg_file("files", "NMT_FA.nii.gz"))
        self.NMT_TENSOR_FILE = _AbsOrNone(pkg_file("files", "NMT_tensor.nii.gz"))
        self.MMORF_CONFIG_TEMPLATE = _AbsOrNone(pkg_file("files", "mmorf_config_template.ini"))
        if self.REG_METHOD.upper().strip() == "MMORF":
            missing = []
            if not (self.NMT_FA_FILE and os.path.isfile(self.NMT_FA_FILE)):
                missing.append("NMT_FA.nii.gz")
            if not (self.NMT_TENSOR_FILE and os.path.isfile(self.NMT_TENSOR_FILE)):
                missing.append("NMT_tensor.nii.gz")
            if not (self.MMORF_CONFIG_TEMPLATE and os.path.isfile(self.MMORF_CONFIG_TEMPLATE)):
                missing.append("mmorf_config_template.ini")
            if missing:
                raise FileNotFoundError("Missing MMORF resources: " + ", ".join(missing))

        # Fibre defaults
        self.FIBRE_PIPELINE_CMD = (self.FIBRE_PIPELINE_CMD or "Pipeline_Ball_3_Sticks_2Shells_SSFP.sh").strip()
        self.FIBRE_INIT_METHOD   = (self.FIBRE_INIT_METHOD or "").strip()
        self.FIBRE_QUEUE         = (self.FIBRE_QUEUE or "").strip()
        self.FIBRE_NJOBS         = int(self.FIBRE_NJOBS or 4)
        self.FIBRE_RUN_MCMC      = bool(self.FIBRE_RUN_MCMC)
        self.FIBRE_NO_LEVMAR     = bool(self.FIBRE_NO_LEVMAR)
        self.FIBRE_BURNIN        = int(self.FIBRE_BURNIN or 10000)
        self.FIBRE_JUMPS         = int(self.FIBRE_JUMPS or 1250)
        self.FIBRE_SAMPLE_EVERY  = int(self.FIBRE_SAMPLE_EVERY or 25)
        self.FIBRE_SUBJECT_DIR   = _abs(self.FIBRE_SUBJECT_DIR or "")

        self.XTRACT_SPECIES      = (self.XTRACT_SPECIES or "CUSTOM").strip()
        self.XTRACT_NO_GPU       = bool(self.XTRACT_NO_GPU)
        self.XTRACT_PTX_STEPLEN  = float(self.XTRACT_PTX_STEPLEN or 0.15)
        self.XTRACT_PROFILES_DIR = _abs(self.XTRACT_PROFILES_DIR or "")

        # --------------- Denoising validation ---------------
        self.RUN_DENOISE = bool(self.RUN_DENOISE)
        self.DENOISE_METHOD = (self.DENOISE_METHOD or "NORDIC").strip()
        self.DENOISE_NAME = (self.DENOISE_NAME or "data").strip()

        if self.RUN_DENOISE:
            # Locate wrapper if not explicitly set; we try a packaged default
            self.DENOISE_SH = _abs(self.DENOISE_SH or pkg_file("nodes", "denoise.sh"))
            if not (self.DENOISE_SH and os.path.isfile(self.DENOISE_SH)):
                raise FileNotFoundError("RUN_DENOISE=True, but DENOISE_SH not found.")
            # Phase image is optional
            if "data_phase" in self.data_paths:
                self.data_paths["data_phase"] = _abs(self.data_paths.get("data_phase"))
