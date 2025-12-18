# src/exvivo_dmri_pipeline/nodes/reg.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shlex
import subprocess
import shutil

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    Directory,
    traits,
)

# ---------------------------------------------------------
# FSL helpers
# ---------------------------------------------------------
def _fsl_root() -> str:
    """Resolve FSLDIR once; fall back to inferring from flirt on PATH."""
    fsl = os.environ.get("FSLDIR")
    if fsl:
        return os.path.abspath(fsl)
    return ""


def _fsl_exe(name: str) -> str:
    root = _fsl_root()
    if root:
        return os.path.join(root, "bin", name)
    return name  # assume on PATH


def _run_quiet(args: list[str]) -> None:
    try:
        subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        cmd = " ".join(shlex.quote(str(a)) for a in args)
        msg = (e.stderr or e.stdout or str(e)).strip()
        raise RuntimeError(f"Command failed: {cmd}\n{msg}") from e


# ======================================================================
# FNIRT option (FLIRT + FNIRT)
# ======================================================================

class Reg2StdFnirtInputSpec(BaseInterfaceInputSpec):
    fa     = File(exists=True,  mandatory=True, desc="Subject FA (DTIFIT)")
    outdir = Directory(mandatory=True,        desc="Output directory (STDreg)")
    atl    = File(exists=True,  mandatory=True, desc="Standard-space FA template")


class Reg2StdFnirtOutputSpec(TraitedSpec):
    outdir        = Directory()
    warp_anat2std = File()
    warp_std2anat = File()
    fa_in_std     = File()


class Reg2Std(BaseInterface):
    """FLIRT (affine) + FNIRT (nonlinear) registration to standard FA."""
    input_spec  = Reg2StdFnirtInputSpec
    output_spec = Reg2StdFnirtOutputSpec

    def _outs(self):
        od = os.path.abspath(self.inputs.outdir)
        return (
            od,
            os.path.join(od, "stdreg_anat_to_std_warp.nii.gz"),
            os.path.join(od, "stdreg_std_to_anat_warp.nii.gz"),
            os.path.join(od, "FA_in_STD.nii.gz"),
            os.path.join(od, "stdreg_anat_to_std.mat"),
            os.path.join(od, "stdreg_anat_to_std.nii.gz"),
        )

    def _run_interface(self, runtime):
        fa    = os.path.abspath(self.inputs.fa)
        reffa = os.path.abspath(self.inputs.atl)
        od, warp_a2s, warp_s2a, fa_in_std, aff_mat, fa_lin = self._outs()
        os.makedirs(od, exist_ok=True)

        # 1) Affine (FLIRT)
        _run_quiet([
            _fsl_exe("flirt"),
            "-in", fa, "-ref", reffa,
            "-omat", aff_mat,
            "-out", fa_lin,
        ])

        # 2) Nonlinear (FNIRT) anat -> std
        _run_quiet([
            _fsl_exe("fnirt"),
            f"--in={fa}",
            f"--ref={reffa}",
            f"--aff={aff_mat}",
            f"--cout={warp_a2s}",
            f"--iout={fa_in_std}",
        ])

        # 3) Invert warp (std -> anat)
        _run_quiet([
            _fsl_exe("invwarp"),
            "-w", warp_a2s,
            "-o", warp_s2a,
            "-r", fa,
        ])

        self._outdir        = od
        self._warp_anat2std = warp_a2s
        self._warp_std2anat = warp_s2a
        self._fa_in_std     = fa_in_std
        return runtime

    def _list_outputs(self):
        return {
            "outdir":        getattr(self, "_outdir", ""),
            "warp_anat2std": getattr(self, "_warp_anat2std", ""),
            "warp_std2anat": getattr(self, "_warp_std2anat", ""),
            "fa_in_std":     getattr(self, "_fa_in_std", ""),
        }


# ======================================================================
# MMORF path driven by an INI template + mode-specific settings
# ======================================================================

class MMORFInputSpec(BaseInterfaceInputSpec):
    # Subject inputs produced earlier in the pipeline
    fa     = File(exists=False, mandatory=True, desc="Subject FA (e.g., dti_FA.nii.gz)")
    tensor = File(exists=False, mandatory=True, desc="Subject tensor (e.g., dti_tensor.nii.gz)")

    # References (provided by pipeline/config; species/template agnostic)
    std_ref_fa     = File(exists=True,  mandatory=True, desc="Standard-space FA template")
    std_ref_tensor = File(exists=True,  mandatory=True, desc="Standard-space tensor template")

    # Config template and output dir
    config_template = File(exists=True, mandatory=True, desc="mmorf_config_template.ini")
    outdir          = Directory(mandatory=True, desc="Output directory (STDreg)")

    # Mode drives warp_res_init and fwhm_set
    mode = traits.Enum("hum", "nhp", usedefault=False,
                       desc="Species mode (hum or nhp) for multiscale MMORF settings")

    # Tool name (assumed on PATH or in FSLDIR/bin)
    mmorf = traits.Str("mmorf", usedefault=True)


class MMORFOutputSpec(TraitedSpec):
    outdir              = Directory()
    affine_mat          = File()
    mmorf_warp          = File()
    combined_warp       = File()
    combined_warp_inv   = File()
    fa_in_std           = File()


class MMORFReg(BaseInterface):
    """
    MMORF registration using an INI template.

    All generic paths come from Config/pipeline, while the multiscale settings
    (warp_res_init and fwhm_set) are injected here based on mode (hum vs nhp).
    """

    input_spec  = MMORFInputSpec
    output_spec = MMORFOutputSpec

    def _verify_file(self, p: str, name: str) -> str:
        ap = os.path.abspath(p)
        if not os.path.isfile(ap):
            raise FileNotFoundError(f"{name} not found: {ap}")
        return ap

    def _assert_mmorf_ini_filled(self, text: str, src_path: str):
        placeholders = [
            "FSLDIR",
            "std_ref_fa_file",
            "std_ref_tensor_file",
            "subject_fa_file",
            "subject_tensor_file",
            "subject_aff_mat",
        ]
        leftovers = [p for p in placeholders if p in text]
        if leftovers:
            raise RuntimeError(
                "MMORF config still contains unresolved placeholders: "
                + ", ".join(leftovers)
                + f"\nTemplate: {src_path}"
            )
        if "warp_res_init" not in text:
            raise RuntimeError("MMORF INI missing 'warp_res_init' key.")

    def _inject_multiscale(self, txt: str, mode: str) -> str:
        """
        Ensure warp_res_init and fwhm_set are present and have sensible values
        depending on 'hum' vs 'nhp'.
        """
        mode = (mode or "").lower()
        if mode == "hum":
            warp_res = "64"
            fwhm_set = "16.0 16.0 8.0 4.0 2.0 1.0 0.5"
        else:  # nhp default
            warp_res = "16"
            fwhm_set = "4.0 4.0 2.0 1.0 0.5 0.25 0.125"

        # warp_res_init
        if "warp_res_init" in txt:
            txt = re.sub(
                r"warp_res_init\s*=\s*\S+",
                f"warp_res_init = {warp_res}",
                txt,
            )
        else:
            txt += f"\nwarp_res_init = {warp_res}\n"

        # fwhm_set
        if "fwhm_set" in txt:
            txt = re.sub(
                r"fwhm_set\s*=\s*.+",
                f"fwhm_set = {fwhm_set}",
                txt,
            )
        else:
            txt += f"\nfwhm_set = {fwhm_set}\n"

        return txt

    def _run_interface(self, runtime):
        out = os.path.abspath(self.inputs.outdir)
        os.makedirs(out, exist_ok=True)

        # Resolve subject and templates
        fa     = self._verify_file(self.inputs.fa, "subject FA")
        tensor = self._verify_file(self.inputs.tensor, "subject tensor")
        std_fa = self._verify_file(self.inputs.std_ref_fa, "STD FA template")
        std_t  = self._verify_file(self.inputs.std_ref_tensor, "STD tensor template")

        # 1) Initial affine: FLIRT FA -> STD FA
        affine_mat = os.path.join(out, "stdreg_anat_to_std.mat")
        lin_out    = os.path.join(out, "stdreg_anat_to_std")
        _run_quiet([
            _fsl_exe("flirt"),
            "-in", fa, "-ref", std_fa,
            "-omat", affine_mat,
            "-out", lin_out,
        ])

        # 2) Build INI from template with literal replacements
        cfg_src = os.path.abspath(self.inputs.config_template)
        cfg_dst = os.path.join(out, "mmorf_config.ini")
        with open(cfg_src, "r") as f:
            txt = f.read()

        fsl_root = _fsl_root()
        if not fsl_root:
            # Try to infer FSLDIR from the flirt path we just used
            flirt_path = shutil.which("flirt") or _fsl_exe("flirt")
            if flirt_path and "/bin/flirt" in flirt_path:
                fsl_root = os.path.dirname(os.path.dirname(os.path.abspath(flirt_path)))

        replacements = {
            "FSLDIR":              fsl_root or "FSLDIR",
            "std_ref_fa_file":     std_fa,
            "std_ref_tensor_file": std_t,
            "subject_fa_file":     fa,
            "subject_tensor_file": tensor,
            "subject_aff_mat":     affine_mat,
        }
        for k, v in replacements.items():
            txt = txt.replace(k, v)

        # Inject mode-specific warp_res_init / fwhm_set
        mode = getattr(self.inputs, "mode", None) or "nhp"
        txt = self._inject_multiscale(txt, mode)

        self._assert_mmorf_ini_filled(txt, cfg_src)

        with open(cfg_dst, "w") as f:
            f.write(txt)

        # 3) Run MMORF with INI
        mmorf_warp = os.path.join(out, "mmorf_warp.nii.gz")
        _run_quiet([self.inputs.mmorf, "--config", cfg_dst, "--warp_out", mmorf_warp])

        # 4) Combine affine and non-linear warp to STD
        combined_warp = os.path.join(out, "combined_warp.nii.gz")
        _run_quiet([
            _fsl_exe("convertwarp"),
            "-m", affine_mat,
            "-w", mmorf_warp,
            "-r", std_fa,
            "-o", combined_warp,
        ])

        # 5) Apply to FA to create QA output in standard space
        fa_in_std = os.path.join(out, "FA_in_STD.nii.gz")
        _run_quiet([
            _fsl_exe("applywarp"),
            "-i", fa,
            "-r", std_fa,
            "-w", combined_warp,
            "-o", fa_in_std,
        ])

        # 6) Inverse warp (std -> anat)
        combined_warp_inv = os.path.join(out, "combined_warp_inv.nii.gz")
        _run_quiet([
            _fsl_exe("invwarp"),
            "-w", combined_warp,
            "-o", combined_warp_inv,
            "-r", fa,
        ])

        # Save outputs for Nipype
        self._outdir            = out
        self._affine_mat        = affine_mat
        self._mmorf_warp        = mmorf_warp
        self._combined_warp     = combined_warp
        self._combined_warp_inv = combined_warp_inv
        self._fa_in_std         = fa_in_std

        return runtime

    def _list_outputs(self):
        return {
            "outdir":            getattr(self, "_outdir", ""),
            "affine_mat":        getattr(self, "_affine_mat", ""),
            "mmorf_warp":        getattr(self, "_mmorf_warp", ""),
            "combined_warp":     getattr(self, "_combined_warp", ""),
            "combined_warp_inv": getattr(self, "_combined_warp_inv", ""),
            "fa_in_std":         getattr(self, "_fa_in_std", ""),
        }