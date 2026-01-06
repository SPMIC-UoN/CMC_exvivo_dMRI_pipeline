#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shlex
import subprocess
import shutil
from typing import Sequence

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


def _run_quiet(args: Sequence[str]) -> None:
    try:
        subprocess.run(
            [str(a) for a in args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
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
    atl    = File(exists=True,  mandatory=True, desc="Standard space FA template")


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
# MMORF path driven by an INI template + mode specific settings
# ======================================================================

class MMORFInputSpec(BaseInterfaceInputSpec):
    # Subject inputs produced earlier in the pipeline
    fa     = File(exists=False, mandatory=True, desc="Subject FA (e.g., dti_FA.nii.gz)")
    tensor = File(exists=False, mandatory=True, desc="Subject tensor (e.g., dti_tensor.nii.gz)")

    # References (provided by pipeline / config; species / template agnostic)
    std_ref_fa     = File(exists=True,  mandatory=True, desc="Standard space FA template")
    std_ref_tensor = File(exists=True,  mandatory=True, desc="Standard space tensor template")

    # Config template and output dir
    config_template = File(exists=True, mandatory=True, desc="mmorf_config_template.ini")
    outdir          = Directory(mandatory=True, desc="Output directory (STDreg)")

    # Mode drives warp_res_init and FWHM schedule
    mode = traits.Enum("hum", "nhp", usedefault=False,
                       desc="Species mode (hum or nhp) for multiscale MMORF settings")

    # Tool name (assumed on PATH or in FSLDIR/bin)
    mmorf = traits.Str("mmorf", usedefault=True)


class MMORFOutputSpec(TraitedSpec):
    outdir            = Directory()
    affine_mat        = File()
    mmorf_warp        = File()
    combined_warp     = File()
    combined_warp_inv = File()
    fa_in_std         = File()


class MMORFReg(BaseInterface):
    """
    MMORF registration using an INI template.

    All generic paths come from Config / pipeline, while the multiscale settings
    (warp_res_init and FWHM schedule) are injected here based on mode (hum vs nhp),
    mirroring the in vivo implementation.
    """

    input_spec  = MMORFInputSpec
    output_spec = MMORFOutputSpec

    def _verify_file(self, p: str, name: str) -> str:
        ap = os.path.abspath(p)
        if not os.path.isfile(ap):
            raise FileNotFoundError(f"{name} not found: {ap}")
        return ap

    def _assert_mmorf_ini_filled(self, text: str, src_path: str):
        """Raise if any of the literal placeholders are still present."""
        placeholders = [
            "FSLDIR",
            "std_ref_fa_file",
            "std_ref_tensor_file",
            "subject_fa_file",
            "subject_tensor_file",
            "subject_aff_mat",
            "warp_res_init_arg",
            "fwhm_set",
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

    def _mode_params(self, mode: str):
        """
        Return warp_res_init and FWHM list according to mode.
        """
        mode = (mode or "nhp").lower()

        if mode == "hum":
            warp_res_init = 64
            fset = [16.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5]
        else:
            warp_res_init = 16
            fset = [4.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125]

        return warp_res_init, fset

    def _inject_multiscale(self, txt: str, mode: str) -> str:
        """
        Inject warp_res_init and FWHM schedule.

        This relies on the exact placeholder forms in the template:
          warp_res_init           = warp_res_init_arg
          fwhm_ref_scalar     = fwhm_set
          fwhm_mov_scalar     = fwhm_set
          fwhm_ref_tensor     = fwhm_set
          fwhm_mov_tensor     = fwhm_set
        """
        warp_res_init, fset = self._mode_params(mode)
        fset_str = " ".join(f"{v:g}" for v in fset)

        # Match the exact strings from the template
        txt = txt.replace(
            "warp_res_init           = warp_res_init_arg",
            f"warp_res_init           = {warp_res_init}"
        )
        txt = txt.replace(
            "fwhm_ref_scalar     = fwhm_set",
            f"fwhm_ref_scalar     = {fset_str}"
        )
        txt = txt.replace(
            "fwhm_mov_scalar     = fwhm_set",
            f"fwhm_mov_scalar     = {fset_str}"
        )
        txt = txt.replace(
            "fwhm_ref_tensor     = fwhm_set",
            f"fwhm_ref_tensor     = {fset_str}"
        )
        txt = txt.replace(
            "fwhm_mov_tensor     = fwhm_set",
            f"fwhm_mov_tensor     = {fset_str}"
        )

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

        # FSL root for any FSLDIR placeholders
        fsl_root = _fsl_root()
        if not fsl_root:
            flirt_path = shutil.which("flirt") or _fsl_exe("flirt")
            if flirt_path and "/bin/flirt" in str(flirt_path):
                fsl_root = os.path.dirname(os.path.dirname(os.path.abspath(flirt_path)))

        repl = {
            "FSLDIR":              fsl_root or "FSLDIR",
            "std_ref_fa_file":     std_fa,
            "std_ref_tensor_file": std_t,
            "subject_fa_file":     fa,
            "subject_tensor_file": tensor,
            "subject_aff_mat":     affine_mat,
        }
        for k, v in repl.items():
            txt = txt.replace(k, v)

        # 3) Inject mode specific warp_res_init and FWHM schedule
        mode = getattr(self.inputs, "mode", None) or "nhp"
        txt = self._inject_multiscale(txt, mode)

        # 4) Sanity check: no core placeholders left
        self._assert_mmorf_ini_filled(txt, cfg_src)

        with open(cfg_dst, "w") as f:
            f.write(txt)

        # 5) Run MMORF with INI
        mmorf_warp = os.path.join(out, "mmorf_warp.nii.gz")
        _run_quiet([self.inputs.mmorf, "--config", cfg_dst, "--warp_out", mmorf_warp])

        # 6) Combine affine and non linear warp to STD
        combined_warp = os.path.join(out, "combined_warp.nii.gz")
        _run_quiet([
            _fsl_exe("convertwarp"),
            "-m", affine_mat,
            "-w", mmorf_warp,
            "-r", std_fa,
            "-o", combined_warp,
        ])

        # 7) Apply to FA to create QA output in standard space
        fa_in_std = os.path.join(out, "FA_in_STD.nii.gz")
        _run_quiet([
            _fsl_exe("applywarp"),
            "-i", fa,
            "-r", std_fa,
            "-w", combined_warp,
            "-o", fa_in_std,
        ])

        # 8) Inverse warp (std -> anat)
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