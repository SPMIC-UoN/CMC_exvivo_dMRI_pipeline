#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shlex
import subprocess

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    Directory,
    traits,
)

# Resolve FSL root once
def _fsl_root() -> str:
    fsl = os.environ.get("FSLDIR")
    if fsl:
        return os.path.abspath(fsl)
    # Fallback: try to infer from flirt location if present in PATH later
    return ""

def _fsl_exe(name: str) -> str:
    root = _fsl_root()
    if root:
        return os.path.join(root, "bin", name)
    return name  # hope PATH contains it

def _run_quiet(args: list[str]) -> None:
    try:
        subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        cmd = " ".join(shlex.quote(str(a)) for a in args)
        msg = (e.stderr or e.stdout or str(e)).strip()
        raise RuntimeError(f"Command failed: {cmd}\n{msg}") from e


# ======================================================================
# FNIRT option
# ======================================================================

class Reg2StdFnirtInputSpec(BaseInterfaceInputSpec):
    fa     = File(exists=True,  mandatory=True, desc="Subject FA (DTIFIT)")
    outdir = Directory(mandatory=True,        desc="Output directory (NMTreg)")
    atl    = File(exists=True,  mandatory=True, desc="Template FA (NMT_FA.nii.gz)")

class Reg2StdFnirtOutputSpec(TraitedSpec):
    outdir        = Directory()
    warp_anat2std = File()
    warp_std2anat = File()
    fa_in_std     = File()

class Reg2Std(BaseInterface):
    """Classic FLIRT+FNIRT wrapper: outputs match earlier pipeline expectations."""
    input_spec  = Reg2StdFnirtInputSpec
    output_spec = Reg2StdFnirtOutputSpec

    def _outs(self):
        od = os.path.abspath(self.inputs.outdir)
        return (
            od,
            os.path.join(od, "stdreg_anat_to_std_warp.nii.gz"),
            os.path.join(od, "stdreg_std_to_anat_warp.nii.gz"),
            os.path.join(od, "FA_in_NMT.nii.gz"),
            os.path.join(od, "stdreg_anat_to_std.mat"),
            os.path.join(od, "stdreg_anat_to_std.nii.gz"),
        )

    def _run_interface(self, runtime):
        fa    = os.path.abspath(self.inputs.fa)
        reffa = os.path.abspath(self.inputs.atl)
        od, warp_a2s, warp_s2a, fa_in_std, aff_mat, fa_lin = self._outs()
        os.makedirs(od, exist_ok=True)

        # 1) Affine
        _run_quiet([
            _fsl_exe("flirt"),
            "-in", fa, "-ref", reffa,
            "-omat", aff_mat,
            "-out", fa_lin,
        ])

        # 2) FNIRT non linear (anat->std)
        _run_quiet([
            _fsl_exe("fnirt"),
            f"--in={fa}",
            f"--ref={reffa}",
            f"--aff={aff_mat}",
            f"--cout={warp_a2s}",
            f"--iout={fa_in_std}",
        ])

        # 3) Inverse
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
# MMORF path driven entirely by the INI
# ======================================================================

class MMORFInputSpec(BaseInterfaceInputSpec):
    # Subject inputs produced earlier in the pipeline
    fa     = File(exists=False, mandatory=True, desc="Subject FA (e.g., dti_FA.nii.gz)")
    tensor = File(exists=False, mandatory=True, desc="Subject tensor (e.g., dti_tensor.nii.gz)")

    # References
    std_ref_fa     = File(exists=True,  mandatory=True, desc="NMT FA template")
    std_ref_tensor = File(exists=True,  mandatory=True, desc="NMT tensor template")

    # Config template and output dir
    config_template = File(exists=True, mandatory=True, desc="mmorf_config_template.ini")
    outdir          = Directory(mandatory=True, desc="Output directory (NMTreg)")

    # Tool name (assumed on PATH or in FSLDIR/bin)
    mmorf = traits.Str("mmorf", usedefault=True)


class MMORFOutputSpec(TraitedSpec):
    outdir              = Directory()
    affine_mat          = File()
    mmorf_warp          = File()
    combined_warp       = File()
    combined_warp_inv   = File()
    fa_in_nmt           = File()


class MMORFReg(BaseInterface):
    """MMORF registration using a filled INI. All parameters must come from the INI."""

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

    def _run_interface(self, runtime):
        out = os.path.abspath(self.inputs.outdir)
        os.makedirs(out, exist_ok=True)

        # Resolve subject and templates
        fa     = self._verify_file(self.inputs.fa, "subject FA")
        tensor = self._verify_file(self.inputs.tensor, "subject tensor")
        std_fa = self._verify_file(self.inputs.std_ref_fa, "NMT FA template")
        std_t  = self._verify_file(self.inputs.std_ref_tensor, "NMT tensor template")

        # 1) Initial affine: FLIRT FA -> NMT FA
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
        if not fsl_root:
            # Last resort, leave as-is but warn by raising if template expects it
            pass

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

        self._assert_mmorf_ini_filled(txt, cfg_src)

        with open(cfg_dst, "w") as f:
            f.write(txt)

        # 3) Run MMORF with INI only
        mmorf_warp = os.path.join(out, "mmorf_warp.nii.gz")
        _run_quiet([self.inputs.mmorf, "--config", cfg_dst, "--warp_out", mmorf_warp])

        # 4) Combine affine and non linear warp to template
        combined_warp = os.path.join(out, "combined_warp.nii.gz")
        _run_quiet([
            _fsl_exe("convertwarp"),
            "-m", affine_mat,
            "-w", mmorf_warp,
            "-r", std_fa,
            "-o", combined_warp,
        ])

        # 5) Apply to FA to create QA output in template space
        fa_in_nmt = os.path.join(out, "dti_FA_in_NMT.nii.gz")
        _run_quiet([
            _fsl_exe("applywarp"),
            "-i", fa,
            "-r", std_fa,
            "-w", combined_warp,
            "-o", fa_in_nmt,
        ])

        # 6) Inverse warp (std->anat)
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
        self._fa_in_nmt         = fa_in_nmt

        return runtime

    def _list_outputs(self):
        return {
            "outdir":            getattr(self, "_outdir", ""),
            "affine_mat":        getattr(self, "_affine_mat", ""),
            "mmorf_warp":        getattr(self, "_mmorf_warp", ""),
            "combined_warp":     getattr(self, "_combined_warp", ""),
            "combined_warp_inv": getattr(self, "_combined_warp_inv", ""),
            "fa_in_nmt":         getattr(self, "_fa_in_nmt", ""),
        }
