#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from typing import List, Optional

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    Directory,
    traits,
)

# -----------------------
# helpers
# -----------------------

def _abs(p: Optional[str]) -> Optional[str]:
    return os.path.abspath(p) if p else p


def _ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def _copy(src: str, dst: str) -> str:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return os.path.abspath(dst)


def _write_lines(path: str, lines: List[str]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for ln in lines:
            f.write(str(ln).strip() + "\n")
    return os.path.abspath(path)


# ============================================================
# 1) CopyImagesToRaw (SSFP: maps must already be coregistered)
#    Inputs are T1map, T2map, B1map (required in SSFP mode)
# ============================================================

class CopyImagesToRawInputSpec(BaseInterfaceInputSpec):
    data_nii = File(exists=True, mandatory=True,
                    desc="4D diffusion NIfTI to copy into proc/data_raw/data.nii.gz")

    # Pre-coregistered maps (naming without '_coreg')
    T1map = File(exists=True,  mandatory=True,  desc="T1 map (already coregistered to diffusion space)")
    T2map = File(exists=True,  mandatory=True,  desc="T2 map (already coregistered to diffusion space)")
    B1map = File(exists=True,  mandatory=True,  desc="B1 map (already coregistered to diffusion space)")

    # Optional extra
    brain_mask = File(exists=False, mandatory=False, desc="Optional brain mask NIfTI")

    outdir = Directory(mandatory=True, desc="Output raw dir (e.g., proc/data_raw)")


class CopyImagesToRawOutputSpec(TraitedSpec):
    data_file = File(desc="proc/data_raw/data.nii.gz")
    t1_file   = File(desc="proc/data_raw/T1map.nii.gz")
    t2_file   = File(desc="proc/data_raw/T2map.nii.gz")
    b1_file   = File(desc="proc/data_raw/B1map.nii.gz")
    mask_file = File(desc="proc/data_raw/brain_mask.nii.gz (if provided)")
    outdir    = Directory(desc="proc/data_raw")


class CopyImagesToRaw(BaseInterface):
    input_spec  = CopyImagesToRawInputSpec
    output_spec = CopyImagesToRawOutputSpec

    def _run_interface(self, runtime):
        raw_dir = _ensure_dir(_abs(self.inputs.outdir))

        self._data_file = _copy(_abs(self.inputs.data_nii), os.path.join(raw_dir, "data.nii.gz"))
        self._t1_file   = _copy(_abs(self.inputs.T1map),   os.path.join(raw_dir, "T1map.nii.gz"))
        self._t2_file   = _copy(_abs(self.inputs.T2map),   os.path.join(raw_dir, "T2map.nii.gz"))
        self._b1_file   = _copy(_abs(self.inputs.B1map),   os.path.join(raw_dir, "B1map.nii.gz"))

        self._mask_file = ""
        if getattr(self.inputs, "brain_mask", None):
            self._mask_file = _copy(_abs(self.inputs.brain_mask), os.path.join(raw_dir, "brain_mask.nii.gz"))

        self._outdir = raw_dir
        return runtime

    def _list_outputs(self):
        outs = self._outputs().get()
        outs["data_file"] = getattr(self, "_data_file", "")
        outs["t1_file"]   = getattr(self, "_t1_file", "")
        outs["t2_file"]   = getattr(self, "_t2_file", "")
        outs["b1_file"]   = getattr(self, "_b1_file", "")
        outs["mask_file"] = getattr(self, "_mask_file", "")
        outs["outdir"]    = getattr(self, "_outdir", "")
        return outs


# ======================================================
# 2) CopyExVivoSidecars
# ======================================================

class CopyExVivoSidecarsInputSpec(BaseInterfaceInputSpec):
    bvals        = File(exists=True,  mandatory=False)
    bvecs        = File(exists=True,  mandatory=False)
    TRs          = File(exists=True,  mandatory=False)
    diffGradAmps = File(exists=True,  mandatory=False)
    diffGradDurs = File(exists=True,  mandatory=False)
    flipAngles   = File(exists=True,  mandatory=False)
    flipMask     = File(exists=True,  mandatory=False)
    noisefloor   = File(exists=True,  mandatory=False)
    whichshell   = File(exists=True,  mandatory=False)

    outdir       = Directory(mandatory=True, desc="Destination sidecars directory (e.g., proc/sidecars)")


class CopyExVivoSidecarsOutputSpec(TraitedSpec):
    out_dir           = Directory(desc="proc/sidecars")
    bvals_file        = File()
    bvecs_file        = File()
    TRs_file          = File()
    diffGradAmps_file = File()
    diffGradDurs_file = File()
    flipAngles_file   = File()
    flipMask_file     = File()
    noisefloor_file   = File()
    whichshell_file   = File()


class CopyExVivoSidecars(BaseInterface):
    input_spec  = CopyExVivoSidecarsInputSpec
    output_spec = CopyExVivoSidecarsOutputSpec

    def _maybe_copy(self, src: Optional[str], dst_name: str, outdir: str) -> str:
        if src and os.path.isfile(src):
            return _copy(src, os.path.join(outdir, dst_name))
        else:
            return ""

    def _run_interface(self, runtime):
        dst = _ensure_dir(_abs(self.inputs.outdir))

        self._bvals_file        = self._maybe_copy(_abs(getattr(self.inputs, "bvals", None)),        "bvals",        dst)
        self._bvecs_file        = self._maybe_copy(_abs(getattr(self.inputs, "bvecs", None)),        "bvecs",        dst)
        self._TRs_file          = self._maybe_copy(_abs(getattr(self.inputs, "TRs", None)),          "TRs",          dst)
        self._diffGradAmps_file = self._maybe_copy(_abs(getattr(self.inputs, "diffGradAmps", None)), "diffGradAmps", dst)
        self._diffGradDurs_file = self._maybe_copy(_abs(getattr(self.inputs, "diffGradDurs", None)), "diffGradDurs", dst)
        self._flipAngles_file   = self._maybe_copy(_abs(getattr(self.inputs, "flipAngles", None)),   "flipAngles",   dst)
        self._flipMask_file     = self._maybe_copy(_abs(getattr(self.inputs, "flipMask", None)),     "flipMask",     dst)
        self._noisefloor_file   = self._maybe_copy(_abs(getattr(self.inputs, "noisefloor", None)),   "noisefloor",   dst)
        self._whichshell_file   = self._maybe_copy(_abs(getattr(self.inputs, "whichshell", None)),   "whichshell",   dst)

        self._out_dir = dst
        return runtime

    def _list_outputs(self):
        outs = self._outputs().get()
        outs["out_dir"]           = getattr(self, "_out_dir", "")
        outs["bvals_file"]        = getattr(self, "_bvals_file", "")
        outs["bvecs_file"]        = getattr(self, "_bvecs_file", "")
        outs["TRs_file"]          = getattr(self, "_TRs_file", "")
        outs["diffGradAmps_file"] = getattr(self, "_diffGradAmps_file", "")
        outs["diffGradDurs_file"] = getattr(self, "_diffGradDurs_file", "")
        outs["flipAngles_file"]   = getattr(self, "_flipAngles_file", "")
        outs["flipMask_file"]     = getattr(self, "_flipMask_file", "")
        outs["noisefloor_file"]   = getattr(self, "_noisefloor_file", "")
        outs["whichshell_file"]   = getattr(self, "_whichshell_file", "")
        return outs


# ======================================================
# 3) WritePathsFile (uses T1map/T2map/B1map; no mask)
# ======================================================

class WritePathsFileInputSpec(BaseInterfaceInputSpec):
    lowershell_dir = Directory(mandatory=True, desc="proc/data_lowershell (CUDIMOT DTI input)")
    data_dir       = Directory(mandatory=False, desc="proc/data (for fibre modelling)")
    sidecars_dir   = Directory(exists=True, mandatory=True, desc="<proc>/sidecars for canonical sidecars")

    # Pre-coregistered maps (required)
    t1_map = File(exists=True, mandatory=True, desc="T1map.nii.gz")
    t2_map = File(exists=True, mandatory=True, desc="T2map.nii.gz")
    b1_map = File(exists=True, mandatory=True, desc="B1map.nii.gz")

    mean_b0 = File(exists=False, mandatory=False, desc="mean_b0 (optional; will be symlinked)")
    acqtype = traits.Str("ssfp", usedefault=True)


class WritePathsFileOutputSpec(TraitedSpec):
    CFP  = File(desc="CFP for DTI_SSFP written in lowershell_dir (SSFP only)")
    FixP = File(desc="FixP file (written in both lowershell_dir and data_dir; SSFP only)")
    lowershell_fixp = File()
    data_fixp       = File()
    data_cfp        = File(desc="CFP for Ball_3_Sticks_2Shells_SSFP written in data_dir")
    lowershell_cfp  = File()


class WritePathsFile(BaseInterface):
    input_spec  = WritePathsFileInputSpec
    output_spec = WritePathsFileOutputSpec

    def _compose_fixp(self, t1: str, t2: str, b1: str) -> List[str]:
        return [t1, t2, b1]

    def _compose_cfp_full_for_ball(self, base_dir: str) -> List[str]:
        """
        Ball_3_Sticks_2Shells_SSFP expects:
          bvecs, bvals, TRs, diffGradAmps, diffGradDurs, flipAngles, whichshell, noisefloor
        """
        bvecs = os.path.join(base_dir, "bvecs")
        return [
            bvecs,
            os.path.join(base_dir, "bvals"),
            os.path.join(base_dir, "TRs"),
            os.path.join(base_dir, "diffGradAmps"),
            os.path.join(base_dir, "diffGradDurs"),
            os.path.join(base_dir, "flipAngles"),
            os.path.join(base_dir, "whichshell"),
            os.path.join(base_dir, "noisefloor"),
        ]

    def _compose_cfp_ls_for_dti(self, base_dir: str) -> List[str]:
        """
        DTI_SSFP expects a lower-shell subset (if whichshell available):
          bvecs, TRs, diffGradAmps, diffGradDurs, flipAngles, noisefloor
        """
        bvecs = os.path.join(base_dir, "bvecs")
        return [
            bvecs,
            os.path.join(base_dir, "TRs"),
            os.path.join(base_dir, "diffGradAmps"),
            os.path.join(base_dir, "diffGradDurs"),
            os.path.join(base_dir, "flipAngles"),
            os.path.join(base_dir, "noisefloor"),
        ]

    def _safe_link(self, target: Optional[str], link_path: str) -> Optional[str]:
        if not target or not os.path.exists(target):
            return None
        os.makedirs(os.path.dirname(link_path), exist_ok=True)
        try:
            if os.path.islink(link_path) or os.path.exists(link_path):
                os.remove(link_path)
        except Exception:
            pass
        try:
            os.symlink(os.path.abspath(target), link_path)
        except OSError:
            shutil.copy2(os.path.abspath(target), link_path)
        return os.path.abspath(link_path)

    def _run_interface(self, runtime):
        if (getattr(self.inputs, "acqtype", "ssfp") or "ssfp").strip().lower() == "se":
            # Nothing to do in SE mode
            self._CFP = self._FixP = self._lowershell_fixp = self._data_fixp = self._data_cfp = self._lowershell_cfp = ""
            return runtime

        ls_dir       = _ensure_dir(_abs(self.inputs.lowershell_dir))
        data_dir     = _ensure_dir(_abs(getattr(self.inputs, "data_dir", None)) or os.path.join(os.path.dirname(ls_dir), "data"))
        sidecars_dir = _ensure_dir(_abs(self.inputs.sidecars_dir))

        if getattr(self.inputs, "mean_b0", None):
            self._safe_link(_abs(self.inputs.mean_b0), os.path.join(ls_dir, "mean_b0.nii.gz"))

        t1 = _abs(self.inputs.t1_map)
        t2 = _abs(self.inputs.t2_map)
        b1 = _abs(self.inputs.b1_map)

        # FixP in both places
        lowershell_fixp = _write_lines(os.path.join(ls_dir,  "FixP"), self._compose_fixp(t1, t2, b1))
        data_fixp       = _write_lines(os.path.join(data_dir, "FixP"), self._compose_fixp(t1, t2, b1))

        # Full-data CFP for Ball_3_Sticks_2Shells_SSFP
        data_cfp = _write_lines(os.path.join(data_dir, "CFP"),
                                self._compose_cfp_full_for_ball(sidecars_dir))

        # Lower-shell sidecars (filter by whichshell == 0 if present)
        ws = os.path.join(sidecars_dir, "whichshell")
        keep_idx = None
        if os.path.isfile(ws):
            vals = open(ws).read().split()
            keep_idx = [i for i, v in enumerate(vals) if v.strip() == "0"] or None

        ls_side = os.path.join(ls_dir, "sidecars_ls")
        os.makedirs(ls_side, exist_ok=True)

        def _filter_cols(src, dst):
            if not os.path.isfile(src):
                return
            with open(src) as f:
                lines = [ln.rstrip("\n") for ln in f]
            if keep_idx is None:
                _write_lines(dst, lines)
                return
            if src.endswith("bvecs"):
                out = []
                for row in lines[:3]:
                    cols = row.split()
                    out.append(" ".join(cols[i] for i in keep_idx))
                _write_lines(dst, out)
            else:
                cols = lines[0].split()
                _write_lines(dst, [" ".join(cols[i] for i in keep_idx)])

        for name in ("bvecs", "bvals", "TRs", "diffGradAmps", "diffGradDurs", "flipAngles", "whichshell", "noisefloor"):
            _filter_cols(os.path.join(sidecars_dir, name), os.path.join(ls_side, name))

        ls_cfp = _write_lines(os.path.join(ls_dir, "CFP"),
                              self._compose_cfp_ls_for_dti(ls_side))

        self._CFP             = ls_cfp
        self._FixP            = lowershell_fixp
        self._lowershell_fixp = lowershell_fixp
        self._data_fixp       = data_fixp
        self._data_cfp        = data_cfp
        self._lowershell_cfp  = ls_cfp
        return runtime

    def _list_outputs(self):
        outs = self._outputs().get()
        outs["CFP"]             = getattr(self, "_CFP", "")
        outs["FixP"]            = getattr(self, "_FixP", "")
        outs["lowershell_fixp"] = getattr(self, "_lowershell_fixp", "")
        outs["data_fixp"]       = getattr(self, "_data_fixp", "")
        outs["data_cfp"]        = getattr(self, "_data_cfp", "")
        outs["lowershell_cfp"]  = getattr(self, "_lowershell_cfp", "")
        return outs
