#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shlex
import subprocess

from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, traits
)

def _run(cmd):
    """Run and echo a command. Accepts list/tuple or string."""
    if isinstance(cmd, (list, tuple)):
        print(" ".join(shlex.quote(str(c)) for c in cmd))
        subprocess.run([str(c) for c in cmd], check=True)
    else:
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)

def _strip_nii_ext(p: str) -> str:
    if p.endswith(".nii.gz"):
        return p[:-7]
    if p.endswith(".nii"):
        return p[:-4]
    return p

def _exists_any(stub: str) -> str:
    """Return the actual file produced among stub, stub.nii, stub.nii.gz (or '' if none)."""
    for cand in (stub, stub + ".nii", stub + ".nii.gz"):
        if os.path.isfile(cand):
            return cand
    return ""

class BrainMaskInputSpec(BaseInterfaceInputSpec):
    data_nii   = File(exists=True, mandatory=True, desc="4D diffusion series")
    bvals      = File(exists=True, mandatory=True, desc="bvals text file")
    threshold  = traits.Float(50.0, usedefault=True, desc="initial intensity threshold")
    # explicit output targets (we write to these exact paths)
    out_meanb0 = File(mandatory=True, desc="path for mean b0 image (will be .nii.gz)")
    out_mask   = File(mandatory=True, desc="final nodif_brain_mask.nii.gz")

class BrainMaskOutputSpec(TraitedSpec):
    mean_b0 = File()
    mask    = File()

class BrainMask(BaseInterface):
    """
    Create a robust ex vivo brain mask:
      1) mean b0 from select_dwi_vols -m
      2) threshold + bin
      3) fill holes
      4) keep largest component
      5) erode then dilate median
    """
    input_spec  = BrainMaskInputSpec
    output_spec = BrainMaskOutputSpec

    def _run_interface(self, runtime):
        data   = os.path.abspath(self.inputs.data_nii)
        bvals  = os.path.abspath(self.inputs.bvals)
        thr    = float(self.inputs.threshold)

        out_meanb0 = os.path.abspath(self.inputs.out_meanb0)
        out_mask   = os.path.abspath(self.inputs.out_mask)
        os.makedirs(os.path.dirname(out_meanb0), exist_ok=True)
        os.makedirs(os.path.dirname(out_mask),   exist_ok=True)

        # 1) mean b0 using select_dwi_vols -m
        mean_stub = _strip_nii_ext(out_meanb0)
        try:
            _run(["select_dwi_vols", data, bvals, mean_stub, "-m"])
        except FileNotFoundError as e:
            raise RuntimeError(
                "Could not find 'select_dwi_vols' on PATH. "
                "Please load the module / environment that provides it."
            ) from e

        produced = _exists_any(mean_stub)
        if not produced:
            raise RuntimeError("select_dwi_vols did not produce a mean b0 image.")
        if os.path.abspath(produced) != out_meanb0:
            _run(["fslmaths", produced, out_meanb0])

        # 2) initial mask: threshold + bin
        _run(["fslmaths", out_meanb0, "-thr", str(thr), "-bin", out_mask])

        # 3) fill holes
        _run(["fslmaths", out_mask, "-fillh", out_mask])

        # 4) keep largest component
        clust = subprocess.run(
            ["fsl-cluster", "-i", out_mask, "-o", out_mask, "-t", "1"],
            text=True, capture_output=True, check=True
        ).stdout.strip().splitlines()

        best_idx, best_vox = None, -1
        for line in clust[1:]:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                cidx = int(float(parts[0]))
                vox  = int(float(parts[1]))
            except Exception:
                continue
            if vox > best_vox:
                best_vox, best_idx = vox, cidx

        if best_idx is not None:
            _run(["fslmaths", "-dt", "int", out_mask, "-thr", str(best_idx), "-uthr", str(best_idx), "-bin", out_mask])
        else:
            print("Warning: could not parse fsl-cluster output; keeping current mask.")

        # 5) erode then dilate median
        _run(["fslmaths", out_mask, "-ero", out_mask])
        _run(["fslmaths", out_mask, "-dilM", "-bin", out_mask])

        self._mean_b0 = out_meanb0
        self._mask    = out_mask
        return runtime

    def _list_outputs(self):
        return {"mean_b0": getattr(self, "_mean_b0", ""), "mask": getattr(self, "_mask", "")}
