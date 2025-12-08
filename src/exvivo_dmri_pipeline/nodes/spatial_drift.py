#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import subprocess
import shlex

from tqdm.auto import tqdm
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    Directory,
    traits,
)

from ..utils import FSL


class SpatialDriftInputSpec(BaseInterfaceInputSpec):
    imain = File(exists=True, mandatory=True, desc="input 4D dMRI (signal-drift corrected)")
    outdir = Directory(mandatory=True, desc="output directory (will contain data_reg.nii.gz)")
    enable = traits.Bool(True, usedefault=True, desc="run spatial drift correction (False = passthrough)")
    interp = traits.Enum("spline", "trilinear", "nearestneighbour", usedefault=True, desc="flirt interpolation")
    schedule = File(
        exists=False,
        usedefault=True,
        value="__FSL_DEFAULT__",
        desc="flirt schedule file; '__FSL_DEFAULT__' uses sch3Dtrans_3dof",
    )
    cleanup = traits.Bool(True, usedefault=True, desc="delete split_dat folder when done")


class SpatialDriftOutputSpec(TraitedSpec):
    out_file = File(desc="spatially drift-corrected 4D image (or passthrough path if disabled)")


class SpatialDrift(BaseInterface):
    """
    Volume-wise rigid registration to first volume:
      1) fslsplit along time
      2) flirt each vol (except first) to first, 3-DOF schedule + chosen interp
      3) fslmerge -t into data_reg.nii.gz
    """

    input_spec = SpatialDriftInputSpec
    output_spec = SpatialDriftOutputSpec

    def _fsl(self, exe: str) -> str:
        return os.path.join(FSL, "bin", exe)

    def _ensure_outdir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def _run_cmd(self, args: list[str], quiet: bool = False) -> None:
        try:
            if quiet:
                subprocess.run(
                    args,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            else:
                subprocess.run(args, check=True)
        except subprocess.CalledProcessError as e:
            cmd = " ".join(shlex.quote(str(a)) for a in args)
            msg = e.stderr or e.stdout or str(e)
            raise RuntimeError(f"Command failed: {cmd}\n{msg}") from e

    def _run_interface(self, runtime):
        imain = os.path.abspath(self.inputs.imain)
        outdir = os.path.abspath(self.inputs.outdir)
        self._ensure_outdir(outdir)

        out_img = os.path.join(outdir, "data_reg.nii.gz")

        # Passthrough
        if not self.inputs.enable:
            print("[spatial_drift] enable=False; passing through original image to eddy.")
            self._out_file = imain
            return runtime

        # Prepare paths and defaults
        split_dir = os.path.join(outdir, "split_dat")
        if os.path.isdir(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)
        schedule = (
            os.path.join(FSL, "etc", "flirtsch", "sch3Dtrans_3dof")
            if self.inputs.schedule == "__FSL_DEFAULT__"
            else self.inputs.schedule
        )

        # 1) Split
        print("[spatial_drift] Splitting 4D image ...")
        self._run_cmd([self._fsl("fslsplit"), imain, split_dir + os.sep, "-t"], quiet=True)

        # 2) Register volumes to the first
        vols = sorted(glob.glob(os.path.join(split_dir, "*.nii.gz")))
        if not vols:
            raise RuntimeError("fslsplit produced no volumes.")

        target = vols[0]
        to_merge = [target]

        print("[spatial_drift] Registering volumes to first volume ...")
        for v in tqdm(vols[1:], desc="Registering", unit="vol"):
            regout = v.replace(".nii.gz", "_b0reg.nii.gz")
            self._run_cmd(
                [
                    self._fsl("flirt"),
                    "-in", v,
                    "-ref", target,
                    "-schedule", schedule,
                    "-out", regout,
                    "-interp", self.inputs.interp,
                ],
                quiet=True,
            )
            to_merge.append(regout)

        # 3) Merge
        print(f"[spatial_drift] Merging {len(to_merge)} volumes ...")
        self._run_cmd([self._fsl("fslmerge"), "-t", out_img] + to_merge, quiet=True)

        # Cleanup
        if self.inputs.cleanup:
            shutil.rmtree(split_dir, ignore_errors=True)

        if not os.path.isfile(out_img):
            raise RuntimeError(f"Expected output not found: {out_img}")

        self._out_file = out_img
        return runtime

    def _list_outputs(self):
        return {"out_file": getattr(self, "_out_file", None)}
