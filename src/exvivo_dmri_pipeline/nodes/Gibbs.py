#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    Directory,
    InputMultiPath,
    traits,
    isdefined,
)


def _stem(p: str) -> str:
    """Return filename without .nii(.gz)."""
    b = os.path.basename(p)
    if b.endswith(".nii.gz"):
        return b[:-7]
    if b.endswith(".nii"):
        return b[:-4]
    return b


def _copy2(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


class GibbsInputSpec(BaseInterfaceInputSpec):
    data_nii = File(exists=True, mandatory=True, desc="input 4D dMRI .nii(.gz)")
    bvals = File(exists=True, desc=".bval to copy next to output")
    bvecs = File(exists=True, desc=".bvec to copy next to output")

    outdir = Directory(mandatory=True, desc="output directory for gibbs-corrected data")

    # 3D or 2D gibbs removal (DW-SSFP -> 3D recommended)
    mode = traits.Enum("3D", "2D", usedefault=True, desc="Gibbs removal mode")

    # Container options only used for 2D (mrdegibbs)
    run_gibbs = traits.Bool(True, usedefault=True, desc="execute gibbs removal")
    use_docker = traits.Bool(True, usedefault=True, desc="use container runtime for 2D mrdegibbs")
    container_runtime = traits.Str("docker", usedefault=True, desc="docker or podman")
    interactive_tty = traits.Bool(True, usedefault=True, desc="add -it to container run")
    gibbs_image = traits.Str(
        "docker.io/mrtrix3/mrtrix3:3.0.7",
        usedefault=True,
        desc="MRtrix image tag (for 2D mrdegibbs)",
    )

    # dummy deps to allow ordering if needed
    deps = InputMultiPath(File, desc="dependencies only to enforce ordering")


class GibbsOutputSpec(TraitedSpec):
    out_file = File(desc="gibbs-corrected .nii.gz")
    out_bvals = File(desc="copied .bval in outdir (if provided)")
    out_bvecs = File(desc="copied .bvec in outdir (if provided)")


class Gibbs(BaseInterface):
    """
    Gibbs ringing removal:
      - mode='3D' -> run `deGibbs3D <in> <out>` (assumed in $PATH)
      - mode='2D' -> run MRtrix `mrdegibbs` (can use Docker/Podman image)
    """

    input_spec = GibbsInputSpec
    output_spec = GibbsOutputSpec

    # ---- helpers ----

    def _rt_prefix(self) -> list[str]:
        rt = [self.inputs.container_runtime, "run"]
        if self.inputs.interactive_tty:
            rt += ["-it"]
        rt += ["--rm"]
        return rt

    def _run_logged(self, cmd: list[str]) -> int:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out, err = p.communicate()
        if out:
            sys.stdout.write(out)
        if err:
            sys.stderr.write(err)
        return p.returncode

    # ---- nipype interface ----

    def _run_interface(self, runtime):
        img = os.path.abspath(self.inputs.data_nii)
        os.makedirs(self.inputs.outdir, exist_ok=True)

        stem = _stem(img)
        suffix = "_gibbs3d" if self.inputs.mode == "3D" else "_gibbs2d"
        out_img = os.path.join(self.inputs.outdir, f"{stem}{suffix}.nii.gz")

        if self.inputs.run_gibbs:
            if self.inputs.mode == "3D":
                # Native binary in PATH
                cmd = ["deGibbs3D", img, out_img]
            else:
                # 2D mrdegibbs (optionally via Docker/Podman)
                if self.inputs.use_docker:
                    in_dir = os.path.dirname(img)
                    mounts = [
                        "-v",
                        f"{in_dir}:/data:ro,z",
                        "-v",
                        f"{self.inputs.outdir}:/out:z",
                    ]
                    cmd = (
                        self._rt_prefix()
                        + mounts
                        + [
                            self.inputs.gibbs_image,
                            "mrdegibbs",
                            f"/data/{os.path.basename(img)}",
                            f"/out/{os.path.basename(out_img)}",
                        ]
                    )
                else:
                    cmd = ["mrdegibbs", img, out_img]

            rc = self._run_logged(cmd)
            if rc != 0:
                raise RuntimeError(f"Gibbs removal failed with command: {' '.join(cmd)}")

            if not os.path.exists(out_img):
                raise RuntimeError(f"Gibbs removal produced no output: {out_img}")

        else:
            if not os.path.exists(out_img):
                raise RuntimeError(
                    f"run_gibbs=False but missing expected output: {out_img}"
                )

        # record outputs
        self._out_file = out_img
        self._out_bvals = None
        self._out_bvecs = None

        if isdefined(self.inputs.bvals) and self.inputs.bvals:
            dst = os.path.join(self.inputs.outdir, f"{stem}.bval")
            _copy2(self.inputs.bvals, dst)
            self._out_bvals = dst

        if isdefined(self.inputs.bvecs) and self.inputs.bvecs:
            dst = os.path.join(self.inputs.outdir, f"{stem}.bvec")
            _copy2(self.inputs.bvecs, dst)
            self._out_bvecs = dst

        return runtime

    def _list_outputs(self):
        return {
            "out_file": getattr(self, "_out_file", None),
            "out_bvals": getattr(self, "_out_bvals", None),
            "out_bvecs": getattr(self, "_out_bvecs", None),
        }
