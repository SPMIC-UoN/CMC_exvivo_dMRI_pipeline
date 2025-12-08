import os
import shutil
import subprocess
import numpy as np

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    Directory,
    traits,
    CommandLine,
    CommandLineInputSpec,
)

from ..utils import fslval  # needed for dim4 query


# -------------------------
# Prepare EDDY inputs (single-PE ex vivo)
# -------------------------
class PrepEddyInputSpec(BaseInterfaceInputSpec):
    imain = File(exists=True, mandatory=True, desc="4D dMRI to pass to eddy (drift or spatial-drift corrected)")
    bvals = File(exists=True, mandatory=True, desc="bvals for imain")
    bvecs = File(exists=True, mandatory=True, desc="bvecs for imain")
    out_root = traits.Str(mandatory=True, desc="root work dir (e.g., <dmri_root>/eddy)")
    echo_spacing_sec = traits.Float(mandatory=True, desc="echo spacing / total readout (seconds) for acqparams 4th field")


class PrepEddyOutputSpec(TraitedSpec):
    eddy_dir = Directory(desc="eddy working directory")
    acqparams = File(desc="acqparams.txt for eddy")
    index = File(desc="index.txt for eddy")
    bvals_out = File(desc="bvals copied into eddy dir")
    bvecs_out = File(desc="bvecs copied into eddy dir")
    imain_out = File(desc="path of imain used by eddy (same as input)")


class PrepareEddy(BaseInterface):
    input_spec = PrepEddyInputSpec
    output_spec = PrepEddyOutputSpec

    def _run_interface(self, runtime):
        eddy_dir = os.path.abspath(self.inputs.out_root)
        os.makedirs(eddy_dir, exist_ok=True)

        imain = os.path.abspath(self.inputs.imain)
        bvals_in = os.path.abspath(self.inputs.bvals)
        bvecs_in = os.path.abspath(self.inputs.bvecs)

        # Single line acqparams: "1 0 0 <echo_spacing_sec>"
        acqparams = os.path.join(eddy_dir, "acqparams.txt")
        with open(acqparams, "w") as f:
            f.write(f"1 0 0 {self.inputs.echo_spacing_sec:.6f}\n")

        # index.txt: one entry per volume, all "1"
        nvols = fslval(imain, "dim4")
        index_txt = os.path.join(eddy_dir, "index.txt")
        with open(index_txt, "w") as f:
            f.write("\n".join(["1"] * nvols) + "\n")

        # Copy gradients into eddy dir for reproducibility
        bvals_out = os.path.join(eddy_dir, "bvals")
        bvecs_out = os.path.join(eddy_dir, "bvecs")
        np.savetxt(bvals_out, np.loadtxt(bvals_in).reshape(1, -1), fmt="%d")
        np.savetxt(bvecs_out, np.loadtxt(bvecs_in), fmt="%.8f")

        self._eddy_dir = eddy_dir
        self._acqparams = acqparams
        self._index = index_txt
        self._bvals_out = bvals_out
        self._bvecs_out = bvecs_out
        self._imain_out = imain
        return runtime

    def _list_outputs(self):
        outs = self._outputs().get()
        outs.update(
            {
                "eddy_dir": self._eddy_dir,
                "acqparams": self._acqparams,
                "index": self._index,
                "bvals_out": self._bvals_out,
                "bvecs_out": self._bvecs_out,
                "imain_out": self._imain_out,
            }
        )
        return outs


# -------------------------
# Run EDDY (no topup) — QC only
# -------------------------
class EddyRunInputSpec(CommandLineInputSpec):
    imain = File(exists=True, argstr="--imain=%s", mandatory=True)
    mask = File(exists=True, argstr="--mask=%s", mandatory=True)
    index = File(exists=True, argstr="--index=%s", mandatory=True)
    acqp  = File(exists=True, argstr="--acqp=%s",  mandatory=True)
    bvecs = File(exists=True, argstr="--bvecs=%s", mandatory=True)
    bvals = File(exists=True, argstr="--bvals=%s", mandatory=True)
    out   = traits.Str(argstr="--out=%s", mandatory=True)

    enable = traits.Bool(True, usedefault=True, desc="skip eddy completely if False")
    no_gpu = traits.Bool(False, usedefault=True, desc="if True use CPU binary (eddy_cpu); else GPU (eddy)")


class EddyRunOutputSpec(TraitedSpec):
    out_file = File(desc="eddy output NIfTI (not used downstream)")
    ran      = traits.Bool(desc="whether eddy actually ran")


class EddyRun(CommandLine):
    _cmd = "bash -lc true"
    input_spec  = EddyRunInputSpec
    output_spec = EddyRunOutputSpec

    def _run_interface(self, runtime):
        if not self.inputs.enable:
            print("[eddy] Skipping eddy (enable=False).")
            self._ran = False
            runtime.returncode = 0
            return runtime

        binary = "eddy_cpu" if self.inputs.no_gpu else "eddy"
        if shutil.which(binary) is None:
            raise FileNotFoundError(f"{binary} not found on PATH. Please install FSL {binary} or add it to PATH.")
        print(f"[eddy] Using binary: {binary}")

        # Log shells
        bvals_arr = np.loadtxt(self.inputs.bvals).reshape(-1)
        shells = np.unique(np.round(bvals_arr, 0)).astype(int).tolist()
        print(f"[eddy] N={bvals_arr.size} volumes; shells={shells}")

        args = [
            binary,
            f"--imain={self.inputs.imain}",
            f"--mask={self.inputs.mask}",
            f"--index={self.inputs.index}",
            f"--acqp={self.inputs.acqp}",
            f"--bvecs={self.inputs.bvecs}",
            f"--bvals={self.inputs.bvals}",
            "--data_is_shelled",
            "--fwhm=0",
            "--b_range=100",
            "--flm=linear",
            "--niter=1",
            "--s2v_niter=0",
            "--mbs_niter=1",
            "--dont_peas",
            "--cnr_maps",
            "-v",
            f"--out={self.inputs.out}",
        ]
        subprocess.run(args, check=True)
        self._ran = True
        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        return {
            "out_file": os.path.abspath(self.inputs.out + ".nii.gz"),
            "ran": getattr(self, "_ran", False),
        }


# -------------------------
# Run eddy_quad QC only
# -------------------------
class EddyQuadInputSpec(CommandLineInputSpec):
    base  = traits.Str(mandatory=True, desc="eddy output base (without extension)")
    idx   = File(exists=True, argstr="-idx %s", mandatory=True)
    acqp  = File(exists=True, argstr="-par %s", mandatory=True)
    mask  = File(exists=True, argstr="-m %s",   mandatory=True)
    bvals = File(exists=True, argstr="-b %s",   mandatory=True)
    enable = traits.Bool(True, usedefault=True, desc="skip eddy_quad if False")


class EddyQuadOutputSpec(TraitedSpec):
    qc_dir = Directory(desc="<base>.qc directory")
    ran    = traits.Bool(desc="whether eddy_quad actually ran")


class EddyQuad(CommandLine):
    _cmd = "eddy_quad"  # assume on PATH
    input_spec  = EddyQuadInputSpec
    output_spec = EddyQuadOutputSpec

    def _run_interface(self, runtime):
        if not self.inputs.enable:
            print("[eddy_quad] Skipping QC (enable=False).")
            self._ran = False
            runtime.returncode = 0
            return runtime

        if shutil.which("eddy_quad") is None:
            raise FileNotFoundError("eddy_quad not found on PATH. Please ensure FSL is on PATH.")

        qc_dir = self.inputs.base + ".qc"

        # If eddy didn't run or no outputs exist, skip QC gracefully
        needed = self.inputs.base + ".nii.gz"
        if not os.path.exists(needed):
            print(f"[eddy_quad] Skipping QC — eddy output not found: {needed}")
            self._ran = False
            runtime.returncode = 0
            return runtime

        if os.path.isdir(qc_dir):
            print(f"[eddy_quad] QC exists at {qc_dir} - skipping.")
            self._ran = True
            runtime.returncode = 0
            return runtime

        args = [
            "eddy_quad",
            self.inputs.base,
            "-v",
            "-idx", self.inputs.idx,
            "-par", self.inputs.acqp,
            "-m",   self.inputs.mask,
            "-b",   self.inputs.bvals,
        ]
        subprocess.run(args, check=True)
        self._ran = True
        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        return {
            "qc_dir": os.path.abspath(self.inputs.base + ".qc"),
            "ran": getattr(self, "_ran", False),
        }
