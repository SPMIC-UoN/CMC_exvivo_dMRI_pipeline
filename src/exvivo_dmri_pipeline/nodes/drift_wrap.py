# src/exvivo_dmri_pipeline/nodes/drift_wrap.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from nipype.interfaces.base import (
    TraitedSpec, File, Directory, traits,
    CommandLine, CommandLineInputSpec, isdefined, InputMultiPath
)

from ..utils import pkg_file


def _fsl_std_vent_path() -> str:
    """Return FSL's HUM ventricles mask if available."""
    fsldir = os.environ.get("FSLDIR", "")
    if not fsldir:
        return ""
    cand = os.path.join(fsldir, "data", "standard", "MNI152_T1_2mm_VentricleMask.nii.gz")
    return cand if os.path.isfile(cand) else ""


class DriftInputSpec(CommandLineInputSpec):
    script = File(exists=True, position=0, argstr="%s", mandatory=True)
    input  = InputMultiPath(File(exists=True), argstr="-i %s",     mandatory=True, sep=" ")
    bvals  = InputMultiPath(File(exists=True), argstr="-bvals %s", mandatory=True, sep=" ")

    model  = traits.Enum("single", "linear", "quadratic", usedefault=True, argstr="-model %s")
    outdir = traits.Str(argstr="-outdir %s", usedefault=True, default_value="drift_analysis")
    output = traits.List(traits.Str, argstr="--output %s", sep=" ")

    mask   = File(exists=True, argstr="--mask %s", desc="STD-space brain mask for QC")
    vent   = File(exists=True, argstr="--vent %s", desc="STD-space ventricles mask for drift QC")
    std    = File(exists=True, argstr="--std %s",  desc="STD-space reference image (e.g., HCP1065 FA or NMT FA)")

    mode   = traits.Enum("hum", "nhp", argstr="--mode %s", mandatory=False,
                         desc="Species mode (hum or nhp) for QC defaults")
    toolsdir = Directory(argstr="--toolsdir %s")


class DriftOutputSpec(TraitedSpec):
    out_files = traits.List(File())


class Drift(CommandLine):
    """
    Thin wrapper around drift.py with explicit HUM/NHP handling for QC assets.
    """
    _cmd = "python3"
    input_spec  = DriftInputSpec
    output_spec = DriftOutputSpec

    def _run_interface(self, runtime):
        # Default toolsdir to packaged nodes directory
        if not isdefined(self.inputs.toolsdir) or not self.inputs.toolsdir:
            reg2std_path = pkg_file("nodes", "reg2std.py")
            self.inputs.toolsdir = os.path.dirname(reg2std_path)

        # If HUM mode and no explicit ventricles mask, fall back to FSL standard
        if isdefined(self.inputs.mode) and str(self.inputs.mode).lower() == "hum":
            if (not isdefined(self.inputs.vent)) or (not self.inputs.vent):
                hum_vent = _fsl_std_vent_path()
                if hum_vent:
                    self.inputs.vent = hum_vent

        return super()._run_interface(runtime)

    def _list_outputs(self):
        outs = self._outputs().get()
        outdir = os.path.abspath(self.inputs.outdir)
        outs["out_files"] = [os.path.join(outdir, p) for p in (self.inputs.output or [])]
        return outs