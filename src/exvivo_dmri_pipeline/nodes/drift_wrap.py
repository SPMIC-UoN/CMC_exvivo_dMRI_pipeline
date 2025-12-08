#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard lib
import os

# Third party
from nipype.interfaces.base import (
    TraitedSpec, File, Directory, traits,
    CommandLine, CommandLineInputSpec, isdefined, InputMultiPath
)

# Local
from ..utils import pkg_file

class DriftInputSpec(CommandLineInputSpec):
    # Script to run (our packaged drift.py)
    script = File(exists=True, position=0, argstr="%s", mandatory=True)

    # Accept one or many inputs; Nipype will coerce a single string into a list
    input  = InputMultiPath(File(exists=True), argstr="-i %s",     mandatory=True, sep=" ")
    bvals  = InputMultiPath(File(exists=True), argstr="-bvals %s", mandatory=True, sep=" ")

    model  = traits.Enum("single", "linear", "quadratic", usedefault=True, argstr="-model %s")
    outdir = traits.Str(argstr="-outdir %s", usedefault=True, default_value="drift_analysis")
    output = traits.List(traits.Str, argstr="--output %s", sep=" ")

    # Optional mask-based QC (ex vivo)
    mask   = File(exists=True, argstr="--mask %s")
    vent   = File(exists=True, argstr="--vent %s")
    std    = File(exists=True, argstr="--std %s")

    # Tools folder must contain reg2std.py; default to packaged nodes/ if not provided
    toolsdir = Directory(argstr="--toolsdir %s")

class DriftOutputSpec(TraitedSpec):
    out_files = traits.List(File())

class Drift(CommandLine):
    _cmd = "python3"
    input_spec  = DriftInputSpec
    output_spec = DriftOutputSpec

    def _run_interface(self, runtime):
        # If user didn't provide toolsdir, point to packaged nodes/ (reg2std.py lives there)
        if not isdefined(self.inputs.toolsdir) or not self.inputs.toolsdir:
            reg2std_path = pkg_file("nodes", "reg2std.py")
            self.inputs.toolsdir = os.path.dirname(reg2std_path)
        return super()._run_interface(runtime)

    def _list_outputs(self):
        outs = self._outputs().get()
        outdir = os.path.abspath(self.inputs.outdir)
        outs["out_files"] = [os.path.join(outdir, p) for p in (self.inputs.output or [])]
        return outs
