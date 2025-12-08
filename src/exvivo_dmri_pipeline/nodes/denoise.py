# src/exvivo_dmri_pipeline/nodes/denoise.py

import os
import shutil
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, Directory, traits,
    CommandLine, CommandLineInputSpec
)

from ..utils import img_stem


class DenoiseInputSpec(CommandLineInputSpec):
    # Wrapper shell script that performs denoising
    denoise_sh = File(exists=True, position=0, argstr="%s", mandatory=True,
                      desc="Path to denoise wrapper (bash)")

    # Method/domain configuration passed through to the wrapper
    meth   = traits.Str(argstr="-meth %s", usedefault=True, default_value="NORDIC")
    domain = traits.Enum("complex", "mag", usedefault=True, argstr="-domain %s")

    # Inputs
    magn = File(exists=True, mandatory=True, argstr="-magn %s",
                desc="Magnitude NIfTI (4D)")
    # Phase is optional — only formatted if present
    phase = File(exists=True, argstr="-phase %s",
                 desc="Optional phase NIfTI (4D)")

    # Output naming
    name  = traits.Str(mandatory=True, argstr="-name %s",
                       desc="Output stem (no extension)")
    oPath = Directory(mandatory=True, argstr="-oPath %s",
                      desc="Output directory")


class DenoiseOutputSpec(TraitedSpec):
    out_nii  = File(exists=True)
    out_bvec = File(exists=False)
    out_bval = File(exists=False)


class Denoise(CommandLine):
    """
    The script is expected to create:
      <oPath>/<name>.nii.gz
      <oPath>/<name>.bvec    (optional)
      <oPath>/<name>.bval    (optional)
    """
    _cmd = "bash"
    input_spec  = DenoiseInputSpec
    output_spec = DenoiseOutputSpec

    def _out_stem(self) -> str:
        return os.path.join(self.inputs.oPath, self.inputs.name)

    def _format_arg(self, name, spec, value):
        # Only include -phase if a real file was provided
        if name == "phase" and (value is None or str(value).strip() == "" or not os.path.exists(value)):
            return ""
        return super()._format_arg(name, spec, value)

    def _list_outputs(self):
        stem = self._out_stem()
        outs = self._outputs().get()
        outs["out_nii"]  = stem + ".nii.gz"
        outs["out_bvec"] = stem + ".bvec"
        outs["out_bval"] = stem + ".bval"
        return outs

    def _run_interface(self, runtime):
        # If outputs already exist, skip execution
        if os.path.exists(self._out_stem() + ".nii.gz"):
            runtime.returncode = 0
            return runtime
        return super()._run_interface(runtime)
