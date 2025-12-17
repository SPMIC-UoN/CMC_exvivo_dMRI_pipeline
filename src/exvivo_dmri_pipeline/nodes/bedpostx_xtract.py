#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import shlex
import shutil
import subprocess

from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, Directory, InputMultiPath, traits,
    CommandLine, CommandLineInputSpec, isdefined
)

from ..utils import FSL  # Path to FSL root (used for xtract and helpers)

# =========================
# SSFP FIBRE MODELLING (CUDIMOT)
# =========================
class FibreCudimotInputSpec(CommandLineInputSpec):
    subject_dir = Directory(position=0, argstr="%s", mandatory=True,
                            desc="Subject directory <proc>/data containing data.nii(.gz), mask, CFP, FixP")
    pipeline_cmd = traits.Str(
        "Pipeline_Ball_3_Sticks_2Shells_SSFP.sh",
        usedefault=True,
        desc="CUDIMOT pipeline script (on PATH or absolute)"
    )
    init_method = traits.Str("", usedefault=True)

    queue        = traits.Str("", desc="-Q queue name (optional)")
    njobs        = traits.Int(usedefault=True, value=4, desc="-NJOBS")
    run_mcmc     = traits.Bool(True,  usedefault=True, desc="--runMCMC")
    no_levmar    = traits.Bool(True,  usedefault=True, desc="--no_LevMar")
    burnin       = traits.Int(usedefault=True, value=10000, desc="-b")
    jumps        = traits.Int(usedefault=True, value=1250,  desc="-j")
    sample_every = traits.Int(usedefault=True, value=25,    desc="-s")

    deps = InputMultiPath(File(), desc="unused; enforce dependency ordering")


class FibreCudimotOutputSpec(TraitedSpec):
    subject_dir = Directory(desc="input subject dir")
    cudimot_dir = Directory(desc="CUDIMOT model dir: <proc>/data.<ModelName>")


class FibreCudimot(CommandLine):
    """
    Run CUDIMOT against <proc>/data, then run the model finish script on:
        <proc>/data.<ModelName>
    """
    _cmd = "bash -lc true"
    input_spec  = FibreCudimotInputSpec
    output_spec = FibreCudimotOutputSpec

    # ---------- helpers ----------
    def _run_quiet(self, args: list[str]) -> None:
        import shutil, shlex, subprocess
        exe = args[0]
        if not os.path.isabs(exe):
            resolved = shutil.which(exe)
            if resolved:
                args[0] = resolved
        subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def _model_name(self, pipeline_cmd: str) -> str:
        base = os.path.basename(pipeline_cmd)
        if base.endswith(".sh"): base = base[:-3]
        if base.startswith("Pipeline_"): base = base[len("Pipeline_"):]
        return base

    def _finish_script(self, pipeline_cmd: str) -> str:
        import shutil
        model = self._model_name(pipeline_cmd)
        cand  = f"{model}_finish.sh"
        wh = shutil.which(cand)
        if wh:
            return wh
        pipe_resolved = shutil.which(pipeline_cmd) or pipeline_cmd
        return os.path.join(os.path.dirname(os.path.abspath(pipe_resolved)), cand)

    def _preflight(self, subj: str):
        data_nii_gz = os.path.join(subj, "data.nii.gz")
        data_nii    = os.path.join(subj, "data.nii")
        if not (os.path.isfile(data_nii_gz) or os.path.isfile(data_nii)):
            raise FileNotFoundError(f"[FibreCudimot] Missing diffusion: {data_nii_gz} or {data_nii}")
        for f, kind in [("CFP", "CFP"), ("FixP", "FixP")]:
            p = os.path.join(subj, f)
            if not os.path.isfile(p):
                raise FileNotFoundError(f"[FibreCudimot] Missing {kind}: {p}")

    # ---------- nipype API ----------
    def _run_interface(self, runtime):
        subj = os.path.abspath(self.inputs.subject_dir)        # <proc>/data
        proc = os.path.dirname(subj)                           # <proc>
        os.makedirs(subj, exist_ok=True)
        self._preflight(subj)

        pipe   = (self.inputs.pipeline_cmd or "").strip() or "Pipeline_Ball_3_Sticks_2Shells_SSFP.sh"
        finish = self._finish_script(pipe)
        model  = self._model_name(pipe)
        model_dir = os.path.join(proc, f"data.{model}")        # <proc>/data.<ModelName>

        # run CUDIMOT
        cmd = [pipe, subj]
        if self.inputs.queue.strip():  cmd += ["-Q", self.inputs.queue.strip()]
        if int(self.inputs.njobs) > 0: cmd += ["-NJOBS", str(int(self.inputs.njobs))]
        if self.inputs.no_levmar:      cmd += ["--no_LevMar"]
        if self.inputs.run_mcmc:
            cmd += ["--runMCMC",
                    "-b", str(int(self.inputs.burnin)),
                    "-j", str(int(self.inputs.jumps)),
                    "-s", str(int(self.inputs.sample_every))]
        cmd += [f"--CFP={os.path.join(subj, 'CFP')}",
                f"--FixP={os.path.join(subj, 'FixP')}"]
        self._run_quiet(cmd)

        # run finish script
        self._run_quiet([finish, model_dir])

        # outputs
        self._cudimot_dir = model_dir
        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        o = self._outputs().get()
        subj = os.path.abspath(self.inputs.subject_dir)
        proc = os.path.dirname(subj)
        model = self._model_name((self.inputs.pipeline_cmd or "").strip() or "Pipeline_Ball_3_Sticks_2Shells_SSFP.sh")
        o["subject_dir"] = subj
        o["cudimot_dir"] = os.path.join(proc, f"data.{model}")
        return o

# =========
# SE: BedpostX
# =========
class BedpostXInputSpec(BaseInterfaceInputSpec):
    data_dir = Directory(mandatory=True, desc="<proc>/data with data, bvals, bvecs, nodif_brain_mask")
    trigger  = File(exists=True, mandatory=False, desc="dummy file to enforce execution order")

class BedpostXOutputSpec(TraitedSpec):
    bpx_dir = Directory()

class BedpostX(BaseInterface):
    input_spec  = BedpostXInputSpec
    output_spec = BedpostXOutputSpec

    def _fslbin(self):
        from ..utils import FSL
        return os.path.join(FSL, "bin")

    def _run_interface(self, runtime):
        d = os.path.abspath(self.inputs.data_dir)
        req = [
            os.path.join(d, "data.nii.gz"),
            os.path.join(d, "bvals"),
            os.path.join(d, "bvecs"),
            os.path.join(d, "nodif_brain_mask.nii.gz"),
        ]
        missing = [p for p in req if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                "[bedpostx] Missing required inputs in data_dir:\n  " + "\n  ".join(missing)
            )

        exe = "bedpostx_gpu"
        if not os.path.isfile(os.path.join(self._fslbin(), exe)):
            exe = "bedpostx"
        
        _ = getattr(self.inputs, "trigger", None)

        subprocess.run([os.path.join(self._fslbin(), exe), d], check=True)
        self._bpx = d + ".bedpostX"
        return runtime

    def _list_outputs(self):
        o = self._outputs().get()
        o["bpx_dir"] = getattr(self, "_bpx", "")
        return o

# =========
# XTRACT (+ optional fsl_streamlines)
# =========

class XtractInputSpec(CommandLineInputSpec):
    # Accept either a BedpostX dir or a CUDIMOT dir (both optional here; one must exist)
    bpx_dir      = Directory(exists=True, mandatory=False, desc="BedpostX directory (…/data.bedpostX)")
    cudimot_dir  = Directory(exists=True, mandatory=False, desc="CUDIMOT dir (…/data.<ModelName>)")

    outdir    = Directory(mandatory=True, desc="Output directory for XTRACT results")

    # Standard space references and warps (conditionally required; see _run_interface)
    stdref         = File(exists=True, mandatory=False, desc="Template reference (HUM: MNI152_T1_1mm_brain; NHP: NMT FA)")
    warp_std2anat  = File(exists=True, mandatory=False, desc="Warp: template->subject")
    warp_anat2std  = File(exists=True, mandatory=False, desc="Warp: subject->template")

    # Species/mode controls
    mode     = traits.Enum("hum", "nhp", usedefault=False, desc="Species mode provided by the pipeline")
    species  = traits.Str("CUSTOM", usedefault=True, desc="Passthrough to xtract -species; overridden by mode when set")
    mac_ext  = traits.Str("_NMT", usedefault=True, desc="Used only when mode=='nhp' -> xtract -mac_ext")

    # Probtrackx options
    ptx_steplength = traits.Float(0.15, usedefault=True, desc="probtrackx2 --steplength (via ptx_options)")
    no_gpu   = traits.Bool(False, usedefault=True)

    # Profiles directory (optional for HUM; required for NHP). If provided, must contain structureList
    profiles_dir = Directory(exists=True, mandatory=False,
                             desc="XTRACT profiles directory that contains 'structureList'")

    # optional streamlines export with fsl_streamlines
    do_streamlines                = traits.Bool(True,  usedefault=True, desc="Export track files from XTRACT")
    streamlines_density_threshold = traits.Float(1e-3, usedefault=True, desc="density threshold (-t)")
    streamlines_format            = traits.Enum("trk", "vtk", usedefault=True, desc="output format (-f)")
    streamlines_ptx2_prefix       = traits.Str("densityNorm", usedefault=True, desc="probtrackx2 prefix (-p)")
    streamlines_num_jobs          = traits.Int(1, usedefault=True, desc="number of jobs/threads (-nj)")

    # tract viewer
    do_viewer     = traits.Bool(True, usedefault=True, desc="Generate XTRACT QC PNGs via tract_viewer.py")
    viewer_script = File(exists=True, desc="Path to tract_viewer.py")


class XtractOutputSpec(TraitedSpec):
    outdir = Directory()
    streamlines = traits.List(File(), desc="List of exported streamline files (.trk/.vtk)")


class Xtract(CommandLine):
    _cmd = "bash -lc true"
    input_spec  = XtractInputSpec
    output_spec = XtractOutputSpec

    # ---- helpers ----
    def _resolve_source(self) -> str:
        """
        Decide which directory to feed to `xtract -bpx`:
          - If bpx_dir is set and looks like a bedpostX dir, use it.
          - Else if cudimot_dir is set, use it.
          - Else error.
        """
        bpx = getattr(self.inputs, "bpx_dir", "") or ""
        cud = getattr(self.inputs, "cudimot_dir", "") or ""

        def _looks_bedpostx(p: str) -> bool:
            return os.path.isdir(p) and (
                p.endswith(".bedpostX")
                or os.path.isfile(os.path.join(p, "bvals"))
                or os.path.isfile(os.path.join(p, "bvecs"))
            )

        def _looks_cudimot(p: str) -> bool:
            return os.path.isdir(p) and (
                glob.glob(os.path.join(p, "Param_*"))
                or os.path.isfile(os.path.join(p, "dti_FA.nii.gz"))
            )

        if bpx and os.path.isdir(bpx):
            if _looks_bedpostx(bpx) or _looks_cudimot(bpx):
                return os.path.abspath(bpx)

        if cud and os.path.isdir(cud):
            return os.path.abspath(cud)

        raise ValueError(
            "[xtract] Provide either bpx_dir (…/data.bedpostX) or cudimot_dir (…/data.<ModelName>). "
            f"Got bpx_dir='{bpx}', cudimot_dir='{cud}'."
        )

    def _fsl_std_mni_brain(self) -> str:
        fsldir = os.environ.get("FSLDIR", "")
        if not fsldir:
            return ""
        cand = os.path.join(fsldir, "data", "standard", "MNI152_T1_1mm_brain.nii.gz")
        return cand if os.path.isfile(cand) else ""

    def _run_interface(self, runtime):
        outdir = os.path.abspath(self.inputs.outdir)
        os.makedirs(outdir, exist_ok=True)

        # Resolve source dir regardless of acquisition mode
        bpx_source = self._resolve_source()

        # Build a tiny ptx options file
        ptx_opts = os.path.join(outdir, "ptx_opts.txt")
        with open(ptx_opts, "w") as f:
            f.write(f"--steplength={self.inputs.ptx_steplength} --savepaths --opathdir")

        # Determine effective mode/species
        mode = None
        if isdefined(self.inputs.mode) and self.inputs.mode:
            mode = str(self.inputs.mode).lower()
        species = str(self.inputs.species or "").strip() or "CUSTOM"

        # Command base
        cmd = [
            os.path.join(FSL, "bin", "xtract"),
            "-bpx", bpx_source,
            "-out", outdir,
        ]

        # GPU toggle
        if not self.inputs.no_gpu:
            cmd.append("-gpu")

        # =========================
        # HUM mode
        # =========================
        if mode == "hum":
            cmd += ["-species", "HUMAN"]
            profiles_dir = getattr(self.inputs, "profiles_dir", "") or ""
            if profiles_dir:
                struct_list = os.path.join(os.path.abspath(profiles_dir), "structureList")
                if not os.path.isfile(struct_list):
                    raise FileNotFoundError(f"[xtract] structureList not found in {profiles_dir}")

                stdref = getattr(self.inputs, "stdref", "") or ""
                if not stdref:
                    stdref = self._fsl_std_mni_brain()
                    if not stdref:
                        raise FileNotFoundError(
                            "[xtract] HUM mode with custom profiles_dir requires stdref; "
                            "default MNI152_T1_1mm_brain.nii.gz not found in FSLDIR."
                        )

                cmd += ["-stdref", os.path.abspath(stdref),
                        "-p", os.path.abspath(profiles_dir),
                        "-str", os.path.abspath(struct_list)]
            
            if isdefined(self.inputs.warp_std2anat) and isdefined(self.inputs.warp_anat2std) \
               and self.inputs.warp_std2anat and self.inputs.warp_anat2std:
                cmd += ["-stdwarp",
                        os.path.abspath(self.inputs.warp_std2anat),
                        os.path.abspath(self.inputs.warp_anat2std)]

        # =========================
        # NHP mode
        # =========================
        elif mode == "nhp":
            os.environ["FSLINFMRIB"] = "1"
            mac_ext = str(getattr(self.inputs, "mac_ext", "_NMT") or "_NMT")
            cmd += ["-species", "MACAQUE", "-mac_ext", mac_ext]

            # Require stdref, warps, and profiles_dir (+structureList)
            stdref = getattr(self.inputs, "stdref", "") or ""
            if not stdref:
                raise ValueError("[xtract] NHP mode requires stdref (e.g., NMT FA).")
            if not (self.inputs.warp_std2anat and self.inputs.warp_anat2std):
                raise ValueError("[xtract] NHP mode requires both warp_std2anat and warp_anat2std.")
            profiles_dir = getattr(self.inputs, "profiles_dir", "") or ""
            if not profiles_dir:
                raise ValueError("[xtract] NHP mode requires profiles_dir containing structureList.")
            struct_list = os.path.join(os.path.abspath(profiles_dir), "structureList")
            if not os.path.isfile(struct_list):
                raise FileNotFoundError(f"[xtract] structureList not found in {profiles_dir}")

            cmd += [
                "-stdref", os.path.abspath(stdref),
                "-stdwarp", os.path.abspath(self.inputs.warp_std2anat), os.path.abspath(self.inputs.warp_anat2std),
                "-p", os.path.abspath(profiles_dir),
                "-str", os.path.abspath(struct_list),
            ]

        # =========================
        # No explicit mode: fall back to explicit species behavior
        # =========================
        else:
            if species.upper() == "HUMAN":
                cmd += ["-species", "HUMAN"]
            elif species.upper() == "MACAQUE":
                os.environ["FSLINFMRIB"] = "1"
                mac_ext = str(getattr(self.inputs, "mac_ext", "_NMT") or "_NMT")
                cmd += ["-species", "MACAQUE", "-mac_ext", mac_ext]
            else:
                cmd += ["-species", species]

            # If profiles_dir supplied, ensure structureList and std assets if needed
            profiles_dir = getattr(self.inputs, "profiles_dir", "") or ""
            if profiles_dir:
                struct_list = os.path.join(os.path.abspath(profiles_dir), "structureList")
                if not os.path.isfile(struct_list):
                    raise FileNotFoundError(f"[xtract] structureList not found in {profiles_dir}")
                cmd += ["-p", os.path.abspath(profiles_dir), "-str", os.path.abspath(struct_list)]

            # Pass stdref/stdwarp only if both warps are provided
            if isdefined(self.inputs.stdref) and self.inputs.stdref:
                cmd += ["-stdref", os.path.abspath(self.inputs.stdref)]
            if isdefined(self.inputs.warp_std2anat) and isdefined(self.inputs.warp_anat2std) \
               and self.inputs.warp_std2anat and self.inputs.warp_anat2std:
                cmd += ["-stdwarp",
                        os.path.abspath(self.inputs.warp_std2anat),
                        os.path.abspath(self.inputs.warp_anat2std)]

        # Always include probtrackx options file
        cmd += ["-ptx_options", ptx_opts]

        # Execute
        subprocess.run(cmd, check=True)

        # ---------- optional fsl_streamlines export ----------
        self._stream_files = []
        if self.inputs.do_streamlines:
            tracts_root = os.path.join(outdir, "tracts")
            tract_dirs = sorted([d for d in glob.glob(os.path.join(tracts_root, "*")) if os.path.isdir(d)])

            if not tract_dirs:
                raise RuntimeError(f"[fsl_streamlines] No tract folders found at {tracts_root}")

            fmt = self.inputs.streamlines_format  # "trk" | "vtk"
            ext = ".trk" if fmt == "trk" else ".vtk"

            for tract_dir in tract_dirs:
                tract_name = os.path.basename(tract_dir.rstrip(os.sep))
                out_prefix = os.path.join(tract_dir, tract_name)

                sl_cmd = [
                    "fsl_streamlines",
                    tract_dir,
                    "-o", out_prefix,
                    "-f", fmt,
                    "-p", str(self.inputs.streamlines_ptx2_prefix),
                    "-t", str(self.inputs.streamlines_density_threshold),
                    "-nj", str(int(self.inputs.streamlines_num_jobs)),
                ]
                subprocess.run(sl_cmd, check=True)
                self._stream_files.append(out_prefix + ext)

        # ---------- optional tract viewer ----------
        try:
            if getattr(self.inputs, "do_viewer", False) and \
               isdefined(self.inputs.viewer_script) and self.inputs.viewer_script:
                tracts_root = os.path.join(outdir, "tracts")
                viewer_cmd = [
                    os.path.abspath(self.inputs.viewer_script),
                    "--xtract", tracts_root,
                    "--brain", os.path.abspath(self.inputs.stdref) if isdefined(self.inputs.stdref) and self.inputs.stdref
                               else self._fsl_std_mni_brain(),
                    "--outfile", outdir
                ]
                subprocess.run(viewer_cmd, check=True)
        except Exception:
            # Non-fatal if fsleyes or viewer is not available on this machine
            pass

        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        outs = self._outputs().get()
        outs["outdir"] = os.path.abspath(self.inputs.outdir)
        outs["streamlines"] = getattr(self, "_stream_files", [])
        return outs
