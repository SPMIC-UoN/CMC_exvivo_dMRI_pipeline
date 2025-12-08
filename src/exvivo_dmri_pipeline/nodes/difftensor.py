#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
from typing import List, Tuple, Optional

import numpy as np

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    Directory,
    traits,
    Undefined
)

from ..utils import FSL


class DiffTensorInputSpec(BaseInterfaceInputSpec):
    # Inputs from earlier nodes
    imain  = File(exists=True, mandatory=True,
                  desc="spatial drift corrected 4D diffusion")
    bvals  = File(exists=True, mandatory=True, desc="bvals")
    bvecs  = File(exists=True, mandatory=True, desc="bvecs")
    mask   = File(exists=True, mandatory=True, desc="nodif_brain_mask.nii.gz")

    # Where processed data live (parent for data_lowershell and dtifit/ or CUDIMOT)
    subject_dir  = Directory(mandatory=True,
                             desc="processing data dir (usually <proc>/data)")

    # Sidecars for SSFP (omit/leave unset for SE)
    sidecars_dir = Directory(exists=False, mandatory=False,
                             desc="<proc>/sidecars; required for SSFP/CUDIMOT DTI")

    # Optional explicit acqtype. If left unset, auto-detects.
    acqtype = traits.Enum("ssfp", "se", usedefault=False,
                          desc="If unset, auto-detects from other inputs")

    # FixP from WritePathsFile (SSFP only; provides T1/T2(/B1) for DTI_SSFP)
    fixp_path = File(exists=False, mandatory=False,
                     desc="FixP file produced by WritePathsFile (SSFP only)")

    # Lower shell selection controls
    approx_b   = traits.Int(usedefault=True, value=0,
                            desc="target shell (lowest nonzero auto-selected if 0)")
    db         = traits.Int(usedefault=True, value=60,
                            desc="tolerance for b-value selection")
    b0         = traits.Int(usedefault=True, value=0,
                            desc="b0 value (kept by default)")

    # CUDIMOT DTI pipeline controls (SSFP only)
    dti_pipeline_cmd = traits.Str(
        "Pipeline_DTI_SSFP.sh",
        usedefault=True,
        desc="CUDIMOT DTI pipeline script on PATH"
    )
    njobs        = traits.Int(usedefault=True, value=4, desc="-NJOBS")
    run_mcmc     = traits.Bool(True,  usedefault=True, desc="--runMCMC")
    no_levmar    = traits.Bool(True,  usedefault=True, desc="--no_LevMar")
    burnin       = traits.Int(usedefault=True, value=10000, desc="-b burn-in")
    jumps        = traits.Int(usedefault=True, value=1250, desc="-j jumps")
    sample_every = traits.Int(usedefault=True, value=25,   desc="-s sample every")


class DiffTensorOutputSpec(TraitedSpec):
    lowershell_dir = Directory(desc="data_lowershell directory with data, bvals, bvecs, mask")
    dtifit_dir     = Directory(desc="dtifit or CUDIMOT DTI output directory")
    fa_file        = File(desc="FA image (dti_FA.nii.gz or equivalent)")
    tensor_file    = File(desc="Tensor image (dti_tensor.nii.gz or equivalent)")
    cudimot_dir    = Directory(desc="CUDIMOT DTI dir (SSFP only; empty for SE)")


class DiffTensor(BaseInterface):
    input_spec  = DiffTensorInputSpec
    output_spec = DiffTensorOutputSpec

    # ---------------------------
    # Basic helpers
    # ---------------------------

    def _read_1d(self, path: str) -> np.ndarray:
        arr = np.loadtxt(path, dtype=float)
        return arr.reshape(-1) if arr.ndim > 1 else arr

    def _write_1d(self, path: str, arr: np.ndarray, fmt: str = "%.8f") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savetxt(path, arr.reshape(1, -1), fmt=fmt)

    def _auto_approx_b(self, bvals_path: str, db: int) -> int:
        b = self._read_1d(bvals_path)
        nz = sorted(set(int(v) for v in b if v >= db))
        if not nz:
            raise RuntimeError(f"[difftensor] Could not auto-detect approx_b from {bvals_path}")
        return nz[0]

    def _parse_indices_from_select_stdout(self, text: str) -> List[int]:
        lines = (text or "").splitlines()
        cands = [ln for ln in lines if "volume list" in ln.lower()]
        if not cands:
            return []
        last = cands[-1]
        part = last.split("=", 1)[1].strip() if "=" in last else last.strip()
        try:
            if " " in part and "," in part and part.index(" ") < part.index(","):
                part = part.split(" ", 1)[0]
        except Exception:
            pass
        try:
            return [int(s.strip()) for s in part.split(",") if s.strip() != ""]
        except Exception:
            return []

    def _compute_indices_fallback(self, bvals: np.ndarray, approx_b: int, db: int) -> List[int]:
        tol = int(db)
        i_b0 = [i for i, b in enumerate(bvals) if abs(b - 0) < tol]
        i_bs = [i for i, b in enumerate(bvals) if abs(b - approx_b) < tol]
        return i_b0 + i_bs

    def _run_select_and_capture(
        self,
        data: str,
        bvals: str,
        bvecs: str,
        outprefix: str,
        approx_b: int,
        b0: int,
        db: int,
    ) -> Tuple[List[int], str, str, str]:
        cmd = [
            os.path.join(FSL, "bin", "select_dwi_vols"),
            os.path.abspath(data),
            os.path.abspath(bvals),
            os.path.abspath(outprefix),
            str(int(b0)),
            "-b", str(int(approx_b)),
            "-db", str(int(db)),
            "-obv", os.path.abspath(bvecs),
        ]
        res = subprocess.run(cmd, check=True, text=True, capture_output=True)
        combined = (res.stdout or "") + "\n" + (res.stderr or "")
        idx = self._parse_indices_from_select_stdout(combined)
        out_nii = outprefix + ".nii.gz"
        return idx, outprefix + ".bval", outprefix + ".bvec", out_nii

    def _run_quiet(self, args: List[str], label: str = "command") -> None:
        try:
            subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            msg = (e.stderr or e.stdout or str(e)).strip()
            raise RuntimeError(f"{label} failed: {' '.join(args)}\n{msg}") from e

    def _list_outputs(self):
        out = {
            "lowershell_dir": Undefined,
            "dtifit_dir":     Undefined,
            "fa_file":        Undefined,
            "tensor_file":    Undefined,
            "cudimot_dir":    Undefined,
        }

        # Use attributes set during _run_interface; guard with hasattr
        if hasattr(self, "_lowershell_dir") and self._lowershell_dir:
            out["lowershell_dir"] = os.path.abspath(self._lowershell_dir)

        if hasattr(self, "_dtifit_dir") and self._dtifit_dir:
            out["dtifit_dir"] = os.path.abspath(self._dtifit_dir)

        if hasattr(self, "_fa_file") and self._fa_file and os.path.isfile(self._fa_file):
            out["fa_file"] = os.path.abspath(self._fa_file)

        if hasattr(self, "_tensor_file") and self._tensor_file and os.path.isfile(self._tensor_file):
            out["tensor_file"] = os.path.abspath(self._tensor_file)

        if hasattr(self, "_cudimot_dir") and self._cudimot_dir and os.path.isdir(self._cudimot_dir):
            out["cudimot_dir"] = os.path.abspath(self._cudimot_dir)

        return out

    # ---------------------------
    # CUDIMOT helpers (SSFP)
    # ---------------------------

    def _model_name_from_pipeline(self, pipeline_cmd: str) -> str:
        base = os.path.basename(pipeline_cmd).replace(".sh", "")
        if base.startswith("Pipeline_"):
            base = base[len("Pipeline_"):]
        return base

    def _finish_script_for_model(self, model_name: str) -> str:
        cand_py = f"{model_name}_finish.py"
        cand_sh = f"{model_name}_finish.sh"
        if shutil.which(cand_py):
            return cand_py
        if shutil.which(cand_sh):
            return cand_sh
        return cand_py

    def _validate_fixp(self, fixp_path: str) -> None:
        if not os.path.isfile(fixp_path):
            raise RuntimeError(f"[difftensor] FixP not found at {fixp_path}. "
                               f"WritePathsFile must run before DiffTensor.")
        with open(fixp_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines()]
        if len(lines) < 2:
            raise RuntimeError(f"[difftensor] FixP malformed (needs at least T1 and T2): {fixp_path}")
        t1, t2 = lines[0], lines[1]
        if not (t1 and os.path.isfile(t1)):
            raise RuntimeError(f"[difftensor] FixP T1 path invalid: {t1}")
        if not (t2 and os.path.isfile(t2)):
            raise RuntimeError(f"[difftensor] FixP T2 path invalid: {t2}")

    def _param_guess(self, cdir: str, idx: int) -> str:
        p1 = os.path.join(cdir, f"Param_{idx}_samples.nii.gz")
        p2 = os.path.join(cdir, f"Param_{idx}.nii.gz")
        if os.path.isfile(p1):
            return p1
        if os.path.isfile(p2):
            return p2
        raise FileNotFoundError(f"Missing parameter volume for Param_{idx}: tried {p1} and {p2}")

    def _write_manifest(self, lines: List[str], path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for ln in lines:
                f.write(str(ln).strip() + "\n")
        return os.path.abspath(path)

    # ---------------------------
    # Main
    # ---------------------------
    def _run_interface(self, runtime):
        parent   = os.path.abspath(self.inputs.subject_dir)   # <proc>/data
        data_nii = os.path.abspath(self.inputs.imain)

        # Decide mode:
        #  - If acqtype explicitly provided -> use it
        #  - Else if both sidecars_dir and fixp_path exist -> SSFP
        #  - Else -> SE
        mode = getattr(self.inputs, "acqtype", None)
        if mode not in ("ssfp", "se"):
            has_sidecars = bool(getattr(self.inputs, "sidecars_dir", "")) and os.path.isdir(self.inputs.sidecars_dir or "")
            has_fixp     = bool(getattr(self.inputs, "fixp_path", "")) and os.path.isfile(self.inputs.fixp_path or "")
            mode = "ssfp" if (has_sidecars and has_fixp) else "se"

        # Prepare <proc>/data_lowershell
        ls_dir = os.path.join(os.path.dirname(parent), "data_lowershell")
        os.makedirs(ls_dir, exist_ok=True)

        # Select target shell (plus b0s)
        bvals_full_path = os.path.abspath(self.inputs.bvals)
        approx_b = int(self.inputs.approx_b) or self._auto_approx_b(bvals_full_path, int(self.inputs.db))

        outprefix = os.path.join(ls_dir, "data")
        idx, ls_bval, ls_bvec, ls_nii = self._run_select_and_capture(
            data=data_nii,
            bvals=bvals_full_path,
            bvecs=os.path.abspath(self.inputs.bvecs),
            outprefix=outprefix,
            approx_b=approx_b,
            b0=int(self.inputs.b0),
            db=int(self.inputs.db),
        )
        if not os.path.isfile(ls_nii):
            raise RuntimeError(f"[difftensor] select_dwi_vols did not create {ls_nii}")

        if not idx:
            print("[difftensor] Warning: could not parse index list; using tolerance-based selection.")
            full_bvals = self._read_1d(bvals_full_path)
            idx = self._compute_indices_fallback(full_bvals, approx_b, int(self.inputs.db))

        take = np.asarray(idx, dtype=int)

        # Canonical names for lower-shell bvals/bvecs
        ls_bvals = os.path.join(ls_dir, "bvals")
        ls_bvecs = os.path.join(ls_dir, "bvecs")
        if os.path.isfile(ls_bval):
            os.replace(ls_bval, ls_bvals)
        else:
            raise RuntimeError(f"[difftensor] Missing output: {ls_bval}")
        if os.path.isfile(ls_bvec):
            os.replace(ls_bvec, ls_bvecs)
        else:
            raise RuntimeError(f"[difftensor] Missing output: {ls_bvec}")

        # Ensure the mask is present inside data_lowershell as nodif_brain_mask.nii.gz
        ls_mask = os.path.join(ls_dir, "nodif_brain_mask.nii.gz")
        if not os.path.isfile(ls_mask):
            shutil.copy2(os.path.abspath(self.inputs.mask), ls_mask)

        # ---------------------------------------------------------------------
        # Branch: SE → dtifit on lower shell
        # ---------------------------------------------------------------------
        if mode == "se":
            dtifit_dir = os.path.join(os.path.dirname(parent), "dtifit")
            os.makedirs(dtifit_dir, exist_ok=True)
            out_prefix = os.path.join(dtifit_dir, "dti")

            self._run_quiet(
                [
                    os.path.join(FSL, "bin", "dtifit"),
                    "-k", ls_nii,
                    "-o", out_prefix,
                    "-m", ls_mask,
                    "-r", ls_bvecs,
                    "-b", ls_bvals,
                    "--save_tensor",
                ],
                label="dtifit",
            )

            fa  = out_prefix + "_FA.nii.gz"
            ten = out_prefix + "_tensor.nii.gz"
            if not os.path.isfile(fa):
                print(f"[difftensor] Warning: FA not found at {fa}")
            if not os.path.isfile(ten):
                print(f"[difftensor] Warning: tensor not found at {ten}")

            self._lowershell_dir = ls_dir
            self._dtifit_dir     = dtifit_dir
            self._fa_file        = fa if os.path.isfile(fa) else ""
            self._tensor_file    = ten if os.path.isfile(ten) else ""
            self._cudimot_dir    = ""

            return runtime

        # ---------------------------------------------------------------------
        # Branch: SSFP → CUDIMOT DTI (Pipeline_DTI_SSFP.sh)
        # ---------------------------------------------------------------------
        sidecars_src = os.path.abspath(self.inputs.sidecars_dir) if getattr(self.inputs, "sidecars_dir", "") else ""
        if not sidecars_src or not os.path.isdir(sidecars_src):
            raise RuntimeError("[difftensor] SSFP mode requires a valid sidecars_dir (<proc>/sidecars).")

        fixp_path = os.path.abspath(self.inputs.fixp_path) if getattr(self.inputs, "fixp_path", "") else ""
        self._validate_fixp(fixp_path)

        # Slice sidecars to lower shell
        def _slice_and_write(name: str, fmt: str):
            src = os.path.join(sidecars_src, name)
            if not os.path.isfile(src):
                raise FileNotFoundError(f"[difftensor] Sidecar not found for SSFP DTI: {src}")
            arr = self._read_1d(src)
            try:
                sub = arr[take]
            except Exception as e:
                raise RuntimeError(f"[difftensor] Failed to slice {name} with indices {take}: {e}")
            dst = os.path.join(ls_dir, name)
            self._write_1d(dst, sub, fmt=fmt)
            return dst

        trs_path      = _slice_and_write("TRs",          fmt="%.4f")
        amps_path     = _slice_and_write("diffGradAmps", fmt="%.4f")
        durs_path     = _slice_and_write("diffGradDurs", fmt="%.5f")
        flips_path    = _slice_and_write("flipAngles",   fmt="%d")
        noisefloor_p  = _slice_and_write("noisefloor",   fmt="%d")
        
        cfp_lines = [
            ls_bvecs,                       # CFP[0:2] bvecs
            trs_path,                       # CFP[3]   TRs
            amps_path,                      # CFP[4]   diffGradAmps
            durs_path,                      # CFP[5]   diffGradDurs
            flips_path,                     # CFP[6]   flipAngle
            noisefloor_p,                   # CFP[7]   Noisefloor
        ]

        self._write_manifest(cfp_lines, os.path.join(ls_dir, "CFP"))

        pipeline = (self.inputs.dti_pipeline_cmd or "Pipeline_DTI_SSFP.sh").strip()
        model    = self._model_name_from_pipeline(pipeline)  # e.g. "DTI_SSFP"

        cudimot_args = [pipeline, ls_dir]
        if int(getattr(self.inputs, "njobs", 4)) > 0:
            cudimot_args += ["-NJOBS", str(int(self.inputs.njobs))]
        if getattr(self.inputs, "no_levmar", True):
            cudimot_args += ["--no_LevMar"]
        if getattr(self.inputs, "run_mcmc", True):
            cudimot_args += [
                "--runMCMC",
                "-b", str(int(self.inputs.burnin)),
                "-j", str(int(self.inputs.jumps)),
                "-s", str(int(self.inputs.sample_every)),
            ]
        cudimot_args += [
            f"--CFP={os.path.join(ls_dir, 'CFP')}",
            f"--FixP={fixp_path}",
        ]

        self._run_quiet(cudimot_args, label="Pipeline_DTI_SSFP")

        cudimot_dir = f"{ls_dir}.{model}"
        if not os.path.isdir(cudimot_dir):
            raise RuntimeError(f"[difftensor] Expected CUDIMOT output dir not found: {cudimot_dir}")

        finish_script = self._finish_script_for_model(model)
        # Run model-specific finish script first
        if finish_script.endswith(".py"):
            lam1  = self._param_guess(cudimot_dir, 0)
            lam2  = self._param_guess(cudimot_dir, 1)
            lam3  = self._param_guess(cudimot_dir, 2)
            theta = self._param_guess(cudimot_dir, 3)
            phi   = self._param_guess(cudimot_dir, 4)
            psi   = self._param_guess(cudimot_dir, 5)
            s0    = self._param_guess(cudimot_dir, 6)

            fin_args = [
                finish_script,
                "--lam1",  lam1,
                "--lam2",  lam2,
                "--lam3",  lam3,
                "--theta", theta,
                "--phi",   phi,
                "--psi",   psi,
                "--s0",    s0,
                "--mask",  ls_mask,
                "--outdir", cudimot_dir,
            ]
            self._run_quiet(fin_args, label=f"{model}_finish")
        else:
            self._run_quiet([finish_script, cudimot_dir], label=f"{model}_finish")

        # Paths produced by the finish step
        fa_path   = os.path.join(cudimot_dir, "dti_FA.nii.gz")
        tens_path = os.path.join(cudimot_dir, "dti_tensor.nii.gz")

        # Build tensor if the finish step didn't create it
        if not os.path.isfile(tens_path):
            v1_path = os.path.join(cudimot_dir, "dti_V1.nii.gz")
            v2_path = os.path.join(cudimot_dir, "dti_V2.nii.gz")
            v3_path = os.path.join(cudimot_dir, "dti_V3.nii.gz")
            l1_path = os.path.join(cudimot_dir, "dti_L1.nii.gz")
            l2_path = os.path.join(cudimot_dir, "dti_L2.nii.gz")
            l3_path = os.path.join(cudimot_dir, "dti_L3.nii.gz")

            for p in (v1_path, v2_path, v3_path, l1_path, l2_path, l3_path):
                if not os.path.isfile(p):
                    raise RuntimeError(f"[difftensor] Missing DTI component: {p}")

            # Resolve the installed path to tensorcalc.py from this package
            import importlib, os as _os
            _pkg = importlib.import_module("exvivo_dmri_pipeline")
            _pkg_dir = _os.path.dirname(_pkg.__file__)
            tensorcalc_py = _os.path.join(_pkg_dir, "nodes", "tensorcalc.py")
            if not _os.path.isfile(tensorcalc_py):
                raise RuntimeError(f"[difftensor] tensorcalc.py not found at {tensorcalc_py}")

            # Run tensorcalc with FSL's Python so fslpy is available
            calc_args = [
                _os.path.join(FSL, "bin", "fslpython"),
                tensorcalc_py,
                "--v1", v1_path,
                "--v2", v2_path,
                "--v3", v3_path,
                "--l1", l1_path,
                "--l2", l2_path,
                "--l3", l3_path,
                "--out", tens_path,
            ]
            self._run_quiet(calc_args, label="tensorcalc")

        # Finalise outputs
        if not os.path.isfile(fa_path):
            print(f"[difftensor] Warning: FA not found at {fa_path}")
        if not os.path.isfile(tens_path):
            print(f"[difftensor] Warning: tensor not found at {tens_path}")

        self._lowershell_dir = ls_dir
        self._dtifit_dir     = cudimot_dir   # for backwards compatibility
        self._fa_file        = fa_path if os.path.isfile(fa_path) else ""
        self._tensor_file    = tens_path if os.path.isfile(tens_path) else ""
        self._cudimot_dir    = cudimot_dir

        return runtime
