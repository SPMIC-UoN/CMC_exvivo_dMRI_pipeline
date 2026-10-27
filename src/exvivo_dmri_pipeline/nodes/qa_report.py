# src/exvivo_dmri_pipeline/qa_report.py
import os
import json
import datetime
import subprocess
import html
import shutil
import textwrap
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yaml

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    Directory,
    traits,
    SimpleInterface,
    isdefined,
)


class QAReportInputSpec(BaseInterfaceInputSpec):
    # General
    dmri_root = traits.Str(desc="Root directory for this dMRI session")
    outdir = traits.Str(desc="Output directory for QA report")
    pipeline_yaml = File(exists=False, desc="Copy of the pipeline YAML (for reference)")

    # Eddy QC
    eddy_qc_json = File(exists=False, desc="FSL eddy_qc qc.json file (optional)")

    # Mode: how pipeline was run
    run_mode = traits.Enum(
        "preproc_only",
        "full",
        "tract_only",
        usedefault=True,
        desc="Pipeline run mode for this QA ('preproc_only', 'full', or 'tract_only')",
    )

    # Nipype logs and graphs
    nipype_dir_preproc = traits.Str(
        desc="Nipype log directory for main preprocessing/full run"
    )
    nipype_dir_tract = traits.Str(
        desc="Nipype log directory for tract-only run (if present)"
    )
    nipype_graph_preproc = File(
        exists=False, desc="Path to Nipype workflow graph image for main run"
    )
    nipype_graph_tract = File(
        exists=False, desc="Path to Nipype workflow graph image for tract-only run"
    )

    # Standard space (generic, not NMT-specific)
    std_mask_img = File(
        exists=False,
        desc="Standard-space brain mask or template image for outlining (e.g., FA template or brain mask)",
    )
    std_fa_img = File(exists=False, desc="Standard-space FA template")

    # Native diffusion-space inputs
    data_4d = File(exists=False, desc="4D diffusion data (post-EDDY if applicable)")
    bvals = File(exists=False, desc="B-values file")
    bvecs = File(exists=False, desc="B-vectors file")
    mask_file = File(exists=False, desc="Brain mask in diffusion space")
    fa_img = File(exists=False, desc="Native-space FA image (DTIFIT)")
    v1_img = File(exists=False, desc="Primary eigenvector image (DTIFIT V1)")

    # XTRACT outputs
    xtract_dir = File(exists=False, desc="XTRACT output directory root")

    # Dummy input to force run-last in the workflow
    deps = traits.Any(desc="Dependencies to force QA run last")


class QAReportOutputSpec(TraitedSpec):
    report_pdf = File(desc="Path to the generated PDF QA report")
    report_pdf_dark = File(desc="Dark PDF export")
    report_html = File(desc="Theme-switchable HTML report")
    manifest_json = File(desc="QA sources and issues")
    figures_dir = Directory(desc="Individual QA panels")


class QAReport(SimpleInterface):
    """Generate a multi-page PDF QA report (ex vivo)."""

    input_spec = QAReportInputSpec
    output_spec = QAReportOutputSpec

    # ------------- helpers -------------

    @staticmethod
    def _safe_load_img(path: str):
        if not path or not os.path.exists(path):
            return None
        try:
            return nib.load(path)
        except Exception:
            return None

    @staticmethod
    def _safe_json(path: str):
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _world_cog_from_mask(img: nib.Nifti1Image) -> np.ndarray:
        data = img.get_fdata()
        mask = data > 0
        if not np.any(mask):
            return np.array([np.nan, np.nan, np.nan])
        coords = np.argwhere(mask)
        cog_vox = coords.mean(axis=0)
        return nib.affines.apply_affine(img.affine, cog_vox)

    @staticmethod
    def _fmt_coord(arr):
        if arr is None or not np.all(np.isfinite(arr)):
            return "N/A"
        return f"[{arr[0]:.2f}, {arr[1]:.2f}, {arr[2]:.2f}]"

    # -------- fsleyes helpers --------

    def _run_fsleyes(self, args):
        """Run fsleyes render and fail gracefully if unavailable."""
        try:
            subprocess.run(
                args,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    def _fsleyes_brain_mask(self, fa_path, mask_path, out_png, world_xyz):
        """Native-space FA + brain mask overlay (red, alpha=50%)."""
        if not (fa_path and mask_path):
            return

        cmd = [
            "fsleyes", "render", "--outfile", out_png, "--size", "1800", "800",
            "--scene", "ortho",
            "--worldLoc", f"{world_xyz[0]}", f"{world_xyz[1]}", f"{world_xyz[2]}",
            "--xzoom", "85", "--yzoom", "85", "--zzoom", "85",
            "--showLocation", "no", "--layout", "horizontal", "--hideCursor",
            "--bgColour", "0.0", "0.0", "0.0", "--fgColour", "1.0", "1.0", "1.0",
            fa_path, "--overlayType", "volume", "--cmap", "greyscale",
            "--brightness", "60", "--contrast", "85", "--volume", "0",
            mask_path, "--overlayType", "mask", "--alpha", "50", "--cmap", "red", "--volume", "0",
        ]
        self._run_fsleyes(cmd)

    def _fsleyes_color_fa(self, fa_path, v1_path, out_png, world_xyz):
        """Colour FA map via rgbvector V1, modulated by FA."""
        if not (fa_path and v1_path):
            return

        cmd = [
            "fsleyes", "render", "--outfile", out_png, "--size", "2000", "800",
            "--scene", "ortho",
            "--worldLoc", f"{world_xyz[0]}", f"{world_xyz[1]}", f"{world_xyz[2]}",
            "--xzoom", "85", "--yzoom", "85", "--zzoom", "85",
            "--showLocation", "no", "--layout", "horizontal", "--hideCursor",
            "--bgColour", "0.0", "0.0", "0.0", "--fgColour", "1.0", "1.0", "1.0",
            fa_path, "--disabled", "--overlayType", "volume", "--cmap", "greyscale",
            "--displayRange", "0.0", "98%", "--volume", "0",
            v1_path, "--overlayType", "rgbvector", "--alpha", "100.0",
            "--brightness", "65", "--contrast", "72",
            "--modulateImage", fa_path, "--modulateRange", "0.0", "1",
            "--modulateMode", "brightness", "--suppressMode", "white",
        ]
        self._run_fsleyes(cmd)

    def _fsleyes_std_outline(self, fa_std_path, std_mask_path, out_png, world_xyz, thr="0.2"):
        """FA_in_STD with generic standard-space outline mask."""
        if not (fa_std_path and std_mask_path):
            return
        cmd = [
            "fsleyes", "render", "--outfile", out_png, "--size", "1800", "800",
            "--scene", "ortho",
            "--worldLoc", f"{world_xyz[0]}", f"{world_xyz[1]}", f"{world_xyz[2]}",
            "--xzoom", "85", "--yzoom", "85", "--zzoom", "85",
            "--showLocation", "no", "--layout", "horizontal", "--hideCursor",
            "--bgColour", "0.0", "0.0", "0.0", "--fgColour", "1.0", "1.0", "1.0",
            fa_std_path, "--overlayType", "volume", "--cmap", "greyscale",
            "--displayRange", "0.0", "98%", "--volume", "0",
            std_mask_path, "--overlayType", "mask",
            "--maskColour", "1.0", "0.0", "0.0",
            "--threshold", thr, "1",
            "--outline", "--outlineWidth", "3", "--volume", "0",
        ]
        self._run_fsleyes(cmd)

    # -------- main interface --------

    def _run_interface(self, runtime):
        # The modern report is self-contained and deliberately follows only
        # pipeline-owned paths. Keep the legacy implementation below temporarily
        # for source compatibility, but do not execute it.
        modern = _ModernReport(self.inputs)
        self._results.update({key: str(value) for key, value in modern.build().items()})
        return runtime

        # Legacy report implementation (unreachable; retained for now).
        # --------------------------------
        # Setup paths
        # --------------------------------
        dmri_root = self.inputs.dmri_root or ""
        outdir = self.inputs.outdir or dmri_root or os.getcwd()
        os.makedirs(outdir, exist_ok=True)

        pdf_path = os.path.join(outdir, "CMC_exvivo_QA.pdf")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --------------------------------
        # Load core images and metadata
        # --------------------------------
        data_img = self._safe_load_img(self.inputs.data_4d) if isdefined(self.inputs.data_4d) else None

        data_shape = None
        voxel_size = None
        if data_img is not None:
            data_shape = data_img.shape
            voxel_size = data_img.header.get_zooms()[:3]

        native_mask_img = self._safe_load_img(self.inputs.mask_file) if isdefined(self.inputs.mask_file) else None
        native_fa_img = self._safe_load_img(self.inputs.fa_img) if isdefined(self.inputs.fa_img) else None
        v1_img = self._safe_load_img(self.inputs.v1_img) if isdefined(self.inputs.v1_img) else None

        # Subject FA in standard space (pipeline convention)
        fa_std_path = None
        if dmri_root:
            cand = os.path.join(dmri_root, "stdreg", "FA_in_STD.nii.gz")
            if os.path.exists(cand):
                fa_std_path = cand

        # Standard-space mask/outline image (generic)
        std_mask_img = self._safe_load_img(self.inputs.std_mask_img) if isdefined(self.inputs.std_mask_img) else None
        std_fa_img = self._safe_load_img(fa_std_path) if fa_std_path else None
        if std_mask_img is None and std_fa_img is not None:
            std_mask_img = std_fa_img  # fallback

        # Centres of gravity
        native_cog_world = self._world_cog_from_mask(native_mask_img) if native_mask_img is not None else None
        std_cog_world = self._world_cog_from_mask(std_mask_img) if std_mask_img is not None else None

        # --------------------------------
        # B-values and B-vectors
        # --------------------------------
        bvals_arr = None
        bvecs_arr = None
        unique_shells = []
        n_b0 = None
        n_dw = None

        if isdefined(self.inputs.bvals) and self.inputs.bvals and os.path.exists(self.inputs.bvals):
            try:
                bvals_arr = np.atleast_1d(np.loadtxt(self.inputs.bvals))
                n_b0 = int(np.sum(bvals_arr == 0))
                n_total = bvals_arr.size
                n_dw = int(n_total - n_b0)
                nonzero = bvals_arr[bvals_arr > 0]
                unique_shells = sorted(np.unique(nonzero.astype(int)).tolist()) if nonzero.size else []
            except Exception:
                bvals_arr = None

        if isdefined(self.inputs.bvecs) and self.inputs.bvecs and os.path.exists(self.inputs.bvecs):
            try:
                bvecs_arr = np.asarray(np.loadtxt(self.inputs.bvecs))
                if bvecs_arr.ndim == 2:
                    if bvecs_arr.shape[0] == 3 and bvecs_arr.shape[1] != 3:
                        bvecs_arr = bvecs_arr.T
                    if bvecs_arr.shape[1] != 3:
                        bvecs_arr = None
                else:
                    bvecs_arr = None
            except Exception:
                bvecs_arr = None

        # --------------------------------
        # Eddy QC JSON and images
        # --------------------------------
        eddy_qc = self._safe_json(self.inputs.eddy_qc_json) if (isdefined(self.inputs.eddy_qc_json) and self.inputs.eddy_qc_json) else None
        qc_cnr_avg = eddy_qc.get("qc_cnr_avg") if isinstance(eddy_qc, dict) else None
        qc_cnr_std = eddy_qc.get("qc_cnr_std") if isinstance(eddy_qc, dict) else None
        b_shells = eddy_qc.get("b_shells") if isinstance(eddy_qc, dict) else None
        qc_dir = os.path.dirname(self.inputs.eddy_qc_json) if (isdefined(self.inputs.eddy_qc_json) and self.inputs.eddy_qc_json and os.path.exists(self.inputs.eddy_qc_json)) else None

        avg_pngs, cnr_pngs = [], []
        if qc_dir and os.path.isdir(qc_dir):
            for fname in sorted(os.listdir(qc_dir)):
                if fname.startswith("avg_b") and fname.endswith(".png"):
                    avg_pngs.append(os.path.join(qc_dir, fname))
                if fname.startswith("cnr") and fname.endswith(".png"):
                    cnr_pngs.append(os.path.join(qc_dir, fname))

        # --------------------------------
        # XTRACT PNGs
        # --------------------------------
        xtract_pngs = []
        if isdefined(self.inputs.xtract_dir) and self.inputs.xtract_dir:
            xdir = self.inputs.xtract_dir
            if os.path.isdir(xdir):
                for name in ("xtract_sagittal.png", "xtract_coronal.png", "xtract_axial.png"):
                    fpath = os.path.join(xdir, name)
                    if os.path.exists(fpath):
                        xtract_pngs.append(fpath)

        # --------------------------------
        # Nipype logs
        # --------------------------------
        nipype_dir_preproc = self.inputs.nipype_dir_preproc if isdefined(self.inputs.nipype_dir_preproc) else ""
        nipype_dir_tract = self.inputs.nipype_dir_tract if isdefined(self.inputs.nipype_dir_tract) else ""

        nipype_graph_preproc = self.inputs.nipype_graph_preproc if (isdefined(self.inputs.nipype_graph_preproc) and self.inputs.nipype_graph_preproc and os.path.exists(self.inputs.nipype_graph_preproc)) else None
        nipype_graph_tract = self.inputs.nipype_graph_tract if (isdefined(self.inputs.nipype_graph_tract) and self.inputs.nipype_graph_tract and os.path.exists(self.inputs.nipype_graph_tract)) else None

        # --------------------------------
        # Run mode description
        # --------------------------------
        mode = self.inputs.run_mode
        if mode == "preproc_only":
            mode_desc = "Preprocessing only (up to DTIFIT + registration)."
        elif mode == "full":
            mode_desc = "Preprocessing and tractography (XTRACT) run together."
        else:
            mode_desc = "Tract-only run using existing preprocessing outputs."

        # --------------------------------
        # fsleyes visualisations
        # --------------------------------
        brain_mask_png = os.path.join(outdir, "fa_mask_native.png")
        dec_fa_png = os.path.join(outdir, "dec_fa_native.png")
        std_outline_png = os.path.join(outdir, "fa_in_std_outline.png")

        if native_cog_world is not None and np.all(np.isfinite(native_cog_world)):
            if (isdefined(self.inputs.fa_img) and self.inputs.fa_img and os.path.exists(self.inputs.fa_img) and
                isdefined(self.inputs.mask_file) and self.inputs.mask_file and os.path.exists(self.inputs.mask_file)):
                self._fsleyes_brain_mask(self.inputs.fa_img, self.inputs.mask_file, brain_mask_png, native_cog_world)

            if (isdefined(self.inputs.fa_img) and self.inputs.fa_img and os.path.exists(self.inputs.fa_img) and
                isdefined(self.inputs.v1_img) and self.inputs.v1_img and os.path.exists(self.inputs.v1_img)):
                self._fsleyes_color_fa(self.inputs.fa_img, self.inputs.v1_img, dec_fa_png, native_cog_world)

        if (std_cog_world is not None and np.all(np.isfinite(std_cog_world)) and
            fa_std_path is not None and os.path.exists(fa_std_path) and
            isdefined(self.inputs.std_mask_img) and self.inputs.std_mask_img and os.path.exists(self.inputs.std_mask_img)):
            self._fsleyes_std_outline(fa_std_path, self.inputs.std_mask_img, std_outline_png, std_cog_world, thr="0.2")

        # --------------------------------
        # Build PDF pages
        # --------------------------------
        with PdfPages(pdf_path) as pdf:
            # -------------------------
            # Page 1: overview
            # -------------------------
            fig = plt.figure(figsize=(8.27, 11.69))  # A4-ish
            fig.suptitle("CMC ex vivo dMRI QA report", fontsize=16, y=0.96)

            text_lines = [
                f"Generated on {timestamp}.",
                "",
                f"Session root: {dmri_root if dmri_root else 'N/A'}",
                f"Run mode: {mode}",
                mode_desc,
                "",
                "Nipype logging directories detected:",
                f"  - Preprocessing/full: {nipype_dir_preproc if (nipype_dir_preproc and os.path.isdir(nipype_dir_preproc)) else 'not found'}",
                f"  - Tract-only: {nipype_dir_tract if (nipype_dir_tract and os.path.isdir(nipype_dir_tract)) else 'not found'}",
            ]

            ax = fig.add_subplot(1, 1, 1)
            ax.axis("off")
            ax.text(0.02, 0.98, "\n".join(text_lines), va="top", ha="left", fontsize=11, wrap=True)
            pdf.savefig(fig); plt.close(fig)

            # -------------------------
            # Page 2: data properties, eddy metrics, bvals/bvecs
            # -------------------------
            fig = plt.figure(figsize=(8.27, 11.69))
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
            fig.suptitle("Page 2: Data properties and eddy QC metrics", fontsize=14, y=0.96)

            # Text block
            ax_info = fig.add_subplot(gs[0, :]); ax_info.axis("off")
            info_lines = ["Data properties:"]
            if data_shape is not None:
                if len(data_shape) == 4:
                    info_lines += [
                        f"  Dimensions: {data_shape[0]}, {data_shape[1]}, {data_shape[2]}",
                        f"  Number of volumes: {data_shape[3]}",
                    ]
                else:
                    info_lines += [f"  Dimensions: {data_shape}"]
                if voxel_size is not None:
                    info_lines += [f"  Voxel size: {voxel_size[0]:.3f}, {voxel_size[1]:.3f}, {voxel_size[2]:.3f} (mm)"]
            else:
                info_lines += ["  Data image not found."]

            info_lines += ["", "b-value summary:"]
            if bvals_arr is not None:
                info_lines += [
                    f"  Unique non-zero b-values: {unique_shells}" if unique_shells else "  No non-zero b-values detected.",
                    f"  # b0 volumes: {n_b0 if n_b0 is not None else 'N/A'}, # DW volumes: {n_dw if n_dw is not None else 'N/A'}",
                ]
            else:
                info_lines += ["  bvals file not found or could not be parsed."]

            info_lines += [
                "",
                "Brain centres (world coordinates):",
                f"  Native-space brain COG: {self._fmt_coord(native_cog_world)}",
                f"  Standard-space brain COG: {self._fmt_coord(std_cog_world)}",
                "",
                "Eddy QC CNR summary (from qc.json):",
            ]
            if isinstance(qc_cnr_avg, (list, tuple)):
                for idx, cnr_val in enumerate(qc_cnr_avg):
                    bval = b_shells[idx] if isinstance(b_shells, (list, tuple)) and len(b_shells) > idx else None
                    std_val = qc_cnr_std[idx] if isinstance(qc_cnr_std, (list, tuple)) and len(qc_cnr_std) > idx else None
                    if bval == 0:
                        label = "b=0 (tSNR)"
                    elif bval is not None:
                        label = f"b={bval}"
                    else:
                        label = f"shell {idx}"
                    info_lines.append(f"  {label}: {cnr_val:.3f}" + (f" (std = {std_val:.3f})" if std_val is not None else ""))
            else:
                info_lines.append("  CNR values not found in qc.json.")

            ax_info.text(0.02, 0.98, "\n".join(info_lines), va="top", ha="left", fontsize=10, wrap=True)

            # b-values per volume
            ax_bvals = fig.add_subplot(gs[1, :])
            if bvals_arr is not None:
                ax_bvals.plot(np.arange(bvals_arr.size), bvals_arr, ".-")
            ax_bvals.set_xlabel("Volume index"); ax_bvals.set_ylabel("b-value"); ax_bvals.set_title("b-values per volume")

            # b-vectors per shell
            shells = (np.unique(bvals_arr.astype(int)).tolist() if bvals_arr is not None else [0])
            subgs = gs[2, :].subgridspec(1, len(shells))
            for i, b in enumerate(shells):
                ax3d = fig.add_subplot(subgs[0, i], projection="3d")
                if bvals_arr is not None and bvecs_arr is not None:
                    idx = (bvals_arr.astype(int) == b)
                    shell_bvecs = bvecs_arr[idx]
                    if shell_bvecs.size > 0:
                        norms = np.linalg.norm(shell_bvecs, axis=1)
                        nz = norms > 0
                        if np.any(nz):
                            uv = shell_bvecs[nz] / norms[nz, None]
                            ax3d.scatter(uv[:, 0], uv[:, 1], uv[:, 2], s=10)
                # unit sphere wireframe
                u = np.linspace(0, 2*np.pi, 40); v = np.linspace(0, np.pi, 40)
                x = np.outer(np.cos(u), np.sin(v)); y = np.outer(np.sin(u), np.sin(v)); z = np.outer(np.ones_like(u), np.cos(v))
                ax3d.plot_wireframe(x, y, z, linewidth=0.3, alpha=0.3)
                ax3d.set_xlim([-1.1, 1.1]); ax3d.set_ylim([-1.1, 1.1]); ax3d.set_zlim([-1.1, 1.1])
                ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
                ax3d.set_title("b=0" if b == 0 else f"b={b}", fontsize=9)
                ax3d.grid(False)

            pdf.savefig(fig); plt.close(fig)

            # -------------------------
            # Page 3: eddy QC brain images
            # -------------------------
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle("Page 3: Eddy QC CNR brain visualisations", fontsize=14, y=0.96)

            if avg_pngs and cnr_pngs and isinstance(qc_cnr_avg, (list, tuple)):
                n_shells = min(len(avg_pngs), len(cnr_pngs), len(qc_cnr_avg))
                b_local = list(b_shells)[:n_shells] if isinstance(b_shells, (list, tuple)) else [None]*n_shells
                g = fig.add_gridspec(n_shells, 2)
                for i in range(n_shells):
                    avg_img = plt.imread(avg_pngs[i]); cnr_img = plt.imread(cnr_pngs[i])
                    cnr_val = qc_cnr_avg[i]
                    bval = b_local[i]
                    std_val = qc_cnr_std[i] if isinstance(qc_cnr_std, (list, tuple)) and len(qc_cnr_std) > i else None

                    shell_label = "b=0 (tSNR)" if bval == 0 else (f"b={bval}" if bval is not None else f"Shell {i}")
                    metric_label = ("tSNR" if bval == 0 else "CNR") + f": {cnr_val:.3f}" + (f" (std = {std_val:.3f})" if std_val is not None else "")

                    axL = fig.add_subplot(g[i, 0]); axL.imshow(avg_img); axL.set_axis_off(); axL.set_title(f"{shell_label} average\n{metric_label}", fontsize=10)
                    axR = fig.add_subplot(g[i, 1]); axR.imshow(cnr_img); axR.set_axis_off(); axR.set_title("tSNR map" if bval == 0 else "CNR map", fontsize=10)
            else:
                ax = fig.add_subplot(1, 1, 1); ax.set_axis_off()
                ax.text(0.5, 0.5, "Eddy QC not available.\nExpected avg_b*.png and cnr*.png alongside qc.json.", ha="center", va="center")
            pdf.savefig(fig); plt.close(fig)

            # -------------------------
            # Page 4: native mask, DEC-FA, FA in STD + outline
            # -------------------------
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle("Page 4: Brain masks and FA visualisations", fontsize=14, y=0.96)
            g = fig.add_gridspec(1, 3)

            # native FA + mask
            ax1 = fig.add_subplot(g[0, 0]); ax1.set_title("Native FA + brain mask", fontsize=10)
            if os.path.exists(brain_mask_png): ax1.imshow(plt.imread(brain_mask_png)); ax1.axis("off")
            else: ax1.axis("off"); ax1.text(0.5, 0.5, "PNG not available.", ha="center", va="center")

            # DEC-FA
            ax2 = fig.add_subplot(g[0, 1]); ax2.set_title("Direction-encoded FA (native)", fontsize=10)
            if os.path.exists(dec_fa_png): ax2.imshow(plt.imread(dec_fa_png)); ax2.axis("off")
            else: ax2.axis("off"); ax2.text(0.5, 0.5, "PNG not available.", ha="center", va="center")

            # FA in STD + outline
            ax3 = fig.add_subplot(g[0, 2]); ax3.set_title("FA in standard space + outline", fontsize=10)
            if os.path.exists(std_outline_png): ax3.imshow(plt.imread(std_outline_png)); ax3.axis("off")
            else: ax3.axis("off"); ax3.text(0.5, 0.5, "PNG not available.", ha="center", va="center")

            pdf.savefig(fig); plt.close(fig)

            # -------------------------
            # Page 5: XTRACT summary PNGs
            # -------------------------
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle("Page 5: XTRACT summary", fontsize=14, y=0.96)
            if xtract_pngs:
                gg = fig.add_gridspec(1, len(xtract_pngs))
                for i, p in enumerate(xtract_pngs):
                    ax = fig.add_subplot(gg[0, i]); ax.imshow(plt.imread(p)); ax.axis("off"); ax.set_title(os.path.basename(p), fontsize=8)
            else:
                ax = fig.add_subplot(1, 1, 1); ax.axis("off")
                ax.text(0.5, 0.5, "No XTRACT summary PNGs found.", ha="center", va="center")
            pdf.savefig(fig); plt.close(fig)

            # -------------------------
            # Page 6: Nipype workflow graphs
            # -------------------------
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle("Page 6: Nipype workflow graphs", fontsize=14, y=0.96)
            gcount = int(bool(nipype_graph_preproc)) + int(bool(nipype_graph_tract))
            if gcount == 0:
                ax = fig.add_subplot(1, 1, 1); ax.axis("off")
                ax.text(0.5, 0.5, "No workflow graphs found.", ha="center", va="center")
            elif gcount == 1:
                ax = fig.add_subplot(1, 1, 1); ax.axis("off")
                gp = nipype_graph_preproc or nipype_graph_tract
                img = plt.imread(gp); ax.imshow(img); ax.set_axis_off()
            else:
                gs = fig.add_gridspec(2, 1)
                img1 = plt.imread(nipype_graph_preproc); ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(img1); ax1.axis("off"); ax1.set_title("Preprocessing/full graph", fontsize=10)
                img2 = plt.imread(nipype_graph_tract);   ax2 = fig.add_subplot(gs[1, 0]); ax2.imshow(img2); ax2.axis("off"); ax2.set_title("Tract-only graph", fontsize=10)
            pdf.savefig(fig); plt.close(fig)

            # (Optional) configuration note
            if isdefined(self.inputs.pipeline_yaml) and self.inputs.pipeline_yaml:
                fig = plt.figure(figsize=(8.27, 3.0))
                fig.suptitle("Pipeline configuration", fontsize=12, y=0.9)
                ax = fig.add_subplot(1, 1, 1); ax.axis("off")
                ax.text(0.02, 0.8, "YAML configuration copied from:\n" + os.path.abspath(self.inputs.pipeline_yaml),
                        va="top", ha="left", fontsize=9, wrap=True)
                pdf.savefig(fig); plt.close(fig)

        self._report_pdf = pdf_path
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs.update(getattr(self, "_results", {}))
        return outputs


class _ModernReport:
    """HTML/dual-PDF report for pipeline-owned ex-vivo outputs."""

    REPORT_VERSION = "2026-10-27.2"

    THEMES = {
        "light": ("#ffffff", "#172033", "#3274a1"),
        "dark": ("#111827", "#f3f4f6", "#60a5fa"),
    }

    def __init__(self, inputs):
        self.inputs = inputs
        self.root = Path(str(inputs.dmri_root or ".")).expanduser().resolve()
        self.out = Path(str(inputs.outdir or self.root / "QA")).expanduser().resolve()
        self.figdir = self.out / "figures"
        self.sections, self.issues, self.sources = [], [], {}
        self.config = {}
        # Match Config defaults when optional YAML keys are absent.
        self.acqtype, self.mode = "ssfp", "nhp"

    def value(self, name):
        value = getattr(self.inputs, name)
        return str(value) if isdefined(value) and value else ""

    def source(self, key, *paths):
        for item in paths:
            path = Path(item).expanduser() if item else None
            if path and path.is_file():
                self.sources[key] = str(path.resolve())
                return path
        return None

    def issue(self, level, title, detail):
        self.issues.append({"severity": level, "title": title, "detail": str(detail)})

    def section(self, title, intro):
        result = {"title": title, "intro": intro, "figures": [], "notes": []}
        self.sections.append(result)
        return result

    @staticmethod
    def volume(path, index=0):
        image = nib.load(str(path))
        data = np.asarray(image.dataobj[..., index] if len(image.shape) == 4 else image.dataobj,
                          dtype=np.float32)
        return np.asarray(nib.as_closest_canonical(nib.Nifti1Image(data, image.affine)).dataobj)

    @staticmethod
    def views(data):
        xyz = tuple(size // 2 for size in data.shape[:3])
        return (np.rot90(data[xyz[0]]), np.rot90(data[:, xyz[1]]), np.rot90(data[:, :, xyz[2]]))

    def add_panel(self, section, stem, title, caption, rows, masks=None):
        names = {}
        for theme, (bg, fg, _) in self.THEMES.items():
            output = self.figdir / f"{stem}_{theme}.png"
            try:
                arrays = [(label, self.volume(path, index)) for label, path, index in rows]
                samples = np.concatenate([a[::5, ::5, ::5].ravel() for _, a in arrays])
                samples = samples[np.isfinite(samples) & (samples > 0)]
                vmax = float(np.percentile(samples, 99.5)) if samples.size else 1
                with plt.rc_context({"figure.facecolor": bg, "axes.facecolor": bg,
                                     "text.color": fg, "axes.labelcolor": fg}):
                    fig, axes = plt.subplots(len(arrays), 3, figsize=(15, 3.4 * len(arrays)),
                                             squeeze=False, layout="constrained")
                    fig.suptitle(title, fontsize=18, weight="semibold")
                    for i, (label, data) in enumerate(arrays):
                        overlay = self.volume(masks[i][0], masks[i][1]) if masks and masks[i] else None
                        for j, view in enumerate(self.views(data)):
                            axes[i, j].imshow(view, cmap="gray", vmin=0, vmax=vmax)
                            if overlay is not None:
                                axes[i, j].contour(self.views(overlay)[j] > .5, levels=[.5],
                                                   colors="#fb7185", linewidths=1)
                            axes[i, j].set_title(("Sagittal", "Coronal", "Axial")[j] if i == 0 else "")
                            axes[i, j].set_ylabel(label, color=fg, weight="semibold")
                            axes[i, j].axis("off")
                    fig.savefig(output, dpi=165, facecolor=bg); plt.close(fig)
                names[theme] = output.name
            except Exception as exc:
                self.issue("warning", f"Could not render {title} ({theme})", exc)
        if names:
            section["figures"].append({"title": title, "caption": caption,
                "light": names.get("light") or names.get("dark"),
                "dark": names.get("dark") or names.get("light")})

    def _read_config(self):
        config = self.source("config", self.value("pipeline_yaml"), self.root / "config_used.yaml")
        if not config:
            self.issue("warning", "Saved configuration unavailable", "SSFP defaults were assumed.")
            return
        try:
            self.config = {str(k).lower(): v for k, v in (yaml.safe_load(config.read_text()) or {}).items()}
            self.acqtype = str(self.config.get("acqtype", "ssfp")).lower()
            self.mode = str(self.config.get("mode", "nhp")).lower()
        except Exception as exc:
            self.issue("error", "Cannot read saved configuration", exc)

    def _config_data(self):
        base = Path(self.sources["config"]).parent if "config" in self.sources else Path.cwd()
        value = (self.config.get("data_paths") or {}).get("data_nii")
        if value:
            path = Path(value).expanduser()
            return path if path.is_absolute() else base / path

        # Config expands input_dir into conventional inputs when data_paths is
        # omitted. Mirror that behaviour for saved minimal configurations.
        input_dir = self.config.get("input_dir")
        if input_dir:
            path = Path(input_dir).expanduser()
            path = path if path.is_absolute() else base / path
            candidate = path / "data.nii.gz"
            if candidate.is_file():
                return candidate

        # Final fallback is the raw copy produced by the SSFP onboarding node.
        candidate = self.root / "data_raw" / "data.nii.gz"
        return candidate if candidate.is_file() else None

    def _stages(self):
        raw = self.source("raw_data", self._config_data())
        current = raw
        pairs = []
        if bool(self.config.get("run_denoise", False)):
            den = self.source("denoised_data", self.root / "denoise" /
                              (str(self.config.get("denoise_name", "data")) + ".nii.gz"))
            pairs.append(("Denoising", current, den, "Raw and denoised data with shared scaling.")); current = den or current
        if current:
            stem = current.name[:-7] if current.name.endswith(".nii.gz") else current.stem
            gibbs = self.source("gibbs_data", self.root / "gibbs" / f"{stem}_gibbs3d.nii.gz",
                                self.root / "gibbs" / f"{stem}_gibbs2d.nii.gz")
        else: gibbs = None
        pairs.append(("Gibbs unringing", current, gibbs, "Check high-contrast boundaries for reduced ringing without blurring."))
        for number, (title, before, after, caption) in enumerate(pairs):
            section = self.section(title, caption)
            if before and after:
                self.add_panel(section, f"stage_{number}", title, caption,
                               [("Before", before, 0), ("After", after, 0)])
            else: section["notes"].append("Required pipeline-owned before/after files were unavailable.")

        # Drift QA is represented by the diagnostic plot produced by the
        # temporal-drift node. Spatial-drift before/after images are deliberately
        # omitted: they add report bulk without a clear ex-vivo acceptance test.
        section = self.section(
            "Temporal drift correction",
            "Pipeline-generated temporal signal-drift fit and correction assessment.",
        )
        plot = self.source(
            "temporal_drift_plot", self.root / "drift_correction" / "drift_plot.png"
        )
        if plot:
            dst = self.figdir / "temporal_drift_metrics.png"
            shutil.copy2(plot, dst)
            section["figures"].append({
                "title": "Temporal drift model",
                "caption": "Pipeline-generated drift fit and correction assessment.",
                "light": dst.name, "dark": dst.name,
            })
        else:
            section["notes"].append("The pipeline temporal-drift plot was unavailable.")

    def _eddy(self):
        section = self.section("EDDY signal quality", "QUAD signal/CNR metrics only. Motion and EDDY-corrected images are intentionally excluded for fixed tissue.")
        qcpath = self.source("eddy_qc", self.value("eddy_qc_json"), self.root / "eddy" / "eddy.qc" / "qc.json")
        if not qcpath: section["notes"].append("EDDY QUAD output was unavailable."); return
        try: qc = json.loads(qcpath.read_text())
        except Exception as exc: self.issue("error", "Cannot parse EDDY QC", exc); return
        values = qc.get("qc_cnr_avg") or []; shells = qc.get("data_unique_bvals") or qc.get("b_shells") or []
        if not values: section["notes"].append("No QUAD SNR/CNR values were present."); return
        for theme, (bg, fg, _) in self.THEMES.items():
            out = self.figdir / f"eddy_signal_{theme}.png"
            with plt.rc_context({"figure.facecolor": bg, "axes.facecolor": bg, "text.color": fg}):
                fig, axes = plt.subplots((len(values)+1)//2, 2, figsize=(14, max(5, 3.5*((len(values)+1)//2))), squeeze=False, layout="constrained")
                for i, ax in enumerate(axes.flat):
                    ax.axis("off")
                    if i >= len(values): continue
                    shell = shells[i] if i < len(shells) else None
                    label = "b0 SNR" if i == 0 or shell == 0 else f"b={shell} CNR"
                    ax.set_title(f"{label}: {values[i]:.2f}", color=fg)
                    png = qcpath.parent / f"cnr{i:04d}.nii.gz.png"
                    if png.is_file(): ax.imshow(plt.imread(png))
                fig.savefig(out, dpi=165, facecolor=bg); plt.close(fig)
        section["figures"].append({"title": "b0 SNR and shell CNR", "caption": "EDDY is diagnostic-only in this pipeline.", "light": "eddy_signal_light.png", "dark": "eddy_signal_dark.png"})

    def _final_maps(self):
        data = self.source("final_data", self.value("data_4d"), self.root / "data" / "data.nii.gz")
        mask = self.source("native_mask", self.value("mask_file"), self.root / "data" / "nodif_brain_mask.nii.gz")
        section = self.section("Final data and brain mask", "The actual downstream data, not EDDY's diagnostic output.")
        if data and mask: self.add_panel(section, "mask", "Final data with mask", "Red outline is the modelling mask.", [("Final data", data, 0)], [(mask, 0)])
        else: section["notes"].append("Final data or mask was unavailable.")

        fa = self.source("fa_native", self.value("fa_img"))
        tensor = self.section("Diffusion tensor model", f"{self.acqtype.upper()} tensor outputs; SSFP CUDIMOT is the default branch.")
        if fa:
            md = self.source("md_native", fa.with_name(fa.name.replace("_FA", "_MD")))
            v1 = self.source("v1_native", fa.with_name(fa.name.replace("_FA", "_V1")))
            rows = [("FA", fa, 0)] + ([("MD", md, 0)] if md else [])
            self.add_panel(tensor, "tensor", "FA and MD", "Pipeline-selected lower-shell tensor maps.", rows)
            if not v1: tensor["notes"].append("V1 was not found beside the pipeline FA output.")
        else: tensor["notes"].append("Pipeline FA output was unavailable.")

        reg = self.section("Standard-space registration", "Subject FA with the configured template FA > 0.2 outlined in red.")
        subject = self.source("fa_standard", self.root / "stdreg" / "FA_in_STD.nii.gz")
        reference = self.source("fa_template", self.value("std_fa_img"))
        if subject and reference:
            try:
                output = self.figdir / "registration_fsleyes.png"
                self._registration_fsleyes(subject, reference, output)
                reg["figures"].append({
                    "title": "FA in standard space with template WM outline",
                    "caption": "Red outline is the configured standard-space FA template thresholded at FA > 0.2. Inspect cortical, callosal, brainstem and cerebellar agreement.",
                    "light": output.name, "dark": output.name,
                })
            except Exception as exc:
                self.issue("warning", "Could not render FSLeyes registration overlay", exc)
                reg["notes"].append("FSLeyes registration rendering failed; see the issue summary.")
        else:
            reg["notes"].append("Subject or template standard-space FA was unavailable.")

    @staticmethod
    def _registration_fsleyes(subject, reference, output):
        executable = shutil.which("fsleyes")
        if not executable:
            raise FileNotFoundError("fsleyes was not found on PATH")
        ref_img = nib.load(str(reference))
        ref_data = np.asarray(ref_img.dataobj)
        coords = np.argwhere(np.isfinite(ref_data) & (ref_data > 0.2))
        centre = coords.mean(axis=0) if coords.size else np.asarray(ref_img.shape[:3]) / 2
        world = nib.affines.apply_affine(ref_img.affine, centre)
        command = [
            executable, "render", "--outfile", str(output), "--size", "2400", "900",
            "--scene", "ortho", "--worldLoc", *(f"{value:.8f}" for value in world),
            "--displaySpace", str(reference),
            "--xzoom", "100", "--yzoom", "100", "--zzoom", "100",
            "--hideLabels", "--showLocation", "no", "--layout", "horizontal",
            "--hideCursor", "--bgColour", "0", "0", "0", "--fgColour", "1", "1", "1",
            str(subject), "--name", "FA_in_STD", "--overlayType", "volume",
            "--alpha", "100", "--cmap", "greyscale", "--displayRange", "0", "1",
            "--interpolation", "none", "--volume", "0",
            str(reference), "--name", "Template_FA", "--overlayType", "mask",
            "--alpha", "100", "--maskColour", "1", "0", "0", "--maskInvert",
            "--threshold", "0", "0.2", "--outline", "--outlineWidth", "3",
            "--interpolation", "none", "--volume", "0",
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)
        if not output.is_file() or not output.stat().st_size:
            raise RuntimeError("FSLeyes completed without creating the registration PNG")

    def _xtract(self):
        section = self.section("XTRACT overview", "Only the pipeline xtract directory is included; xtract.*, TDI and comparison directories are ignored.")
        xdir = Path(self.value("xtract_dir") or self.root / "xtract")
        for name in ("xtract_sagittal.png", "xtract_coronal.png", "xtract_axial.png"):
            src = xdir / name
            if src.is_file():
                dst = self.figdir / name; shutil.copy2(src, dst)
                section["figures"].append({"title": src.stem.replace("xtract_", "").title(), "caption": "Pipeline viewer export.", "light": dst.name, "dark": dst.name})
        if not section["figures"]: section["notes"].append("XTRACT was not requested or viewer outputs were unavailable.")

    def build(self):
        # Avoid leaving obsolete panels (for example the removed spatial-drift
        # images) when a report is regenerated in the same output directory.
        if self.figdir.is_dir():
            shutil.rmtree(self.figdir)
        self.figdir.mkdir(parents=True, exist_ok=True); self._read_config(); self._stages(); self._eddy(); self._final_maps(); self._xtract()
        return self._write()

    def _write(self):
        stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        html_path, light, dark = self.out / "CMC_exvivo_QA.html", self.out / "CMC_exvivo_QA.pdf", self.out / "CMC_exvivo_QA_dark.pdf"
        manifest = self.out / "QA_manifest.json"
        selected_config = {
            key: self.config.get(key) for key in (
                "input_dir", "brain_mask", "run_denoise", "denoise_method",
                "run_gibbs", "run_eddy", "run_eddy_qc", "run_spatial_drift",
                "dti_approx_b", "dti_db", "dti_b0", "reg_method",
                "xtract_species", "xtract_profiles_dir",
            ) if key in self.config
        }
        meta = {"report_version": self.REPORT_VERSION, "generated": stamp,
                "root": str(self.root), "acqtype": self.acqtype,
                "mode": self.mode, "run_mode": self.inputs.run_mode,
                "selected_config": selected_config}
        manifest.write_text(json.dumps({"metadata": meta, "issues": self.issues, "sources": self.sources, "sections": self.sections}, indent=2) + "\n")
        html_path.write_text(self._html(meta), encoding="utf-8")
        for theme, path in (("light", light), ("dark", dark)): self._pdf(path, theme, meta)
        return {"report_pdf": light, "report_pdf_dark": dark, "report_html": html_path, "manifest_json": manifest, "figures_dir": self.figdir}

    def _html(self, meta):
        esc = lambda x: html.escape(str(x), quote=True)
        alerts = self.issues or [{"severity":"ok","title":"No automated issues detected","detail":"Visual review remains required."}]
        cards = "".join(f'<div class="alert {esc(x["severity"])}"><b>{esc(x["title"])}</b><p>{esc(x["detail"])}</p></div>' for x in alerts)
        body = []
        for sec in self.sections:
            figs = "".join(f'<article><h3>{esc(f["title"])}</h3><img class="light" src="figures/{esc(f["light"])}"><img class="dark" src="figures/{esc(f["dark"])}"><p>{esc(f["caption"])}</p></article>' for f in sec["figures"])
            notes = "".join(f'<p class="note">{esc(n)}</p>' for n in sec["notes"])
            body.append(f'<section><h2>{esc(sec["title"])}</h2><p>{esc(sec["intro"])}</p>{notes}<div class="grid">{figs}</div></section>')
        return f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>CMC ex-vivo QA</title><style>body{{--bg:#f5f7fb;--paper:#fff;--fg:#172033;--sub:#586579;--line:#dfe5ee;margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui}}body[data-theme=dark]{{--bg:#0b1320;--paper:#111827;--fg:#f3f4f6;--sub:#aeb9c9;--line:#324053}}header{{padding:16px 3%;background:var(--paper);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:2}}button{{float:right}}main{{max-width:1450px;margin:auto;padding:35px 3% 70px}}h1{{font-size:46px}}section{{padding-top:36px}}.grid,.alerts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:14px}}article,.alert,.note{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:16px}}article img{{width:100%;background:#111827}}p{{color:var(--sub)}}.dark{{display:none}}body[data-theme=dark] .dark{{display:block}}body[data-theme=dark] .light{{display:none}}</style></head><body data-theme="light"><header><b>CMC · EX-VIVO DWI QA</b><button onclick="document.body.dataset.theme=document.body.dataset.theme==='dark'?'light':'dark'">Light / dark</button></header><main><h1>{esc(self.root.parent.name)}</h1><p>{esc(self.acqtype.upper())} · {esc(self.mode.upper())} · {esc(self.inputs.run_mode)}</p><h2>Action needed and caveats</h2><div class="alerts">{cards}</div>{''.join(body)}<p><a href="QA_manifest.json">QA manifest</a></p></main></body></html>'''

    def _pdf(self, path, theme, meta):
        bg, fg, accent = self.THEMES[theme]
        with PdfPages(path) as pdf:
            fig = plt.figure(figsize=(13.33, 7.5), facecolor=bg); fig.text(.055,.86,"CMC / EX-VIVO DWI",color=accent,weight="bold"); fig.text(.055,.73,"Quality assurance",color=fg,fontsize=32,weight="bold"); fig.text(.055,.62,f"{self.acqtype.upper()} · {self.mode.upper()} · {self.inputs.run_mode}",color=fg,fontsize=18); fig.text(.055,.52,str(self.root),color=fg,fontsize=9); pdf.savefig(fig,facecolor=bg); plt.close(fig)
            for sec in self.sections:
                for item in sec["figures"]:
                    image = self.figdir / item[theme]
                    if not image.is_file(): continue
                    fig=plt.figure(figsize=(13.33,7.5),facecolor=bg); fig.text(.05,.92,sec["title"].upper(),color=accent,weight="bold"); fig.text(.05,.86,item["title"],color=fg,fontsize=19); ax=fig.add_axes((.05,.17,.9,.64)); ax.imshow(plt.imread(image)); ax.axis("off"); fig.text(.05,.11,textwrap.fill(item["caption"],150),color=fg,fontsize=9); pdf.savefig(fig,facecolor=bg); plt.close(fig)
