# src/exvivo_dmri_pipeline/qa_report.py
import os
import json
import datetime
import subprocess

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
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
        outputs["report_pdf"] = getattr(self, "_report_pdf", None)
        return outputs
