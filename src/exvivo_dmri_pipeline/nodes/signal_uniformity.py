"""Low-frequency spatial dMRI signal-uniformity assessment for QA."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def estimate_signal_field(data, mask, voxel_sizes, fwhm_mm=30.0):
    """Estimate a smooth multiplicative signal field and local signal ratio."""
    valid = mask & np.isfinite(data) & (data > 0)
    if np.count_nonzero(valid) < 1000:
        raise ValueError("Mask contains too few positive finite voxels")
    signal = data[valid].astype(np.float64)
    low, high = np.percentile(signal, (1, 99))
    floor = max(float(low), np.finfo(np.float32).tiny)
    clipped = np.clip(data.astype(np.float64), floor, float(high))
    logged = np.zeros(data.shape, dtype=np.float64)
    logged[valid] = np.log(clipped[valid])
    sigma_mm = float(fwhm_mm) / np.sqrt(8 * np.log(2))
    sigma = tuple(sigma_mm / max(float(size), 1e-6) for size in voxel_sizes)
    weights = ndimage.gaussian_filter(valid.astype(float), sigma, mode="constant")
    smooth = ndimage.gaussian_filter(logged, sigma, mode="constant")
    smooth /= np.maximum(weights, 1e-6)
    centre = float(np.median(smooth[valid]))
    field = np.zeros(data.shape, dtype=np.float32)
    field[mask] = np.exp(smooth[mask] - centre).astype(np.float32)
    ratio = np.zeros(data.shape, dtype=np.float32)
    ratio[valid] = (data[valid] /
                    np.maximum(field[valid] * np.exp(centre), floor)).astype(np.float32)
    low_signal = np.zeros(data.shape, dtype=np.float32)
    low_signal[valid] = np.clip((0.75 - ratio[valid]) / 0.75, 0, 1)
    metrics = {
        "fwhm_mm": float(fwhm_mm),
        "mask_voxels": int(np.count_nonzero(mask)),
        "field_p05_p50_p95": [float(v) for v in np.percentile(field[valid], (5, 50, 95))],
        "observed_expected_p05_p50_p95": [float(v) for v in np.percentile(ratio[valid], (5, 50, 95))],
        "fraction_below_0.75_expected": float(np.mean(ratio[valid] < .75)),
    }
    return field, ratio, low_signal, metrics


def _save_like(reference, data, path):
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data.astype(np.float32), reference.affine, header), str(path))


def process_image(image_path, output_dir, mask_path, fwhm_mm=30.0):
    image_path, output_dir, mask_path = map(Path, (image_path, output_dir, mask_path))
    image = nib.load(str(image_path)); data = np.asarray(image.dataobj, dtype=np.float32)
    mask_image = nib.load(str(mask_path))
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D shell average, got {data.shape}")
    if mask_image.shape != image.shape or not np.allclose(mask_image.affine, image.affine):
        raise ValueError("Mask and shell average do not share the same grid")
    mask = np.asarray(mask_image.dataobj) > 0
    field, ratio, low_signal, metrics = estimate_signal_field(
        data, mask, image.header.get_zooms()[:3], fwhm_mm)
    target = output_dir / image_path.name.removesuffix(".nii.gz").removesuffix(".nii")
    target.mkdir(parents=True, exist_ok=True)
    _save_like(image, field, target / "smooth_signal_field.nii.gz")
    _save_like(image, ratio, target / "observed_expected_ratio.nii.gz")
    _save_like(image, low_signal, target / "low_signal_score.nii.gz")
    metrics.update({"input": str(image_path.resolve()), "mask_source": str(mask_path.resolve())})
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
