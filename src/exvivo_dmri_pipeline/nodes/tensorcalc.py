#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import nibabel as nib
import numpy as np

# Try fslpy first; if not present, fall back to a local implementation
try:
    from fsl.data.dtifit import eigendecompositionToComponents
    _USE_FSLPY = True
except Exception:
    _USE_FSLPY = False

    def eigendecompositionToTensor(V1, V2, V3, L1, L2, L3):
        # Shape checks similar to fslpy
        check_shape = L1.shape
        for ev in (L2, L3):
            if ev.shape != check_shape:
                raise ValueError("Eigenvalues have different shapes")
        for vec in (V1, V2, V3):
            if vec.shape != check_shape + (3,):
                raise ValueError("Eigenvectors do not match eigenvalue shape")
        # (...,3,3)
        return (
            L1[..., None, None] * V1[..., None, :] * V1[..., :, None] +
            L2[..., None, None] * V2[..., None, :] * V2[..., :, None] +
            L3[..., None, None] * V3[..., None, :] * V3[..., :, None]
        )

    def eigendecompositionToComponents(V1, V2, V3, L1, L2, L3):
        T = eigendecompositionToTensor(V1, V2, V3, L1, L2, L3)
        # Dxx,Dxy,Dxz,Dyy,Dyz,Dzz
        comps = np.stack([
            T[..., 0, 0],
            T[..., 0, 1],
            T[..., 0, 2],
            T[..., 1, 1],
            T[..., 1, 2],
            T[..., 2, 2],
        ], axis=-1)
        return comps

def _squeeze_last_if_singleton(a):
    return a[..., 0] if (a.ndim >= 1 and a.shape[-1] == 1) else a

def main():
    p = argparse.ArgumentParser(description="Build dti_tensor (6-vol) from V1,V2,V3,L1,L2,L3")
    p.add_argument("--v1", required=True)
    p.add_argument("--v2", required=True)
    p.add_argument("--v3", required=True)
    p.add_argument("--l1", required=True)
    p.add_argument("--l2", required=True)
    p.add_argument("--l3", required=True)
    p.add_argument("--out", required=True, help="Output NIfTI path (e.g., dti_tensor.nii.gz)")
    args = p.parse_args()

    # Load
    V1_img = nib.load(args.v1); V1 = V1_img.get_fdata(dtype=np.float32)
    V2_img = nib.load(args.v2); V2 = V2_img.get_fdata(dtype=np.float32)
    V3_img = nib.load(args.v3); V3 = V3_img.get_fdata(dtype=np.float32)
    L1_img = nib.load(args.l1); L1 = L1_img.get_fdata(dtype=np.float32)
    L2_img = nib.load(args.l2); L2 = L2_img.get_fdata(dtype=np.float32)
    L3_img = nib.load(args.l3); L3 = L3_img.get_fdata(dtype=np.float32)

    # Handle eigenvalues that may have trailing singleton
    L1 = _squeeze_last_if_singleton(L1)
    L2 = _squeeze_last_if_singleton(L2)
    L3 = _squeeze_last_if_singleton(L3)

    # Sanity checks (match fslpy expectations)
    for name, V in (("V1", V1), ("V2", V2), ("V3", V3)):
        if V.ndim < 1 or V.shape[-1] != 3:
            raise RuntimeError(f"{name} has shape {V.shape}, expected (...,3)")
    if not (L1.shape == L2.shape == L3.shape):
        raise RuntimeError(f"Eigenvalue shape mismatch: {L1.shape}, {L2.shape}, {L3.shape}")
    if V1.shape[:-1] != L1.shape:
        raise RuntimeError(f"Vector/value shape mismatch: V1 {V1.shape} vs L1 {L1.shape}")

    comps = eigendecompositionToComponents(V1, V2, V3, L1, L2, L3).astype(np.float32)

    # Ensure 4D (X,Y,Z,6)
    if comps.ndim == 3:
        comps = comps[..., np.newaxis]

    out_img = nib.Nifti1Image(comps, V1_img.affine, V1_img.header)
    out_img.set_data_dtype(np.float32)
    try:
        out_img.header.set_intent('symmetric matrix', (), '')
    except Exception:
        pass

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    nib.save(out_img, args.out)

if __name__ == "__main__":
    sys.exit(main())
