# src/exvivo_dmri_pipeline/utils/__init__.py
import os
import re
import subprocess
import shutil
from pathlib import Path
from importlib import resources as ir

__all__ = [
    "FSL", "require_fsl", "fslval", "img_stem", "pkg_file"
]

# FSL root directory (string), default to 'fsldir' if env not set
FSL = os.environ.get("FSLDIR", "fsldir")

def require_fsl():
    """
    Ensure FSL is available. Raises RuntimeError with a helpful message if not.
    """
    # Accept either FSLDIR env or discover a few key binaries on PATH.
    fsl_root = os.environ.get("FSLDIR")
    needed = ["fslval", "fslroi", "fslmerge", "fslmaths", "cluster"]
    missing = []

    def _exists(path):
        return path and os.path.exists(path)

    if fsl_root:
        # check typical locations in $FSLDIR/bin
        for b in needed:
            if not _exists(os.path.join(fsl_root, "bin", b)):
                missing.append(b)
    else:
        # fall back to PATH search
        for b in needed:
            if not shutil.which(b):
                missing.append(b)

    if missing:
        hint = (
            "Set FSLDIR and source FSL's env script, e.g.\n"
            "  export FSLDIR=/usr/local/fsl\n"
            "  . ${FSLDIR}/etc/fslconf/fsl.sh\n"
            "and ensure ${FSLDIR}/bin is on PATH."
        )
        raise RuntimeError(f"FSL not found or incomplete; missing: {', '.join(missing)}.\n{hint}")

def fslval(img_path: str, key: str) -> int | float | str:
    """
    Call `fslval <img> <key>` and return a numeric value if possible, else the raw string.
    """
    exe = os.path.join(FSL, "bin", "fslval") if os.path.exists(os.path.join(FSL, "bin", "fslval")) else "fslval"
    out = subprocess.check_output([exe, img_path, key], text=True).strip()
    # try int, then float, else return string
    try:
        return int(out)
    except ValueError:
        try:
            return float(out)
        except ValueError:
            return out

def img_stem(p: str) -> str:
    """
    Strip .nii or .nii.gz and return the path without extension.
    """
    p = str(p)
    if p.endswith(".nii.gz"):
        return p[:-7]
    if p.endswith(".nii"):
        return p[:-4]
    return p

def pkg_file(subdir: str, filename: str) -> str:
    with ir.as_file(ir.files("exvivo_dmri_pipeline") / subdir / filename) as p:
        return str(p)

def ensure_parent_dir(p: str):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

def copy_if_exists(src: str, dst: str) -> str:
    if src and os.path.exists(src):
        ensure_parent_dir(dst)
        shutil.copy2(src, dst)
        return dst
    return ""