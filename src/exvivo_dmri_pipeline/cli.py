#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List

from .config import Config
from .pipeline import build_workflow


# -------------------------------
# Small helpers
# -------------------------------
def _read_nonempty_lines(path: str) -> List[str]:
    out = []
    try:
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    out.append(s)
    except Exception:
        return []
    return out

def _require_files(paths: List[str], what: str) -> None:
    import os
    missing: List[str] = []
    broken: List[str] = []

    for p in paths:
        if not p:
            missing.append(str(p))
            continue

        if os.path.islink(p):
            if not os.path.exists(p):
                broken.append(p)
        else:
            if not os.path.isfile(p):
                missing.append(p)

    if missing or broken:
        msgs = [f"[cmc_exvivo_pipeline] {what} not found or invalid."]
        if missing:
            msgs.append("Missing:\n  - " + "\n  - ".join(missing))
        if broken:
            msgs.append("Broken symlinks:\n  - " + "\n  - ".join(broken))
        raise SystemExit("\n".join(msgs))


# -------------------------------
# Checks for TRACT-ONLY mode
# (expects preproc outputs on disk)
# -------------------------------
def _check_tractonly_outputs(cfg: Config) -> None:
    dmri_root = cfg.abs_dmri_root
    proc_dir = os.path.join(dmri_root, "proc")
    data_dir = os.path.join(proc_dir, "data")
    reg_dir = os.path.join(proc_dir, "NMTreg")

    data_4d = os.path.join(data_dir, "data.nii.gz")
    mask = os.path.join(data_dir, "nodif_brain_mask.nii.gz")
    _require_files([data_4d, mask], "preprocessed diffusion outputs in <dmri_root>/proc/data")

    # SE tract-only additionally needs bvals/bvecs present in <proc>/data
    if cfg.acqtype.lower().strip() == "se":
        bvals = os.path.join(data_dir, "bvals")
        bvecs = os.path.join(data_dir, "bvecs")
        _require_files([bvals, bvecs], "preprocessed diffusion gradients in <dmri_root>/proc/data")

    # Warps required for XTRACT
    if cfg.REG_METHOD.upper().strip() == "FNIRT":
        warp_s2a = os.path.join(reg_dir, "warp_std2anat.nii.gz")
        warp_a2s = os.path.join(reg_dir, "warp_anat2std.nii.gz")
        _require_files([warp_s2a, warp_a2s], "FNIRT warps in <dmri_root>/proc/NMTreg")
    else:
        warp = os.path.join(reg_dir, "combined_warp.nii.gz")
        warp_inv = os.path.join(reg_dir, "combined_warp_inv.nii.gz")
        _require_files([warp, warp_inv], "MMORF warps in <dmri_root>/proc/NMTreg")

    # SSFP needs CFP/FixP already present (they’re created during preproc)
    if cfg.acqtype.lower().strip() == "ssfp":
        cfp = os.path.join(data_dir, "CFP")
        fixp = os.path.join(data_dir, "FixP")
        _require_files([cfp, fixp], "SSFP manifests (CFP/FixP) in <dmri_root>/proc/data")

        bad_refs = []
        for manifest in (cfp, fixp):
            refs = _read_nonempty_lines(manifest)
            if not refs:
                bad_refs.append(f"{manifest} (empty or unreadable)")
                continue
            for r in refs:
                if not os.path.isfile(r):
                    bad_refs.append(f"{manifest} -> missing: {r}")
        if bad_refs:
            raise SystemExit(
                "[cmc_exvivo_pipeline] Invalid paths in CFP/FixP manifests:\n  - " +
                "\n  - ".join(bad_refs)
            )

# -------------------------------
# Checks for FULL mode
# -------------------------------
def _check_full_inputs(cfg: Config) -> None:
    acq = cfg.acqtype.lower().strip()

    # Core diffusion inputs
    if acq == "se":
        need = [
            cfg.data_paths.get("data_nii"),
            cfg.data_paths.get("bvals"),
            cfg.data_paths.get("bvecs"),
        ]
        _require_files(need, "raw diffusion inputs (SE) from YAML")
    else:
        # SSFP: gradients are not mandatory here (they may be unused);
        # we still require the main data and SSFP resources below.
        need = [cfg.data_paths.get("data_nii")]
        _require_files(need, "raw diffusion input (SSFP) from YAML")

    # SSFP: sidecars + maps are required up front
    if acq == "ssfp":
        sidecars = [cfg.sidecars.get(k, "") for k in
                    ("TRs", "diffGradAmps", "diffGradDurs", "flipAngles", "flipMask", "noisefloor")]
        maps = [cfg.maps.get(k, "") for k in ("T1map", "T2map", "B1map")]
        _require_files(sidecars, "SSFP sidecar files (TRs/diffGradAmps/diffGradDurs/flipAngles/flipMask/noisefloor)")
        _require_files(maps, "SSFP quantitative maps (T1map/T2map/B1map)")

# -------------------------------
# CLI entry
# -------------------------------
def _compute_run_mode(args) -> str:
    if args.full:
        return "full"
    if args.preproc and args.tractography:
        return "full"
    if args.tractography:
        return "tract_only"
    if args.preproc:
        return "preproc_only"
    return "preproc_only"


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="cmc_exvivo_pipeline",
        description="CMC ex-vivo dMRI pipeline"
    )
    p.add_argument("config", help="Path to YAML config file")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose / debug logs")
    p.add_argument("--preproc", action="store_true",
                   help="Build/run preprocessing only")
    p.add_argument("--tractography", action="store_true",
                   help="Build/run tractography stage (assumes preproc outputs unless combined with --preproc)")
    p.add_argument("--full", action="store_true",
                   help="Build/run full pipeline (equivalent to --preproc --tractography)")

    args = p.parse_args(argv)
    cfg = Config.from_yaml(args.config)
    run_mode = _compute_run_mode(args)

    if args.verbose:
        print(f"[cli] run_mode='{run_mode}'")

    # Sanity checks
    if run_mode == "tract_only":
        _check_tractonly_outputs(cfg)
    elif run_mode == "full":
        _check_full_inputs(cfg)

    # Build + graph + run
    wf = build_workflow(cfg, verbose=args.verbose, run_mode=run_mode)

    try:
        wf.write_graph(graph2use="colored", format="png", simple_form=True)
    except Exception:
        pass

    wf.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
