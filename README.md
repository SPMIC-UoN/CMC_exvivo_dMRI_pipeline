# CMC Ex Vivo dMRI Pipeline (Macaque, 10.5 T)

**Processing pipeline for ex vivo macaque ultra-high-field (10.5 T) dMRI data acquired as part of the Center for Mesoscale Connectomics (CMC).**

**Citation:** Warrington et al. (in prep)

---

## 1) Overview

This pipeline ingests ex vivo diffusion MRI in one of two acquisition modes:
 - DW-SE: diffusion-weighted spin echo, single-line readout
 - DW-SSFP: diffusion-weighted steady-state free precession with sidecars

It performs optional denoising, Gibbs unringing, temporal drift and spatial drift correction, brain masking, diffusion tensor modeling, registration to a user-specified standard space, optional crossing-fibre modeling, XTRACT tract segmentation, and generates an integrated QA report.

Command line entrypoint:

```bash
cmc_exvivo_pipeline config.yaml -v
```

The workflow is Nipype-based and modular. Most heavy lifting is in FSL, with optional Docker helpers for Gibbs unringing.
If the pipeline fails, or you need to adjust something, you can re-run from any point through Nipype caching framework.

---

## 2) Supported modes

Species mode:
- mode: nhp Non-human primate ex vivo
- mode: hum Human ex vivo

The species mode controls sensible defaults and registration and XTRACT behaviour.

Acquisition mode:
- acqtype: se DW-SE with bvals and bvecs
- acqtype: ssfp DW-SSFP with required DW-SSFP sidecars and parameter maps (T1, T2, B1 maps)

---

## 3) Requirements

### Core software
- **Linux** with bash, coreutils
- **Python 3.10+**  
- **FSL 6.0.7.x or newer**  
- **MRtrix3 3.0.7+**  
  Used for optional Gibbs step
  **Note on Docker:** you can run MRtrix tools in the `mrtrix3/mrtrix3` container
- **CUDA** for GPU `eddy`, bedpostx and XTRACT - these should be installed with FSL already  
  If no GPU, set `NO_GPU: true` in config to force CPU paths
- **CUDIMOT**
  Used for DW-SSFP tensor and crossing fibre modelling

### External scripts and resources
- **Denoising wrapper script** (from https://github.com/SPMIC-UoN/EDDEN/tree/main/code). Path supplied in `config.yaml` as `denoise_sh`.

---

## 4) Installation

### Docker version: https://hub.docker.com/repository/docker/swarrington1/cmc-exvivo
- docker pull swarrington1/cmc-exvivo:v0.1.0

Below is a minimal, reproducible setup using Conda.

1. **Install core software**
   - Install **FSL** and **MRtrix3**, and set environments.

2. **Create a Python environment**
   ```bash
   mamba create -n cmc_exvivo python=3.10 -y
   mamba activate cmc_exvivo
   ```

3. **Install the pipeline package**
   - From source (recommended during development):
     ```bash
     pip install -e .
     ```
     This exposes the CLI `cmc_exvivo_pipeline`.

---

## 5) Input data requirements

## DW-SE inputs (`acqtype: se`)

Data and sidecars:
```bash
data_paths:
  data_nii: /path/to/diffusion.nii.gz

sidecars:
  bvals:    /path/to/diffusion.bval
  bvecs:    /path/to/diffusion.bvec
```
Optional:
```bash
data_paths:
  data_phase: /path/to/phase.nii.gz   # if using complex denoising

brain_mask: /path/to/precomputed_mask.nii.gz
```

## DW-SSFP inputs (`acqtype: ssfp`)

Data:
```bash
data_paths:
  data_nii: /path/to/diffusion.nii.gz
```
Optional:
```bash
data_paths:
  data_phase: /path/to/phase.nii.gz   # if using complex denoising

brain_mask: /path/to/precomputed_mask.nii.gz
```
Parameter maps:
```bash
maps:
  T1map: /path/to/T1map.nii.gz
  T2map: /path/to/T2map.nii.gz
  B1map: /path/to/B1map.nii.gz
```
Sidecars:
```bash
sidecars:
  TRs:           /path/to/TRs.txt
  diffGradAmps:  /path/to/diffGradAmps.txt
  diffGradDurs:  /path/to/diffGradDurs.txt
  flipAngles:    /path/to/flipAngles.txt
  flipMask:      /path/to/flipMask.nii.gz
  noisefloor:    /path/to/noisefloor.txt
  whichshell:    /path/to/whichshell.txt
```

---

## 6) The `config.yaml`

A minimal example illustrates the key fields. Paths may be relative or absolute; the pipeline writes all products into `dmri_root`.

```yaml
# Required roots
dmri_root: /path/to/processing/proc
input_root: /path/to/INPUT_ROOT

mode: nhp
acqtype: ssfp

data_paths:
  data_nii: /path/to/dwssfp.nii.gz

sidecars:
  TRs:           /path/to/TRs.txt
  diffGradAmps:  /path/to/diffGradAmps.txt
  diffGradDurs:  /path/to/diffGradDurs.txt
  flipAngles:    /path/to/flipAngles.txt
  flipMask:      /path/to/flipMask.nii.gz
  noisefloor:    /path/to/noisefloor.txt
  whichshell:    /path/to/whichshell.txt

maps:
  T1map: /path/to/T1map.nii.gz
  T2map: /path/to/T2map.nii.gz
  B1map: /path/to/B1map.nii.gz

B0RANGE: 60                  # threshold for identifying b0 volumes
ECHO_MS: 0.34                # echo spacing (ms)

run_gibbs: true # run Gibbs unringing?
run_n4: true # run bias field correction?
use_docker: true
docker_image: mrtrix3/mrtrix3
interactive_tty: true

LOWER_B: 3200
brain_mask: null # option to supply a precomputed brain mask in T1 space
t1: null # required if providing T1 space brain mask

# Registration and tractography
stdreg_method: mmorf         # mmorf or fnirt

```

### Notes on key parameters
- **`B0RANGE`** defines what counts as b0 when identifying or removing initial volumes and reporting counts.  
- **`ECHO_MS` and `PIFACTOR`** are used to compute readout time for TOPUP/EDDY acqparams.  
- **`LOWER_B`** defines the lower shell used for the DTI tensor fit.  
- **Docker helpers**: when `use_docker: true`, Gibbs and N4 steps run inside a container you name in `docker_image`.

---

## 7) Outputs (selected)

Under `dmri_root` the pipeline creates:

```
PROC/
  data_raw/
  sidecars/
  data/
    data.nii.gz
    bvals
    bvecs
    nodif_brain_mask.nii.gz
  data.DTI_SSFP/
      dti_FA.nii.gz
      dti_V1.nii.gz
      dti_tensor.nii.gz
  eddy/
    eddy.qc/
      qc.json
      avg_b*.png, cnr*.png
  stdreg/
    FA_in_STD.nii.gz
    warp_anat2std.nii.gz
    warp_std2anat.nii.gz
    # or MMORF:
    combined_warp.nii.gz
    combined_warp_inv.nii.gz
  xtract/
    tracts/                 # containing xtract tract density maps
  QA/
    QA_report.pdf
  nipype_logging*/          # working directories and graph.png

```

---

## 8) Running the pipeline

```bash
cmc_exvivo_pipeline /path/to/config.yaml -v
```
- `--preproc` sets the pipeline to only run pre-processing up to and including diffusion tensor modelling and standard space registrations.
- `--tractography` sets the pipeline to additionally run crossing fibre modelling and landmark-based tractoraphy with XTRACT.
- `-v` enables informative Nipype logging to help with debugging.
- The pipeline copies your YAML into `dmri_root/config_used.yaml` for provenance.

---
