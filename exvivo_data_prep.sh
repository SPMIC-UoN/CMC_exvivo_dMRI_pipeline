#!/usr/bin/env bash

# ------------------------------------------------------------
# Ex vivo data prep (intended as interactive, line-by-line)
# - DW-SSFP prep (bvals/bvecs/header fix + sanity dtifit)
# - Anatomicals + parameter maps (TSE, T1, T2 mapping)
# - T2 multi-echo (dcm2niix → merge by echo → TE.txt → reorient)
# - B1 TurboFLASH map (explicit series, divide by 900, reorient)
# - CFP files for DW-SSFP pipeline
# ------------------------------------------------------------

# ============================================================
# CONFIG - edit this block for each dataset
# ============================================================

# Root directory for this dataset
datadir="/home/data/NIH_CONNECTS/NHP_Exvivo/20250404-Vlad_NHP_exvivo_ssfp"

# DICOM session folder name inside ${datadir}
session_name="DICOM"
session_dir="${datadir}/${session_name}"

# DW-SSFP online recon directory (under the session dir)
dwssfp_ssfp_subdir="ssfp_dwi"                     # typically "ssfp"
dwssfp_online_dir="${session_dir}/${dwssfp_ssfp_subdir}"

# DW-SSFP DICOM search term - used to list matching SSFP series
dwssfp_search_term="ssfp_dwi"

# Protocol details (b-value calculator + DVS)
protocol_dir="${datadir}/protocol_details"
params_qlow="${protocol_dir}/params_qlow.json"
params_qhigh="${protocol_dir}/params_qhigh.json"
dvs_qlow="${protocol_dir}/DiffusionVectors_QLOW.dvs"
dvs_qhigh="${protocol_dir}/DiffusionVectors_QHIGH.dvs"

# Nominal q-values corresponding to q=0, qlow, qhigh
q0_nominal=50
q1_nominal=175
q2_nominal=275

# Offline recon NIFTI directory
# offline_recon_dir="${datadir}/NIFTI_offline_recon"
offline_recon_dir="${datadir}/Vlad_exvivo_040425_OFFLINE"

# ============================================================
# Modules
# ============================================================
module load dcm2niix fsl

# Output NIFTI dirs
n_online="${datadir}/NIFTI_online"
n_proc="${datadir}/NIFTI_for_proc"
mkdir -p "${n_online}" "${n_proc}"

# Simple helper for existence checks
check_dir() {
  local d="$1"
  local label="$2"
  if [ ! -d "${d}" ]; then
    echo "[ERROR] ${label} directory not found: ${d}"
    # exit 1
  fi
}

check_file() {
  local f="$1"
  local label="$2"
  if [ ! -f "${f}" ]; then
    echo "[ERROR] ${label} file not found: ${f}"
    # exit 1
  fi
}

# Check key inputs
check_dir "${session_dir}" "Session"
check_dir "${dwssfp_online_dir}" "DW-SSFP online"
check_dir "${offline_recon_dir}" "Offline recon"
check_dir "${protocol_dir}" "Protocol details"
check_file "${params_qlow}" "params_qlow.json"
check_file "${params_qhigh}" "params_qhigh.json"
check_file "${dvs_qlow}" "DiffusionVectors_QLOW.dvs"
check_file "${dvs_qhigh}" "DiffusionVectors_QHIGH.dvs"

# ============================================================
# STEP 0: DW-SSFP series report (based on search term)
# ============================================================

echo "------------------------------------------------------------"
echo "[DW-SSFP] Reporting SSFP DICOM series matching '${dwssfp_search_term}'"
echo "           under session: ${session_dir}"
echo "------------------------------------------------------------"

mapfile -t dwssfp_hits < <(find "${session_dir}" -maxdepth 2 -type d -iname "*${dwssfp_search_term}*" 2>/dev/null || true)

if [ ${#dwssfp_hits[@]} -eq 0 ]; then
  echo "[DW-SSFP] WARNING: no DICOM directories matched '${dwssfp_search_term}'."
  echo "           Check dwssfp_search_term in the config block."
else
  printf '  %s\n' "${dwssfp_hits[@]}"
fi

# ------------------------------------------------------------
# STEP 1: Basic DW-SSFP preproc
# ------------------------------------------------------------

# Block variables
combined_mag="${n_proc}/combined_mag.nii.gz"
data_nifti="${n_proc}/data.nii.gz"
bvecs="${n_proc}/bvecs"
bvals="${n_proc}/bvals"
bvecs_orig="${n_proc}/bvecs_orig"

echo "------------------------------------------------------------"
echo "[DW-SSFP] Estimating b-values from protocol parameters"
echo "------------------------------------------------------------"

# Parse effective b-value from the calculator output: Effective b-value = ... (####)
bval_qlow=$(fslpython ~/CMC_tools/DW-SSFP/b_value_calculator.py "${params_qlow}"  | awk -F'[()]' '/Effective b-value/{gsub(/ /,"",$2); print $2}')
bval_qhigh=$(fslpython ~/CMC_tools/DW-SSFP/b_value_calculator.py "${params_qhigh}" | awk -F'[()]' '/Effective b-value/{gsub(/ /,"",$2); print $2}')

if [ -z "${bval_qlow}" ] || [ -z "${bval_qhigh}" ]; then
  echo "[ERROR] Failed to parse effective b-values from b_value_calculator.py output."
  # exit 1
fi

echo "[DW-SSFP] Parsed effective b-values:"
echo "  qlow  = ${bval_qlow}"
echo "  qhigh = ${bval_qhigh}"

# Convenience lists for downstream pyrep calls
bval_list_2="${bval_qlow},${bval_qhigh}"
bval_list_3="0,${bval_qlow},${bval_qhigh}"

echo "------------------------------------------------------------"
echo "[DW-SSFP] Combining online recon data with combine_exvivo"
echo "  - Input DICOM dir: ${dwssfp_online_dir}"
echo "  - Output NIFTI dir: ${n_online}"
echo "------------------------------------------------------------"

fslpython ~/CMC_tools/data_prep/combine_exvivo.py -datadir "${dwssfp_online_dir}" -outdir "${n_online}"

# Combine offline recon data
echo "------------------------------------------------------------"
echo "[DW-SSFP] Combining offline recon NIFTI to combined_mag"
echo "  - Offline recon dir: ${offline_recon_dir}"
echo "------------------------------------------------------------"

fslmerge -t "${combined_mag}" "${offline_recon_dir}/"*nii.gz

# Quick check: list volumes in offline recon
echo "[DW-SSFP] Offline recon volume counts:"
for img in "${offline_recon_dir}"/*nii.gz; do
  echo "$(basename "$img")  dim4=$(fslval "$img" dim4)"
done

# Review online merge log
echo "------------------------------------------------------------"
echo "[DW-SSFP] Online merge log:"
echo "------------------------------------------------------------"
cat "${n_online}/mergelog_mag.txt"

# Convert DVS to bvec/bval using nominal q-values and effective b-values
echo "------------------------------------------------------------"
echo "[DW-SSFP] Converting DVS to bvec/bval"
echo "------------------------------------------------------------"

fslpython ~/my_dvs2bvec.py \
  -dvs1 "${dvs_qlow}" \
  -dvs2 "${dvs_qhigh}" \
  -qb0 "${q0_nominal}" 0 \
  -qb1 "${q1_nominal}" "${bval_qlow}" \
  -qb2 "${q2_nominal}" "${bval_qhigh}" \
  -mlog "${n_online}/mergelog_mag.txt" \
  -out "${n_online}/"

cp "${n_online}/bvecs" "${bvecs}"
cp "${n_online}/bvals" "${bvals}"

# Fix NIFTI header and axes for DW-SSFP combined data
echo "------------------------------------------------------------"
echo "[DW-SSFP] Reorienting combined_mag.nii.gz to data.nii.gz"
echo "------------------------------------------------------------"

python3 /home/data/NIH_CONNECTS/nipype_dev/prep_dmri_reorient.py -i "${combined_mag}" -o "${data_nifti}" -v x y z -r 1 1 1 --hard-fix-header

# Flip bvecs (x, y, z) → (x, z, y) with positive signs
echo "------------------------------------------------------------"
echo "[DW-SSFP] Flipping bvecs (x,y,z → x,z,y)"
echo "------------------------------------------------------------"

cp "${bvecs}" "${bvecs_orig}"
fslpython ~/CMC_tools/data_prep/bvecflip_generic.py -in "${bvecs_orig}" -out "${bvecs}" -vr x z y -vf 1 1 1

# Restrict FOV - speeds up processing later
fslroi "${data_nifti}" "${data_nifti}" 90 160 70 200 0 -1 0 -1

# ------------------------------------------------------------
# STEP 2: Quick sanity dtifit (orientation check)
# ------------------------------------------------------------

# Block variables
dtifit_dir="${n_proc}/test_dtifit"
mask="${n_proc}/brain_mask"

echo "------------------------------------------------------------"
echo "[DTIFIT] Running quick sanity dtifit for orientation check"
echo "------------------------------------------------------------"

rm -rf "${dtifit_dir}"
mkdir -p "${dtifit_dir}"

# make mask using FA
fslmaths "${data_nifti}" -Tmean ${n_proc}/meanvols
fslmaths ${n_proc}/meanvols -thr 100 -ero -dilM -fillh ${mask}
fslmaths ${mask} -dilM -dilM -ero -ero ${mask}

rm -rf "${dtifit_dir}"
mkdir -p "${dtifit_dir}"

dtifit -k "${data_nifti}" -o "${dtifit_dir}/dti" -m "${mask}" -r "${bvecs}" -b "${bvals}"

echo "[DTIFIT] Inspect outputs visually as needed before proceeding."

# ------------------------------------------------------------
# STEP 3: Anatomicals and parameter maps (TSE, T1, T2 mapping, T2 ME)
# ------------------------------------------------------------

# 3.1 TSE anatomical (convert + reorient)
echo "------------------------------------------------------------"
echo "[TSE] Converting and reorienting TSE anatomical"
echo "------------------------------------------------------------"

tse_series="MR-SE017-tse_pt2iso"
tse_dicom_dir="${session_dir}/${tse_series}"

check_dir "${tse_dicom_dir}" "TSE anatomical"

dcm2niix -b y -z y -o "${n_online}" "${tse_dicom_dir}"

python3 /home/data/NIH_CONNECTS/nipype_dev/prep_dmri_reorient.py -i "${n_online}/${tse_series}"*.nii.gz -o "${n_proc}/TSE_anatomical.nii.gz" -v x z y -r -1 1 -1 --hard-fix-header

# 3.2 T1 map (convert + reorient + mean)
echo "------------------------------------------------------------"
echo "[T1] Converting and reorienting T1 map and INV2, computing mean T1"
echo "------------------------------------------------------------"

t1map_series="MR-SE039-t1_map_mp2rage_sag_p3_1mm_T1_Images"
t1inv2_series="MR-SE038-t1_map_mp2rage_sag_p3_1mm_INV2"
t1map_dicom_dir="${session_dir}/${t1map_series}"
t1inv2_dicom_dir="${session_dir}/${t1inv2_series}"
t1map_nifti="${n_online}/T1map.nii.gz"
t1mask_nifti="${n_online}/T1map_mask.nii.gz"
t1map_masked="${n_online}/T1map_masked.nii.gz"

check_dir "${t1map_dicom_dir}" "T1 map"
check_dir "${t1inv2_dicom_dir}" "T1 INV2"

# T1 map
dcm2niix -b y -z y -o "${n_online}" "${t1map_dicom_dir}"
python3 /home/data/NIH_CONNECTS/nipype_dev/prep_dmri_reorient.py -i "${n_online}/${t1map_series}"*.nii.gz -o "${t1map_nifti}" -v x z y -r -1 1 -1 --hard-fix-header

# INV2 for mask
dcm2niix -b y -z y -o "${n_online}" "${t1inv2_dicom_dir}"
python3 /home/data/NIH_CONNECTS/nipype_dev/prep_dmri_reorient.py -i "${n_online}/${t1inv2_series}"*.nii.gz -o "${t1mask_nifti}" -v x z y -r -1 1 -1 --hard-fix-header

fslmaths "${t1mask_nifti}" -thr 1500 -bin "${t1mask_nifti}"
fslmaths "${t1mask_nifti}" -fillh -ero -dilM -dilM "${t1mask_nifti}"

fslmaths "${t1map_nifti}" -mul "${t1mask_nifti}" "${t1map_masked}"

mean_T1=$(fslstats "${t1map_masked}" -M)
echo "mean T1 for whole brain: ${mean_T1}"
echo "${mean_T1}" > "${n_online}/meanT1.txt"

# 3.4 T2 multi-echo (user-specified series → 4D + TE.txt)
echo "------------------------------------------------------------"
echo "[T2ME] Converting and merging T2 multi-echo from user-specified series"
echo "------------------------------------------------------------"

t2me_series="MR-SE113-T2_SE_MC_RR" # if available, use the retrorecon version
t2_src_dir="${session_dir}/${t2me_series}"
t2_nifti="${n_online}/T2_nifti"
t2_merged="${n_online}/T2_merged"
t2_4d="${n_online}/T2_4D.nii.gz"

check_dir "${t2_src_dir}" "T2 multi-echo"

rm -rf "${t2_nifti}" "${t2_merged}"
mkdir -p "${t2_nifti}" "${t2_merged}"

# (1) convert
dcm2niix -z y -o "${t2_nifti}" "${t2_src_dir}"

# (2) merge in _eNN order and write TE.txt
pushd "${t2_nifti}" >/dev/null

mapfile -t T2_LIST < <(ls -1 *.nii.gz | awk 'match($0,/_e([0-9]+)/,a){printf "%04d %s\n",a[1],$0}' | sort -n | cut -d" " -f2-)

if [ ${#T2_LIST[@]} -eq 0 ]; then
  echo "[T2ME] ERROR: no *_eNN*.nii.gz in ${t2_nifti}"
  popd >/dev/null
else
  echo "[T2ME] Merge order:"; printf '  %s\n' "${T2_LIST[@]}"
  fslmerge -t "${t2_merged}/TSE_4D.nii.gz" "${T2_LIST[@]}"

  TE_TXT="${t2_merged}/TE.txt"
  : > "${TE_TXT}"
  for nif in "${T2_LIST[@]}"; do
    j="${nif%.nii.gz}.json"
    et=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("EchoTime","N/A"))' "$j" 2>/dev/null)
    echo "$(basename "$j"): $et" >> "${TE_TXT}"
  done
  echo "[T2ME] Echo times saved to: ${TE_TXT}"
  popd >/dev/null

  # (3) reorient header/axes
  python3 /home/data/NIH_CONNECTS/nipype_dev/prep_dmri_reorient.py \
    -i "${t2_merged}/TSE_4D.nii.gz" \
    -o "${t2_4d}" \
    -v x z y -r -1 1 -1 --hard-fix-header
fi

# ------------------------------------------------------------
# T2 MASK CREATION (Tmax → thr 200 → fillh → largest component → ero → dilM×2)
# ------------------------------------------------------------

echo "------------------------------------------------------------"
echo "[T2ME] Creating T2 mask from T2_4D"
echo "------------------------------------------------------------"

t2_max="${n_online}/T2_4D_max.nii.gz"
t2_mask="${n_online}/T2_mask.nii.gz"

fslmaths "${t2_4d}" -Tmax "${t2_max}"
fslmaths "${t2_max}" -thr 300 -bin "${t2_mask}"
fslmaths "${t2_mask}" -fillh "${t2_mask}"

# Keep largest component via fsl-cluster
t2_cluster_tmp="$(mktemp "${n_online}/T2_cluster_XXXX.txt")"
fsl-cluster -i "${t2_mask}" -o "${t2_mask}" -t 1 > "${t2_cluster_tmp}"

best_idx=$(awk 'NR>1 && NF>=2 { if ($2>max) {max=$2; best=$1} } END { if (max>0) print best }' "${t2_cluster_tmp}")
rm -f "${t2_cluster_tmp}"

echo "[T2ME] Keeping largest T2 cluster index: ${best_idx}"
fslmaths "${t2_mask}" -thr "${best_idx}" -uthr "${best_idx}" -bin "${t2_mask}"

# Erode then dilate twice
fslmaths "${t2_mask}" -ero -dilM -dilM -bin "${t2_mask}"

echo "[T2ME] T2 mask created: ${t2_mask}"

# ------------------------------------------------------------
# STEP 4: TurboFLASH B1 (explicit series for anat and flipmap)
# ------------------------------------------------------------

echo "------------------------------------------------------------"
echo "[B1] Converting and preparing B1 TurboFLASH maps"
echo "------------------------------------------------------------"

b1_anat_series="MR-SE006-tfl_b1map"
b1_flipmap_series="MR-SE007-tfl_b1map"
b1_anat_dicom_dir="${session_dir}/${b1_anat_series}"
b1_flipmap_dicom_dir="${session_dir}/${b1_flipmap_series}"
b1_nifti_dir="${n_online}/B1_nifti"

check_dir "${b1_anat_dicom_dir}" "B1 anatomical"
check_dir "${b1_flipmap_dicom_dir}" "B1 flipmap"

rm -rf "${b1_nifti_dir}"
mkdir -p "${b1_nifti_dir}"

# Convert both series
dcm2niix -z y -o "${b1_nifti_dir}" "${b1_anat_dicom_dir}"
dcm2niix -z y -o "${b1_nifti_dir}" "${b1_flipmap_dicom_dir}"

# Reorient both to processing space
python3 /home/data/NIH_CONNECTS/nipype_dev/prep_dmri_reorient.py \
  -i "${b1_nifti_dir}/${b1_anat_series}"*.nii.gz \
  -o "${b1_nifti_dir}/B1map_anat.nii.gz" \
  -v x z y -r -1 1 -1 --hard-fix-header

python3 /home/data/NIH_CONNECTS/nipype_dev/prep_dmri_reorient.py \
  -i "${b1_nifti_dir}/${b1_flipmap_series}"*.nii.gz \
  -o "${b1_nifti_dir}/B1map_flip.nii.gz" \
  -v x z y -r -1 1 -1 --hard-fix-header

# Scale flipmap by 900 (in-place)
fslmaths "${b1_nifti_dir}/B1map_flip.nii.gz" -div 900 "${b1_nifti_dir}/B1map_flip.nii.gz"

# ROI to remove noisy bands in y direction (apply to anat + flip)
fslroi "${b1_nifti_dir}/B1map_anat.nii.gz" "${b1_nifti_dir}/B1map_anat.nii.gz" 22 50 5 32 0 -1 0 -1
fslroi "${b1_nifti_dir}/B1map_flip.nii.gz" "${b1_nifti_dir}/B1map_flip.nii.gz" 22 50 5 32 0 -1 0 -1

# ------------------------------------------------------------
# B1 MASKING (your updated version – unchanged)
# ------------------------------------------------------------

b1_mask="${b1_nifti_dir}/B1_mask.nii.gz"
fslmaths "${b1_nifti_dir}/B1map_anat.nii.gz" -thr 1000 -bin "${b1_mask}"
fslmaths "${b1_mask}" -fillh -dilM -bin -ero -bin "${b1_mask}"

# Apply mask to flipmap and compute mean B1
b1_flip_masked="${b1_nifti_dir}/B1map_flip_masked.nii.gz"
fslmaths "${b1_nifti_dir}/B1map_flip.nii.gz" -mul "${b1_mask}" "${b1_flip_masked}"

mean_B1=$(fslstats "${b1_flip_masked}" -M)
echo "mean B1 for whole brain: ${mean_B1}"
echo "${mean_B1}" > "${n_online}/meanB1.txt"

echo "[B1] Created:"
echo "  ${b1_nifti_dir}/B1map_anat.nii.gz"
echo "  ${b1_nifti_dir}/B1map_flip.nii.gz"
echo "  ${b1_mask}"
echo "  ${b1_flip_masked}"
echo "  ${n_online}/meanB1.txt"


# ------------------------------------------------------------
# STEP 5: Prepare CFP files for DW-SSFP
# ------------------------------------------------------------

# Block variables
flipMask_txt="${n_proc}/flipMask"
noisefloor_txt="${n_proc}/noisefloor"
TRs_txt="${n_proc}/TRs"
flipAngles_txt="${n_proc}/flipAngles"
diffGradDurs_txt="${n_proc}/diffGradDurs"
diffGradAmps_txt="${n_proc}/diffGradAmps"
whichshell_txt="${n_proc}/whichshell"

echo "------------------------------------------------------------"
echo "[CFP] Preparing CFP text files for DW-SSFP"
echo "------------------------------------------------------------"

cp -f "${bvals}" "${flipMask_txt}"
fslpython ~/CMC_tools/tools/pyrep.py -i "${flipMask_txt}" -t "${bval_list_2}" -r 0,1

cp -f "${flipMask_txt}" "${noisefloor_txt}"
fslpython ~/CMC_tools/tools/pyrep.py -i "${noisefloor_txt}" -t 1 -r 0

cp -f "${noisefloor_txt}" "${TRs_txt}"
fslpython ~/CMC_tools/tools/pyrep.py -i "${TRs_txt}" -t 0 -r 0.032 -dtype float

cp -f "${noisefloor_txt}" "${flipAngles_txt}"
fslpython ~/CMC_tools/tools/pyrep.py -i "${flipAngles_txt}" -t 0 -r 14

cp -f "${bvals}" "${diffGradDurs_txt}"
fslpython ~/CMC_tools/tools/pyrep.py -i "${diffGradDurs_txt}" -t "${bval_list_3}" -r 0.0125,0.0125,0.0125 -dtype float -p 5

cp -f "${bvals}" "${diffGradAmps_txt}"
fslpython ~/CMC_tools/tools/pyrep.py -i "${diffGradAmps_txt}" -t "${bval_list_3}" -r 0.94,3.76,5.17 -dtype float -p 2

cp -f "${bvals}" "${whichshell_txt}"
fslpython ~/CMC_tools/tools/pyrep.py -i "${whichshell_txt}" -t "${bval_list_3}" -r 0,0,1

echo "------------------------------------------------------------"
echo "[DONE] Pre-proc finished. Check key outputs in ${n_proc}."
echo "  - DW-SSFP data: data.nii.gz, bvecs, bvals"
echo "  - T1/T2 maps: T1map_masked.nii.gz, T2map_masked.nii.gz"
echo "  - Mean values: meanT1.txt, meanB1.txt"
echo "------------------------------------------------------------"
