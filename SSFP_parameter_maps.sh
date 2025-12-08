#!/usr/bin/env bash

#######################################################################
# CONFIG
#######################################################################

# Docker Gibbs settings
CONTAINER_RUNTIME="docker"
GIBBS_IMAGE="docker.io/mrtrix3/mrtrix3:3.0.7"
INTERACTIVE_TTY=1     # set 0 on HPC without tty

# KT modelling root
KT_ROOT="~/"
MATLAB_BIN="/usr/local/MATLAB/R2023a/bin/matlab"

# Fixed registration target
REG_TARGET="/home/data/NIH_CONNECTS/NHP_Exvivo/Pear_Oct2025/NIFTI_for_proc/data.nii.gz"

#######################################################################
# ENV / PATHS
#######################################################################

echo "[INFO] Running in NIFTI_online: $(pwd)"

NIFTI_DIR="/home/data/NIH_CONNECTS/NHP_Exvivo/Pear_Oct2025/NIFTI_online"
DMRI_ROOT="$(dirname "${NIFTI_DIR}")"

T1_RAW="${NIFTI_DIR}/T1map_masked.nii.gz"
T2_4D_RAW="${NIFTI_DIR}/T2_4D.nii.gz"
B1_ANAT_RAW="${NIFTI_DIR}/B1_nifti/B1map_anat.nii.gz"
B1_FLIP_RAW="${NIFTI_DIR}/B1_nifti/B1map_flip_masked.nii.gz"
TE_TXT_RAW="${NIFTI_DIR}/T2_merged/TE.txt"

#######################################################################
# SAFE CHECKERS (do NOT exit)
#######################################################################

check_file() {
  if [ ! -f "$1" ]; then
    echo "[WARN] Missing file: $1"
  fi
}

#######################################################################
# Sanitize TE
#######################################################################

sanitize_te() {
  local src="$1"
  local dst="$2"
  : > "$dst"

  while IFS= read -r line; do
    line="$(echo "$line" | sed 's/^[ \t]*//;s/[ \t]*$//')"
    [ -z "$line" ] && continue

    if [[ "$line" == *:* ]]; then
      line="${line#*:}"
      line="$(echo "$line" | sed 's/^[ \t]*//')"
    fi

    val="$(echo "$line" | awk '{print $NF}')"
    echo "$val" >> "$dst"
  done < "$src"
}

#######################################################################
# Docker Gibbs wrapper
#######################################################################

docker_mrdegibbs() {
  local in="$1"
  local out="$2"
  local axes="$3"

  local in_abs out_abs
  in_abs="$(readlink -f "$in")"
  out_abs="$(readlink -f "$out")"

  local tty_flag=""
  if [ "$INTERACTIVE_TTY" -eq 1 ]; then
    tty_flag="-it"
  fi

  echo "[GIBBS] $in_abs -> $out_abs  (axes=$axes)"

  ${CONTAINER_RUNTIME} run --rm ${tty_flag} \
    --user 0:0 \
    -v "${DMRI_ROOT}:${DMRI_ROOT}" \
    -w "${NIFTI_DIR}" \
    "${GIBBS_IMAGE}" \
    mrdegibbs "$in_abs" "$out_abs" -axes "$axes" -force
}

gibbs_all_axes() {
  local in="$1"
  local out="$2"

  local xy="${out%.nii.gz}_xy.nii.gz"
  local xz="${out%.nii.gz}_xy_xz.nii.gz"

  docker_mrdegibbs "$in" "$xy" "0,1"
  docker_mrdegibbs "$xy" "$xz" "0,2"
  docker_mrdegibbs "$xz" "$out" "1,2"

  rm -f "$xy" "$xz"
}

#######################################################################
# 1. GIBBS UNRINGING
#######################################################################

echo "============================================================"
echo "[STEP 1] Gibbs unringing for T1, T2, B1"
echo "============================================================"

T1_GIBBS="${NIFTI_DIR}/T1map_gibbs.nii.gz"
T2_GIBBS="${NIFTI_DIR}/T2_4D_gibbs.nii.gz"
B1_FLIP_GIBBS="${NIFTI_DIR}/B1_nifti/B1map_flip_gibbs.nii.gz"

gibbs_all_axes "$T1_RAW"      "$T1_GIBBS"
gibbs_all_axes "$T2_4D_RAW"   "$T2_GIBBS"
gibbs_all_axes "$B1_FLIP_RAW" "$B1_FLIP_GIBBS"

#######################################################################
# 2. TE sanitisation
#######################################################################

echo "============================================================"
echo "[STEP 2] Sanitising TE.txt"
echo "============================================================"

sanitize_te "${TE_TXT_RAW}" "${NIFTI_DIR}/TE.txt"
echo "[INFO] Sanitised TE written to ${NIFTI_DIR}/TE.txt"

#######################################################################
# 3. KT MODELLING
#######################################################################

echo "============================================================"
echo "[STEP 3] KT modelling"
echo "============================================================"

KT_DIR="${NIFTI_DIR}/T2mapping"
mkdir -p "$KT_DIR"

MATLAB_CMD="addpath('${KT_ROOT}/KT_model-master/EPG'); \
addpath('${KT_ROOT}/KT_model-master/EPG/bin'); \
out_path='T2mapping/'; \
TSE_path='T2_4D_gibbs.nii.gz'; \
TE_path='TE.txt'; \
mask_path='T2_mask.nii.gz'; \
EPG_wrapper_step1_parfor(out_path,TSE_path,TE_path,mask_path); exit;"

(
  cd "$NIFTI_DIR"
  "${MATLAB_BIN}" -nodisplay -nosplash -r "$MATLAB_CMD"
)

# ------------------------------------------------------------
# Copy geometry from T2_mask to ALL KT outputs
# ------------------------------------------------------------

T2_MASK="${NIFTI_DIR}/T2_mask.nii.gz"

echo "[KT] Copying geometry from T2_mask.nii.gz to all KT outputs…"

for f in "${KT_DIR}"/*.nii.gz; do
    echo "  → $(basename "$f")"
    fslcpgeom "$T2_MASK" "$f"
    fslorient -copysform2qform "$f" >/dev/null 2>&1
done

# ------------------------------------------------------------
# Make clipped T2map (keep original name for T2map.nii.gz)
# ------------------------------------------------------------
T2MAP="${KT_DIR}/T2map.nii.gz"
T2MAP_CLIP="${KT_DIR}/T2map_clipped.nii.gz"

echo "[KT] Clipping T2map → T2map_clipped"
fslmaths "$T2MAP" -thr 0 -uthr 200 "$T2MAP_CLIP"

mean_T2=$(fslstats "$T2MAP_CLIP" -M)
echo "mean T2 (clipped): ${mean_T2}"
echo "${mean_T2}" > "${NIFTI_DIR}/meanT2.txt"

#######################################################################
# 4. REGISTRATION (all → REG_TARGET)
#######################################################################

echo "============================================================"
echo "[STEP 4] Registration to ${REG_TARGET}"
echo "============================================================"

check_file "$REG_TARGET"

# 4a. T1 registration (non-linear)
T1_COREF="${NIFTI_DIR}/T1map_coreg.nii.gz"
T1_MAT="${NIFTI_DIR}/T1_to_tgt.mat"
T1_WARP="${NIFTI_DIR}/T1_to_tgt_warpcoef.nii.gz"

# Linear pre-alignment
flirt -in "$T1_GIBBS" -ref "$REG_TARGET" \
      -omat "$T1_MAT" \
      -dof 12 -interp trilinear

# Non-linear warp (driven by T1_GIBBS)
fnirt --in="$T1_GIBBS" --ref="$REG_TARGET" \
      --aff="$T1_MAT" \
      --iout="$T1_COREF" \
      --cout="$T1_WARP" -v

# 4b. B1 registration (non-linear, using raw B1 anat as driver)
echo "[REG] B1 registration (non-linear)"

B1_MAT="${NIFTI_DIR}/B1_to_tgt.mat"
B1_WARP="${NIFTI_DIR}/B1_to_tgt_warpcoef.nii.gz"
B1_ANAT_COREF="${NIFTI_DIR}/B1map_anat_coreg.nii.gz"
B1_FLIP_COREF="${NIFTI_DIR}/B1map_flip_coreg.nii.gz"

# Linear pre-alignment: B1map_anat_raw → target
flirt -in "$B1_ANAT_RAW" -ref "$REG_TARGET" \
      -omat "$B1_MAT" \
      -out "$B1_ANAT_COREF" \
      -dof 12 -interp trilinear

# Non-linear warp driven by B1map_anat_raw
fnirt --in="$B1_ANAT_RAW" --ref="$REG_TARGET" \
      --aff="$B1_MAT" \
      --iout="$B1_ANAT_COREF" \
      --cout="$B1_WARP" -v

# Apply B1 warp to Gibbs-unringed flipmap
applywarp --in="$B1_FLIP_GIBBS" \
          --ref="$REG_TARGET" \
          --warp="$B1_WARP" \
          --out="$B1_FLIP_COREF"

# 4c. T2 registration (non-linear, apply warp to T2map_clipped)
echo "[REG] T2 registration (non-linear)"

T2_MAT="${NIFTI_DIR}/T2_to_tgt.mat"
T2_WARP="${NIFTI_DIR}/T2_to_tgt_warpcoef.nii.gz"
T2_COREF="${NIFTI_DIR}/T2_coreg.nii.gz"
T2MAP_COREF="${NIFTI_DIR}/T2map_coreg.nii.gz"

# Linear pre-alignment: T2_4D_gibbs → target
flirt -in "$T2_GIBBS" -ref "$REG_TARGET" \
      -omat "$T2_MAT" \
      -out "$T2_COREF" \
      -dof 12 -interp trilinear

# Non-linear warp driven by full T2_4D_gibbs
fnirt --in="$T2_GIBBS" --ref="$REG_TARGET" \
      --aff="$T2_MAT" \
      --iout="$T2_COREF" \
      --cout="$T2_WARP" -v

# Apply T2 warp to clipped T2map
applywarp --in="$T2MAP_CLIP" \
          --ref="$REG_TARGET" \
          --warp="$T2_WARP" \
          --out="$T2MAP_COREF"

#######################################################################
# 5. DILATE FINAL IMAGES TO SET ALL BACKGROUND TO AVERAGE BRAIN VALUES
#######################################################################

OUT_DIR="$(dirname "$REG_TARGET")"

fslmaths "$T1_COREF"     -dilall "${OUT_DIR}/T1map"
fslmaths "$B1_FLIP_COREF" -dilall "${OUT_DIR}/B1map"
fslmaths "$T2MAP_COREF"  -dilall "${OUT_DIR}/T2map"

#######################################################################
# SUMMARY
#######################################################################

echo ""
echo "============================================================"
echo "[DONE] SSFP dependency prep COMPLETE"
echo "Output coreg files:"
echo "  T1map → ${OUT_DIR}/T1map"
echo "  B1map → ${OUT_DIR}/B1map"
echo "  T2map → ${OUT_DIR}/T2map"
echo "============================================================"
echo ""
