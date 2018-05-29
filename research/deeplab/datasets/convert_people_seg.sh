

set -e

CURRENT_DIR=$(pwd)
WORK_DIR="."

cd "${CURRENT_DIR}"

# Root path for FACE_SEG_ROOT dataset.
FACE_SEG_ROOT="${WORK_DIR}/face_seg"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${FACE_SEG_ROOT}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

BUILD_SCRIPT="${WORK_DIR}/build_face_seg_data.py"

echo "Converting FACE_SEG dataset..."
python "${BUILD_SCRIPT}" \
  --FACE_SEG_ROOT="${FACE_SEG_ROOT}" \
  --output_dir="${OUTPUT_DIR}" \
