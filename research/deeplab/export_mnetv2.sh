# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

cd "/opt/data1/tensorflow_model/models/research/"
# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

cd "${CURRENT_DIR}"

SIZE=513

# Set up the working directories.
#ADE_FOLDER="ADE20K_SP/ADEChallengeData2016"
#ADE_FOLDER="landscape"
DATASET_DIR="datasets"
ADE_FOLDER="ADE20K_SP/Supervisely"
EXP_FOLDER="exp_mobilenet/train_on_train_set"
INIT_FOLDER="${WORK_DIR}/models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ADE_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ADE_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ADE_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${ADE_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

NUM_ITERATIONS=30000
  # Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

python "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="mobilenet_v2" \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=2 \
  --crop_size=${SIZE} \
  --crop_size=${SIZE} \
  --inference_scales=1.0
