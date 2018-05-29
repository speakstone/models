# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
#echo -e ${WORK_DIR}

# Update PYTHONPATH.
cd "/opt/data1/tensorflow_model/models/research/"
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

cd "${CURRENT_DIR}"

SIZE=513

# Set up the working directories.
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

FACE_SEG_DATASET="${WORK_DIR}/${DATASET_DIR}/${ADE_FOLDER}/tfrecord_classchange"

NUM_ITERATIONS=30000
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="mobilenet_v2" \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=${SIZE} \
  --train_crop_size=${SIZE} \
  --train_batch_size=4 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=True \
  --initialize_last_layer=False \
  --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_mnv2_pascal_trainval/model.ckpt-30000" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${FACE_SEG_DATASET}" \
  --dataset="supervisely_seg" \
  --base_learning_rate=5e-5
  #--learning_policy="step" \
  #--learning_rate_decay_step=5000 \
  #--learning_rate_decay_factor=0.5

  #--min_resize_value=350 \
  #--max_resize_value=500 \
  #--resize_factor=16 \

  #--tf_initial_checkpoint="${INIT_FOLDER}/xception/model.ckpt" \

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
  --num_classes=3 \
  --crop_size=${SIZE} \
  --crop_size=${SIZE} \
  --inference_scales=1.0
