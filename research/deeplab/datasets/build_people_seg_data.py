
import glob
import math
import os.path
import re
import sys
import build_data
import tensorflow as tf
import copy

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('FACE_SEG_ROOT',
                           './face_seg',
                           'FACE_SEG_ROOT dataset root folder.')

tf.app.flags.DEFINE_string(
    'output_dir',
    './tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')


_NUM_SHARDS = {
  'train': 5,
  'val': 1
}
# A map from data type to folder name that saves the data.
_FOLDERS_MAP = {
    'image': 'images',
    'label': 'annotations',
}

def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, val).

  Raises:
    RuntimeError: If loaded image and label have different shape, or if the
      image file with specified postfix could not be found.
  """
  image_files = []
  label_files = []

  image_dir = os.path.join(FLAGS.FACE_SEG_ROOT, _FOLDERS_MAP["image"], dataset_split)
  label_dir = os.path.join(FLAGS.FACE_SEG_ROOT, _FOLDERS_MAP["label"], dataset_split)
  image_list = sorted(os.listdir(image_dir))
  for i in range(len(image_list)):
    file = image_list[i]
    image_files.append(os.path.join(image_dir, file))
    label_files.append(os.path.join(label_dir, file[:-4] + ".png"))

  num_images = len(image_files)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS[dataset_split])))

  image_reader = build_data.ImageReader('jpg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS[dataset_split]):
    shard_filename = '%s-%05d-of-%05d.tfrecord' % (
        dataset_split, shard_id, _NUM_SHARDS[dataset_split])
    output_filename = os.path.join(FLAGS.output_dir, shard_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, num_images, shard_id))
        sys.stdout.flush()
        # Read the image.
        image_data = tf.gfile.FastGFile(image_files[i], 'r').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_data = tf.gfile.FastGFile(label_files[i], 'r').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)

        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        filename = os.path.basename(image_files[i][:-4])
        example = build_data.image_seg_to_tfexample(
            image_data, filename, height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  # Only support converting 'train', 'test' sets for now.
  for dataset_split in ['val']:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  tf.app.run()
