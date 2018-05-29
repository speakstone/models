import collections
import os
import StringIO
import sys
import time

import numpy as np
np.set_printoptions(threshold='nan')
from PIL import Image
from scipy.misc import imsave

from tensorflow.python.platform import gfile
import  cv2
import tensorflow as tf

if tf.__version__ < '1.5.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.5.0 or newer!')

# Needed to show segmentation colormap labels
sys.path.append('/home/lilei/deeplab_change/deeplab/utils')
import get_dataset_colormap

#Load model in TensorFlow

_FROZEN_GRAPH_EXACT_NAME = '/home/lilei/deeplab_change/deeplab/models/deeplabv3_pascal_trainval/frozen_inference_graph.pb'
_FROZEN_GRAPH_FAST_NAME = '/home/lilei/deeplab_change/deeplab/models/deeplabv3_mnv2_pascal_trainval/frozen_inference_graph.pb'

class DeepLabEXACTModel(object):
    """Class to load deeplab model and run inference."""
    
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513

    def __init__(self):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        
        graph_def = None
        # Extract frozen graph from tar archive.
        with gfile.FastGFile(_FROZEN_GRAPH_EXACT_NAME, 'rb') as f:
            graph_def = tf.GraphDef.FromString(f.read())
        
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():      
            tf.import_graph_def(graph_def, name='')
        
        self.sess = tf.Session(graph=self.graph)
            
    def run(self, image):
        """Runs inference on a single image.
        
        Args:
            image: A PIL.Image object, raw input image.
            
        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map
		
class DeepLabFASTModel(object):
    """Class to load deeplab model and run inference."""
    
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE =  513

    def __init__(self):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        
        graph_def = None
        # Extract frozen graph from tar archive.
        with gfile.FastGFile(_FROZEN_GRAPH_FAST_NAME, 'rb') as f:
            graph_def = tf.GraphDef.FromString(f.read())
        
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():      
            tf.import_graph_def(graph_def, name='')
        
        self.sess = tf.Session(graph=self.graph)
            
    def run(self, image):
        """Runs inference on a single image.
        
        Args:
            image: A PIL.Image object, raw input image.
            
        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map
		
model_exact = DeepLabEXACTModel()
model_fast = DeepLabFASTModel()


#Helper methods
LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tv'
])


FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = get_dataset_colormap.label_to_color_image(FULL_LABEL_MAP)

def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

#Run on sample images
# Note that we are using single scale inference in the demo for fast
# computation, so the results may slightly differ from the visualizations
# in README, which uses multi-scale and left-right flipped inputs.

#IMAGE_DIR = '/home/jxd/workspace/segment/deeplab/images/hq_demo/'
#IMAGE_DIR = '/home/lilei/deeplab_change/deeplab/img/Test/Train/'
#OUTPUT_DIR = '/home/lilei/deeplab_change/deeplab/img/Test/Test/'
#if not os.path.isdir(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

#def run_exact_demo_image(image_name):
#    try:
#        image_path = os.path.join(IMAGE_DIR, image_name)
#        orignal_im = Image.open(image_path)
#    except IOError:
#        print 'Failed to read image from %s.' % image_path 
#        return 
#    print 'running deeplab on image %s...' % image_name
#    resized_im, seg_map = model_exact.run(orignal_im)
#    return seg_map
#	
#def run_fast_demo_image(image_name):
#    try:
#        image_path = os.path.join(IMAGE_DIR, image_name)
#        orignal_im = Image.open(image_path)
#    except IOError:
#        print 'Failed to read image from %s.' % image_path 
#        return 
#    print 'running deeplab on image %s...' % image_name
#    resized_im, seg_map = model_fast.run(orignal_im)
#    return seg_map


#for file in os.listdir(IMAGE_DIR):
#    time_start = time.clock()
#    if (file[-3:] != "jpg" and file[-3:] != "png"): continue
#    seg_map = run_demo_image(file).astype(np.uint8)
#    seg_image = label_to_color_image(seg_map).astype(np.uint8)
#	
#	
#    imsave(OUTPUT_DIR + file[:-4] + "_seg" + file[-4:], seg_map)
#    imsave(OUTPUT_DIR + file[:-4] + "_seg_color" + file[-4:], seg_image)
#    unique_labels = np.unique(seg_map)
#    time_end = time.clock()
#    print file, unique_labels
#    print LABEL_NAMES[unique_labels]
#    print('Running time: %s Seconds'%(time_end-time_start))
IMAGE_BACK = cv2.imread('/home/lilei/deeplab_change/deeplab/img/back.jpg')    
cap = cv2.VideoCapture('img/test.avi')  
img_num = 0


while(cap.isOpened()): 

    time_start = time.clock()
    ret, frame = cap.read() 
    frame = Image.fromarray(np.uint8(frame))
    if img_num == 0:     ####The first frame is handled separately by model_exact
      resized_im, seg_map = model_exact.run(frame) 
      seg_map = seg_map.astype(np.uint8)
      Previous_seg = seg_map.astype('bool').astype('float')

    else:
      resized_im, seg_map = model_fast.run(frame) 
      seg_map = seg_map.astype(np.uint8)
    seg_map =  np.asarray(seg_map).astype('float') 
    Current_seg = seg_map.astype('bool').astype('float') #####change segment
    MIX_Current_seg =  Current_seg*0.5 + Previous_seg*0.5
#    if img_num < 1: MIX_Current_seg = Current_seg ; img_num = 1####get new segment by Previous_seg
    
    Previous_seg = MIX_Current_seg
    Back_seg = np.ones(list(Previous_seg.shape)).astype('int') - MIX_Current_seg   ####get background segment 
    IMAGE_BACK = cv2.resize(IMAGE_BACK, ( (list(Previous_seg.shape))[1], (list(Previous_seg.shape))[0]) , interpolation=cv2.INTER_CUBIC )
    IMAGE_BACK = np.asarray(IMAGE_BACK).astype('float')
    IMAGE_DIR = np.asarray(resized_im).astype('float')
    IMAGE_BACK_DIR = np.zeros(list(np.array(IMAGE_DIR).shape)).astype('float')
    
#    print IMAGE_BACK_DIR.shape
#    print IMAGE_BACK.shape
#    print Previous_seg.shape
#    print Back_seg.shape
    IMAGE_BACK_DIR[:,:,0] = IMAGE_DIR[:,:,0]*Previous_seg + IMAGE_BACK[:,:,0]*Back_seg
    IMAGE_BACK_DIR[:,:,1] = IMAGE_DIR[:,:,1]*Previous_seg + IMAGE_BACK[:,:,1]*Back_seg
    IMAGE_BACK_DIR[:,:,2] = IMAGE_DIR[:,:,2]*Previous_seg + IMAGE_BACK[:,:,2]*Back_seg

    
    cv2.imwrite('img/image'+str(img_num) +'.jpg', IMAGE_BACK_DIR) 
    cv2.imshow('test', IMAGE_BACK_DIR)

    img_num += 1
    k = cv2.waitKey(10)  

    time_end = time.clock()
    print('Running time: %s Seconds'%(time_end-time_start))
    if (k & 0xff == ord('q')):  
        break  


