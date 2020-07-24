import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
import vis_utils

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_name = model_name.split('/')[-1]
  model_dir = tf.keras.utils.get_file(
    fname=model_name,
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"
  print(model_dir)
  model = tf.saved_model.load(str(model_dir))

  return model

# def load_model_using_path(model_dir):
#   model = tf.saved_model.load(str(model_dir))
#   return model

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/nadeem/Desktop/tf/models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


import os
import pathlib

# http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
# http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
model_name = 'tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8'
detection_model = load_model(model_name)
# model_dir = "/home/nadeem/.keras/datasets/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/saved_model"
# detection_model = load_model_using_path(model_dir)

#print(detection_model.signatures['serving_default'].inputs)

detection_model.signatures['serving_default'].output_dtypes

detection_model.signatures['serving_default'].output_shapes


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

def EncodeDecodeLabels():
    encode = {}
    decode = {}
    for key, value in category_index.items():
    #    print(key)
    #    print(type(value))
        Id = 0
        Name = 'name'
        for ki, val in value.items():
            if ki == 'id':
                Id = val
            if ki == 'name':
                Name = val
        encode[Id] = Name
        decode[Name] = Id
    encode, decode = decode, encode
    return encode, decode

encode, decode = EncodeDecodeLabels()

def show_inference(model, image_path, objects_list, detection_score=.5):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  detection_boxes = output_dict['detection_boxes']
  detection_classes = output_dict['detection_classes']
  detection_scores = output_dict['detection_scores']
  n = detection_scores.shape[0]
  output_boxes = []
  cnt = 1
  for i in range(n):
    if detection_scores[i]>detection_score:
        dict = {}
        dict["score"] = detection_scores[i]
        dict["object_name"] = decode[detection_classes[i]]
        dict["box_coordinates"] = str(detection_boxes[i])
        print("Box Number:"+str(cnt))
        print("detection_score: "+str(detection_scores[i]))
        print("detection_class: "+str(detection_classes[i]))
        print("detection_boxes: "+str(detection_boxes[i]))
        output_boxes.append(dict)
        cnt+=1
  print(objects_list)
  vis_utils.visualize_boxes_and_labels_on_image_array_improved(
      image_np,
      objects_list,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8,
      min_score_thresh=detection_score)
  im = Image.fromarray(image_np)
  image_path = str(image_path)
  image_path = image_path.split('/')[-1][:-4]+'_res.jpg'
  im.save(image_path)
#  im.show()
  print(image_path)
  output_dix = {}
  output_dix["objects"] = output_boxes
  return image_path, output_dix

def RunInference(image_path, objects_list, detection_score=.5):
  return show_inference(detection_model, image_path, objects_list, detection_score=detection_score)
#print(category_index)
#print(type(category_index))
"""
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('/home/nadeem/Desktop/tf/models/research/object_detection/test_images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
print(TEST_IMAGE_PATHS)

cnt = 1
for image_path in TEST_IMAGE_PATHS:
  print("Processing Image "+str(cnt))
  show_inference(detection_model, image_path)
  cnt+=1
"""
