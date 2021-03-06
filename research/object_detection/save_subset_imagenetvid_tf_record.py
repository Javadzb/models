# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_imagenetvid_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  #img_path = os.path.join(data['folder'], image_subdirectory,
                          #data['filename']+'.JPEG')
  img_path = os.path.join('Data/VID/train/',data['folder'], data['filename']+'.JPEG')


  full_path = os.path.join(dataset_directory, img_path)

  #print ('full_path= ',full_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()
  encoded_key = key.encode('utf8')

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      # No difficult flag in ImageNet-VId
      #difficult = bool(int(obj['difficult']))
      #if ignore_difficult_instances and difficult:
        #continue

      #difficult_obj.append(int(difficult))

      if obj['name']!='n02834778' and \
             obj['name']!='n02958343' and \
         obj['name']!='n03790512':
          print('out of scope class= ', obj['name'])
          print('object class is not among the target classes')
          continue

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)


      classes_text.append(obj['name'].encode('utf8'))
      classes.append(label_map_dict[obj['name']])

      #truncated.append(int(obj['truncated']))
      #poses.append(obj['pose'].encode('utf8'))


  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(encoded_key),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example

def save_tf_record(data_info,indices):
    #if FLAGS.set not in SETS:
        #raise ValueError('set must be in : {}'.format(SETS))

    data_dir = data_info['data_dir']
    output_path = data_info['output_path']
    writer = tf.python_io.TFRecordWriter(output_path)

    label_map_dict = label_map_util.get_label_map_dict(data_info['label_map_path'])

    logging.info('Reading from ImageNet-VID dataset.')
    examples_path = os.path.join(data_dir,'AL', data_info['set'] + '.txt')

    # Annotations always come from train set now (revisit if we include val)
    annotations_dir = os.path.join(data_dir, data_info['annotations_dir'],'VID','train')
    examples_list = dataset_util.read_examples_list(examples_path)

    for idx, example in enumerate(examples_list):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples_list))

      if idx in indices:

          example_xml = example[-63:-5]+'.xml'
          path = os.path.join(annotations_dir,example_xml) # indexing of example to remove .JPEG from the end of file name
          #path = os.path.join(annotations_dir, example[-63:-5] + '.xml') # indexing of example to remove .JPEG from the end of file name
          #print ('Annotation path= ',path)


          with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
          xml = etree.fromstring(xml_str)
          data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

          tf_example = dict_to_tf_example(data, data_info['data_dir'], label_map_dict,
                                          FLAGS.ignore_difficult_instances)
          writer.write(tf_example.SerializeToString())
          #pdb.set_trace()


    writer.close()

def save_tf_record_val(data_info,indices):
    #if FLAGS.set not in SETS:
        #raise ValueError('set must be in : {}'.format(SETS))

    data_dir = data_info['data_dir']
    output_path = data_info['output_path']
    writer = tf.python_io.TFRecordWriter(output_path)

    label_map_dict = label_map_util.get_label_map_dict(data_info['label_map_path'])

    logging.info('Reading from ImageNet-VID dataset.')
    examples_path = os.path.join(data_dir,'AL', data_info['set'] + '.txt')

    # Annotations always come from train set now (revisit if we include val)
    annotations_dir = os.path.join(data_dir, data_info['annotations_dir'],'VID','val')
    examples_list = dataset_util.read_examples_list(examples_path)

    for idx, example in enumerate(examples_list):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples_list))

      if idx in indices:

          #example_xml = example[-63:-5]+'.xml'
          example_xml = example[-35:-5]+'.xml'
          path = os.path.join(annotations_dir,example_xml) # indexing of example to remove .JPEG from the end of file name


          with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
          xml = etree.fromstring(xml_str)
          data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

          tf_example = dict_to_tf_example(data, data_info['data_dir'], label_map_dict,
                                          FLAGS.ignore_difficult_instances)
          writer.write(tf_example.SerializeToString())


    writer.close()



if __name__ == '__main__':
  tf.app.run()
