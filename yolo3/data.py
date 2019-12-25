import tensorflow as tf
from functools import reduce
from yolo3.utils import get_random_data, preprocess_true_boxes
from yolo3.enum import DATASET_MODE
import tensorflow_datasets as tfds
import numpy as np
AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset:

    def parse_tfrecord(self, example_proto):
        feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/label': tf.io.VarLenFeature(tf.int64)
        }
        features = tf.io.parse_single_example(example_proto,
                                              feature_description)
        image = tf.image.decode_image(features['image/encoded'],
                                      channels=3,
                                      dtype=tf.float32)
        image.set_shape([None, None, 3])
        xmins = features['image/object/bbox/xmin'].values
        xmaxs = features['image/object/bbox/xmax'].values
        ymins = features['image/object/bbox/ymin'].values
        ymaxs = features['image/object/bbox/ymax'].values
        labels = features['image/object/bbox/label'].values
        image, bbox = get_random_data(image,
                                      xmins,
                                      xmaxs,
                                      ymins,
                                      ymaxs,
                                      labels,
                                      self.input_shape,
                                      train=self.mode == DATASET_MODE.TRAIN)
        y1, y2, y3 = tf.py_function(
            preprocess_true_boxes,
            [bbox, self.input_shape, self.anchors, self.num_classes],
            [tf.float32, tf.float32, tf.float32])
        y1.set_shape([None, None, len(self.anchors) // 3, self.num_classes + 5])
        y2.set_shape([None, None, len(self.anchors) // 3, self.num_classes + 5])
        y3.set_shape([None, None, len(self.anchors) // 3, self.num_classes + 5])

        return image, (y1, y2, y3)

    def parse_text(self, line):
        print("line:", line)
        
        values = tf.strings.split([line], ' ').values
        image = tf.image.decode_image(tf.io.read_file(values[0]),
                                      channels=3,
                                      dtype=tf.float32)
        #print(np.shape(values))
        image.set_shape([None, None, 3])

        reshaped_data = tf.reshape(values[1:], [-1, 3])
        xs = tf.strings.to_number(reshaped_data[:, 0], tf.float32)
        ys = tf.strings.to_number(reshaped_data[:, 1], tf.float32)
        labels = tf.strings.to_number(reshaped_data[:, 2], tf.int64)

        # TODO: add data argumentation logic

        xs = tf.expand_dims(xs, 0)
        ys = tf.expand_dims(ys, 0)
        labels = tf.expand_dims(labels, 0)

        bbox = tf.concat([xs, ys, tf.cast(labels, tf.float32)], 0)
        bbox = tf.transpose(bbox, [1, 0])

        y1, y2 = tf.py_function(
            preprocess_true_boxes,
            [bbox, self.input_shape, self.num_classes],
            [tf.float32, tf.float32])

        y1.set_shape([None, None, self.num_classes + 3])
        y2.set_shape([None, None, self.num_classes + 3])

        return image, (y1, y2)

    def _dataset_internal(self, files, dataset_builder, parser):
        dataset = tf.data.Dataset.from_tensor_slices(files)
        if self.mode == DATASET_MODE.TRAIN:
            train_num = reduce(
                lambda x, y: x + y,
                map(lambda file: int(self._get_num_from_name(file)), files))
            print("train_num:",train_num)
            dataset = dataset.interleave(
                lambda file: dataset_builder(file),
                cycle_length=len(files),
                num_parallel_calls=AUTOTUNE).shuffle(train_num).map(
                    parser, num_parallel_calls=AUTOTUNE).prefetch(
                        self.batch_size).batch(self.batch_size).repeat()
        elif self.mode == DATASET_MODE.VALIDATE:
            dataset = dataset.interleave(
                lambda file: dataset_builder(file),
                cycle_length=len(files),
                num_parallel_calls=AUTOTUNE).map(
                    parser, num_parallel_calls=AUTOTUNE).prefetch(
                        self.batch_size).batch(self.batch_size).repeat()
        elif self.mode == DATASET_MODE.TEST:
            dataset = dataset.interleave(
                lambda file: dataset_builder(file),
                cycle_length=len(files),
                num_parallel_calls=AUTOTUNE).map(
                    parser, num_parallel_calls=AUTOTUNE).prefetch(
                        self.batch_size).batch(self.batch_size)
        return dataset

    def __init__(self,
                 glob_path: str,
                 batch_size: int,
                 num_classes=None,
                 input_shapes=None,
                 mode=DATASET_MODE.TRAIN):
        self.glob_path = glob_path
        self.batch_size = batch_size
        if isinstance(input_shapes, list):
            self.input_shapes = input_shapes
            self.input_shape = tf.Variable(name="input_shape",
                                           initial_value=self.input_shapes,
                                           trainable=False)
            print("init input shape")
            print(self.input_shape)
        else:
            self.input_shape = input_shapes
            print("init  else input shape")
            print(self.input_shape)
        self.num_classes = num_classes
        self.mode = mode

    def _get_num_from_name(self, name):
        return int(name.split('/')[-1].split('.')[0].split('_')[-1])

    def build(self, split=None):
        if self.glob_path is None:
            return None,0
        print("hi 1")
        print(self.glob_path)
        if self.glob_path in tfds.list_builders():
            return tfds.load(name=self.glob_path,
                             split=split,
                             with_info=True,
                             as_supervised=True,
                             try_gcs=tfds.is_dataset_on_gcs(self.glob_path))
        print("hi 2")
        files = tf.io.gfile.glob(self.glob_path)
        if len(files) == 0:
            raise ValueError('No file found')
        try:
            print("hi 3")
            num = reduce(lambda x, y: x + y,
                         map(lambda file: self._get_num_from_name(file), files))
            print(num)
        except Exception:
            raise ValueError(
                'Please format file name like <name>_<number>.<extension>')
        else:
            print("hi 4")
            tfrecords = list(
                filter(lambda file: file.endswith('.tfrecords'), files))
            txts = list(filter(lambda file: file.endswith('.txt'), files))
            if len(tfrecords) > 0:
                print(len(tfrecords))
                tfrecords_dataset = self._dataset_internal(
                    tfrecords, tf.data.TFRecordDataset, self.parse_tfrecord)
            if len(txts) > 0:
                print("hi 5")
                print(len(txts))
                txts_dataset = self._dataset_internal(txts,
                                                      tf.data.TextLineDataset,
                                                      self.parse_text)
            if len(tfrecords) > 0 and len(txts) > 0:
                print("hi 6")
                return tfrecords_dataset.concatenate(txts_dataset), num
            elif len(tfrecords) > 0:
                print("hi 7")
                return tfrecords_dataset, num
            elif len(txts) > 0:
                print("hi 8")
                print(num)
                return txts_dataset, num
