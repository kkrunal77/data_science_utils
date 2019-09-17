
# See available datasets
# print(tfds.list_builders())
import numpy as np
import os
import tensorflow_datasets as tfds
import tensorflow as tf

"""https://www.tensorflow.org/datasets"""
class datareader(object):
    def __init__(self):
        super(datareader, self).__init__()
        tf.enable_eager_execution()

    def getdata(self, data_set_name, batch_size=False, train_split=True):
        super(datareader, self).__init__()
        self.data_set_name = data_set_name
        self.batch_size = batch_size
        self.train_split = train_split
        if self.batch_size and self.train_split:
            """exp. [mnist, cifar10,...]"""
            dataset = tfds.load(name=data_set_name.lower(), split=tfds.Split.TRAIN)
            dataset = dataset.shuffle(1024).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            return  dataset
        else:
            dataset = tfds.load(name=data_set_name.lower(), split=tfds.Split.TRAIN)
            return dataset


import tensorflow_datasets as tfds


import logging
logger = logging.getLogger("TFRecord Helpers")
logger.setLevel(logging.INFO)

def byte_to_tf_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_to_tf_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int64_to_tf_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def build_serializer_for_examples(feature_names,dtypes_converter):
    def serialize_example(*args):
        feature = {feature:dtypes_converter[feature](args[i]) for i,feature in enumerate(feature_names)}
        proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return proto.SerializeToString()
    return serialize_example

def build_tf_serializer_examples(feature_names,dtypes_converter):
    serializer = build_serializer_for_examples(feature_names,dtypes_converter)
    def tf_serialize_example(*args):
        tf_string = tf.py_function(serializer,args,tf.string)
        return tf.reshape(tf_string, ())
    return tf_serialize_example

def store_numpy_arrays_as_tfrecord_examples(numpy_arrays,filename,feature_names,dtypes_converter, compression_type=None):
    assert type(numpy_arrays) == tuple or type(numpy_arrays) == list
    ds = tf.data.Dataset.from_tensor_slices(numpy_arrays)
    store_dataset_as_tfrecord_examples(ds,filename,feature_names,dtypes_converter, compression_type=compression_type)

def store_dataset_as_tfrecord_examples(dataset,filename,feature_names,dtypes_converter, compression_type=None):
    dataset = dataset.map(build_tf_serializer_examples(feature_names,dtypes_converter))
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type=compression_type)
    logger.debug("Storing to: %s",filename)
    writer.write(dataset)


def read_tfrecord_as_dataset_examples(filename, feature_description, batch_size,
                        shuffle=True, shuffle_buffer_size=10000, compression_type=None):
    ds = tf.data.TFRecordDataset(filename,compression_type=compression_type)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    parser = lambda x: tf.io.parse_example(x, feature_description)
    ds = ds.map(parser)
    return ds


def store_numpy_arrays_as_tfrecord(numpy_arrays, filename, serializer, compression_type=None):
    assert type(numpy_arrays) == tuple or type(numpy_arrays) == list
    ds = tf.data.Dataset.from_tensor_slices(numpy_arrays)
    store_dataset_as_tfrecord(ds, filename, serializer, compression_type=compression_type)


def store_dataset_as_tfrecord(dataset, filename, serializer, compression_type=None):
    dataset = dataset.map(serializer)
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type=compression_type)
    logger.debug("Storing to: %s", filename)
    writer.write(dataset)


def read_tfrecord_as_dataset(filename, deserializer, batch_size,
                             shuffle=True, shuffle_buffer_size=10000, compression_type=None):
    ds = tf.data.TFRecordDataset(filename, compression_type=compression_type)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(deserializer)
    ds = ds.batch(batch_size)
    return ds


def get_cifar10(dir, batch_size, shuffle=True, shuffle_buffer_size=10000, compression_type=None):
    # check if cifar10 tfrecord exists locally, just check file exists
    # if it does then get it from tfrecord
    # else download and store the tfrecord and then get it from tfreccord
    if not os.path.exists(dir):
        os.makedirs(dir)
    train_loc = os.path.join(dir, "cifar10.train.tfrecords")
    test_loc = os.path.join(dir, "cifar10.test.tfrecords")
    cifar10_exists = os.path.exists(train_loc) and os.path.exists(test_loc)

    def serializer(x,y):
        x = tf.io.serialize_tensor(x)
        y = tf.io.serialize_tensor(tf.cast(y, tf.int64))
        z = tf.convert_to_tensor((x,y))
        z = tf.io.serialize_tensor(z)
        return z

    def deserializer(x):
        x = tf.io.parse_tensor(x, out_type=tf.string)
        img = x[0]
        label = x[1]
        img = tf.io.parse_tensor(img, out_type=tf.uint8)
        img = tf.cast(img,tf.float32)
        label = tf.io.parse_tensor(label, out_type=tf.int64)
        return img, label

    if not cifar10_exists:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        store_numpy_arrays_as_tfrecord((x_train, y_train), train_loc, serializer, compression_type=compression_type)
        store_numpy_arrays_as_tfrecord((x_test, y_test), test_loc, serializer, compression_type=compression_type)

    train = read_tfrecord_as_dataset(train_loc, deserializer, batch_size, shuffle, shuffle_buffer_size,
                                     compression_type)
    test = read_tfrecord_as_dataset(test_loc, deserializer, batch_size, shuffle, shuffle_buffer_size, compression_type)

    train = train.prefetch(tf.data.experimental.AUTOTUNE)
    test = test.prefetch(tf.data.experimental.AUTOTUNE)
    return train, test


def get_cifar10_examples(dir, batch_size, shuffle=True, shuffle_buffer_size=10000, compression_type=None):
    # check if cifar10 tfrecord exists locally, just check file exists
    # if it does then get it from tfrecord
    # else download and store the tfrecord and then get it from tfreccord
    if not os.path.exists(dir):
        os.makedirs(dir)
    train_loc = os.path.join(dir,"cifar10.train.examples.tfrecords")
    test_loc = os.path.join(dir, "cifar10.test.examples.tfrecords")
    cifar10_exists = os.path.exists(train_loc) and os.path.exists(test_loc)

    if not cifar10_exists:


        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        feature_names = ["image", "label"]
        dtypes_converter = {"image":lambda img:byte_to_tf_feature(bytes(img)),"label":lambda label:int64_to_tf_feature(label)}


        store_numpy_arrays_as_tfrecord_examples((x_train, y_train), train_loc, feature_names=feature_names,
                                  dtypes_converter=dtypes_converter, compression_type=compression_type)
        store_numpy_arrays_as_tfrecord_examples((x_test, y_test), test_loc, feature_names=feature_names,
                                  dtypes_converter=dtypes_converter, compression_type=compression_type)

    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64)
    }
    train = read_tfrecord_as_dataset_examples(train_loc,feature_description, batch_size, shuffle, shuffle_buffer_size, compression_type)
    test = read_tfrecord_as_dataset_examples(test_loc,feature_description, batch_size, shuffle, shuffle_buffer_size, compression_type)

    def parser(x):
        labels = tf.map_fn(lambda y: tf.cast(y, tf.int64), x['label'])
        imgs = tf.map_fn(lambda y:tf.cast(tf.io.decode_raw(y, tf.uint8),tf.float32), x['image'], dtype=tf.float32)
        imgs = tf.map_fn(lambda im: tf.reshape(im, shape=[32, 32, 3]),imgs, dtype=tf.float32)
        return imgs,labels

    train = train.map(parser).prefetch(tf.data.experimental.AUTOTUNE)
    test = test.map(parser).prefetch(tf.data.experimental.AUTOTUNE)
    
    return train,test
