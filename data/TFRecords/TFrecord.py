import tensorflow as tf
import numpy as np

def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_function(x, y):
	example = tf.train.Example(features=tf.train.Features(
					feature={
							'image': _bytes_feature(tf.compat.as_bytes(x.tostring())),
							'label': _int64_feature(int(y))
					}))
	return example.SerializeToString()


def convert_to_tfrecord(data_set, file_name):
	with tf.python_io.TFRecordWriter(file_name) as record_writer:
		for x, y in data_set:
			if isinstance(x, (np.ndarray, np.generic)) and isinstance(y, (np.ndarray, np.generic)):
				record = serialize_function(x, y)
			else:
				record = serialize_function(x.numpy(), y.numpy())
			record_writer.write(record)
		print(f"TFRecord id created at path : '{file_name}', Done")


def parser(record):
	feature_description = {
	'image': tf.FixedLenFeature([], tf.string, default_value=''),
	'label': tf.FixedLenFeature([], tf.int64, default_value=0)
	# 'feature2': tf.FixedLenFeature([], tf.string, default_value=''),
	# 'feature3': tf.FixedLenFeature([], tf.float32, default_value=0.0),
		}
	parsed = tf.parse_single_example(record, feature_description)
	image = tf.decode_raw(parsed["image"], tf.uint8)
	image = tf.cast(image, tf.float32)
	image = tf.reshape(image, shape=[32, 32, 3])
	label = tf.cast(parsed["label"], tf.int32)
	return {'image': image}, label


def input_foo(filenames):
	dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=40)

	dataset = dataset.apply(
		tf.contrib.data.shuffle_and_repeat(buffer_size=1024,
										   seed=1)
		)
	dataset = dataset.apply(
		tf.contrib.data.map_and_batch(
								  map_func=parser,
								  batch_size=32,
								  num_parallel_calls=tf.data.experimental.AUTOTUNE)
		)
	dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/GPU:0', buffer_size=None))
#     dataset = dataset.map(parser, num_parallel_calls=12)
#     dataset = dataset.batch(batch_size=1000)
#     dataset = dataset.prefetch(buffer_size=2)
	return dataset

class CreateTFRecord(object):
    """docstring for CreateTFRecord"""
    '''
    EXP.
    data_to_write = zip(x_train, y_train)
    file name = 'train.tfrecord'
    '''
    def __init__(self, data_to_write, file_name):
        super().__init__()
        self.data_to_write = data_to_write
        self.file_name = file_name
        convert_to_tfrecord(self.data_to_write, self.file_name)

# CreateTFRecord("kk.tfrecord", "kk")