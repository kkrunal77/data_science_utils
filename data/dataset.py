import tensorflow as tf

# See available datasets
# print(tfds.list_builders())
import tensorflow_dataset as tfds

"""https://www.tensorflow.org/datasets"""
class datareader(object):
    def __init__(self):
        super(datareader, self).__init__()
        tf.enable_eager_execution()

    def getdata(self , data_set_name, batch_size=False, train_split=True):
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