import os
import urllib
import tarfile

import tensorflow as tf

num_training_steps = 5000
inception_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
inception_dir_filepath = '/tmp/inception/'
inception_filepath = inception_dir_filepath + 'classify_image_graph_def.pb'

class NeuralNetwork(object):
    def __init__(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            with tf.gfile.FastGFile(inception_filepath, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

        self.session = tf.Session(graph=self.graph)


    '''
    Initialize a softmax layer to the inception model
    '''
    def add_final_layer(self):
        pass

    '''
    Evaluation step is used to evaluate the accuracy of the model
    '''
    def add_evaluation_step(self):
        pass

'''
Download and initialize the inception model
'''
def initialize_model():
    zipped_filepath = inception_dir_filepath + inception_url.split('/')[-1]

    urllib.urlretrieve(inception_url, zipped_filepath)
    if not os.path.exists(inception_dir_filepath):
        os.makedirs(inception_dir_filepath)
    tarfile.open(zipped_filepath, 'r:gz').extractall(inception_dir_filepath)

    return NeuralNetwork()

'''
Get arrays of training and testing images
'''
def get_images():
    pass

if __name__ == '__main__':
    model = initialize_model()

    images = get_images()

    for _ in xrange(num_training_steps):
        # train model
        # test model every i steps
        # save results to graph
        pass
