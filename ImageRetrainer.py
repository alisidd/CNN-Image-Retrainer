import os
import urllib
import tarfile

import tensorflow as tf

num_training_steps = 5000
inception_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
inception_dir_filepath = './inception/'
inception_filepath = inception_dir_filepath + 'classify_image_graph_def.pb'


class NeuralNetwork(object):
    def __init__(self):
        self.tensor_name = 'pool_3/_reshape:0'
        self.tensor_size = 2048
        self.graph = tf.Graph()

        with self.graph.as_default():
            with tf.gfile.FastGFile(inception_filepath, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                self.tensor = tf.import_graph_def(graph_def, name='', return_elements=[self.tensor_name])[0]

        self.session = tf.Session(graph=self.graph)

        with self.session as sess:
            self.add_final_layer(20)
            self.add_evaluation_step()


    '''
    Initialize a softmax layer to the inception model
    '''
    def add_final_layer(self, class_count):
        tensor_input = tf.placeholder_with_default(self.tensor, shape=[1, self.tensor_size], name='InputPlaceholder')
        self.ground_truth_input = tf.placeholder(tf.float32, [None, class_count], name='GroundTruthInput')

        layer_weights = tf.Variable(tf.truncated_normal([self.tensor_size, class_count], stddev=0.001), name='final_weights')
        layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')

        logits = tf.add(tf.matmul(tensor_input, layer_weights), layer_biases)

        self.final_tensor = tf.nn.softmax(logits, name='final_result')


    '''
    Evaluation step is used to evaluate the accuracy of the model
    '''
    def add_evaluation_step(self):
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                prediction = tf.argmax(self.final_tensor, 1)
                correct_prediction = tf.equal(prediction, tf.argmax(self.ground_truth_input, 1))
        with tf.name_scope('accuracy'):
            self.evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
Download and initialize the inception model
'''
def initialize_model():
    urllib.urlretrieve(inception_url, inception_url.split('/')[-1])
    if not os.path.exists(inception_dir_filepath):
        os.makedirs(inception_dir_filepath)
    tarfile.open(inception_url.split('/')[-1], 'r:gz').extractall(inception_dir_filepath)

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
