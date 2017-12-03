import os
import urllib
import tarfile
import re
import random
import hashlib

import tensorflow as tf
from tensorflow.python.util import compat
import numpy as np

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1
num_training_steps = 800
eval_step_interval = 10
inception_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
inception_dir_filepath = './inception/'
inception_filepath = inception_dir_filepath + 'classify_image_graph_def.pb'


class NeuralNetwork(object):
    def __init__(self):
        tensor_name = 'pool_3/_reshape:0'
        resized_input_tensor_name = 'Mul:0'
        self.tensor_size = 2048
        self.train_batch_size = 100
        self.graph = tf.Graph()

        self.initialize_dataset(10, 10)

        with self.graph.as_default():
            with tf.gfile.FastGFile(inception_filepath, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                self.bottleneck_input, self.resized_input_tensor = tf.import_graph_def(graph_def, name='', return_elements=[tensor_name, resized_input_tensor_name])

        self.session = tf.Session(graph=self.graph)

        with self.session as sess:
            self.jpeg_data_tensor, self.decoded_image_tensor = self.add_jpeg_decoding(299, 299, 3, 128, 128)
            self.add_final_layer(len(self.images_dict.keys()))
            self.add_evaluation_step()

            init = tf.global_variables_initializer()
            sess.run(init)

            self.train_network(num_training_steps)

    '''
    Get arrays of training and testing images
    '''
    def initialize_dataset(self, testing_percentage, validation_percentage):
        self.images_dict = {}
        artist_dirs = [x[0] for i, x in enumerate(os.walk('./dataset')) if i != 0]

        for artist_dir in artist_dirs:
            artist_name = os.path.basename(artist_dir)
            artist_images = filter(lambda f: f.split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'JPEG'], os.listdir(artist_dir))

            validation_images, testing_images, training_images = [], [], []

            for artist_image in artist_images:
                base_name = os.path.basename(artist_image)
                hash_name = re.sub(r'_nohash_.*$', '', base_name)
                hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
                percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_IMAGES_PER_CLASS + 1)) * (100.0 / MAX_NUM_IMAGES_PER_CLASS))

                if percentage_hash < validation_percentage:
                    validation_images.append(base_name)
                elif percentage_hash < (testing_percentage + validation_percentage):
                    testing_images.append(base_name)
                else:
                    training_images.append(base_name)

            self.images_dict[artist_name] = {
                'training': training_images,
                'testing': testing_images,
                'validation': validation_images
            }

    def add_jpeg_decoding(self, input_width, input_height, input_depth, input_mean, input_std):
        jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
        decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
        decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
        decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
        resize_shape = tf.stack([input_height, input_width])
        resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
        resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                               resize_shape_as_int)
        offset_image = tf.subtract(resized_image, input_mean)
        mul_image = tf.multiply(offset_image, 1.0 / input_std)

        return jpeg_data, mul_image

    '''
    Initialize a softmax layer to the inception model
    '''
    def add_final_layer(self, class_count):
        self.bottleneck_tensor = tf.placeholder_with_default(self.bottleneck_input, shape=[None, self.tensor_size], name='InputPlaceholder')
        self.ground_truth_input = tf.placeholder(tf.float32, [None, class_count], name='GroundTruthInput')

        layer_weights = tf.Variable(tf.truncated_normal([self.tensor_size, class_count], stddev=0.001), name='final_weights')
        layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')

        logits = tf.add(tf.matmul(self.bottleneck_tensor, layer_weights), layer_biases)

        self.final_tensor = tf.nn.softmax(logits, name='final_result')

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.ground_truth_input, logits=logits)
            with tf.name_scope('total'):
                cross_entropy_mean = tf.reduce_mean(cross_entropy)

        tf.summary.scalar('cross_entropy', cross_entropy_mean)

        with tf.name_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            self.train_step = optimizer.minimize(cross_entropy_mean)

    '''
    Evaluation step is used to evaluate the accuracy of the model
    '''
    def add_evaluation_step(self):
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                self.prediction = tf.argmax(self.final_tensor, 1)
                correct_prediction = tf.equal(self.prediction, tf.argmax(self.ground_truth_input, 1))
            self.evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    '''

    '''
    def train_network(self, num_training_steps):
        print("Starting Training")

        for i in range(num_training_steps):
            train_bottlenecks, train_ground_truth = self.get_tensors_for_training('training')

            _ = self.session.run([self.train_step], feed_dict={self.bottleneck_tensor: train_bottlenecks, self.ground_truth_input: train_ground_truth})

            if (i % eval_step_interval) == 0:
                train_accuracy = self.session.run([self.evaluation_step], feed_dict={self.bottleneck_tensor: train_bottlenecks, self.ground_truth_input: train_ground_truth})[0]
                print('Step %d: Train accuracy = %.1f%%' % (i, train_accuracy * 100))

        test_bottlenecks, test_ground_truth = self.get_tensors_for_testing('testing')
        test_accuracy, predictions = self.session.run([self.evaluation_step, self.prediction], feed_dict={self.bottleneck_tensor: test_bottlenecks, self.ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))


    def get_tensors_for_training(self, category):
        class_count = len(self.images_dict.keys())
        tensor_inputs, ground_truth_inputs, filenames = [], [], []

        for unused_i in range(self.train_batch_size):
            label_index = random.randrange(class_count)
            label_name = list(self.images_dict.keys())[label_index]

            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_path = self.get_image_path(label_name, image_index, category)
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            tensor_input = self.run_bottleneck_on_image(image_data)

            ground_truth_input = np.zeros(class_count, dtype=np.float32)
            ground_truth_input[label_index] = 1.0

            tensor_inputs.append(tensor_input)
            ground_truth_inputs.append(ground_truth_input)

        return tensor_inputs, ground_truth_inputs

    def get_tensors_for_testing(self, category):
        class_count = len(self.images_dict.keys())
        bottlenecks = []
        ground_truths = []
        for label_index, label_name in enumerate(self.images_dict.keys()):
            for image_index, image_name in enumerate(self.images_dict[label_name][category]):
                image_path = self.get_image_path(label_name, image_index, category)
                image_data = tf.gfile.FastGFile(image_path, 'rb').read()
                bottleneck = self.run_bottleneck_on_image(image_data)

                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0

                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)

        return bottlenecks, ground_truths


    def get_image_path(self, label_name, index, category):
        label_lists = self.images_dict[label_name]
        category_list = label_lists[category]
        mod_index = index % len(category_list)
        base_name = category_list[mod_index]
        full_path = os.path.join('./dataset', label_name, base_name)
        return full_path

    def run_bottleneck_on_image(self, image_data):
        # First decode the JPEG image, resize it, and rescale the pixel values.
        resized_input_values = self.session.run(self.decoded_image_tensor, {self.jpeg_data_tensor: image_data})
        # Then run it through the recognition network.
        bottleneck_values = self.session.run(self.bottleneck_tensor, {self.resized_input_tensor: resized_input_values})
        bottleneck_values = np.squeeze(bottleneck_values)
        return bottleneck_values
'''
Download and initialize the inception model
'''
def initialize_model():
    urllib.request.urlretrieve(inception_url, inception_url.split('/')[-1])
    if not os.path.exists(inception_dir_filepath):
        os.makedirs(inception_dir_filepath)
    tarfile.open(inception_url.split('/')[-1], 'r:gz').extractall(inception_dir_filepath)
    return NeuralNetwork()

if __name__ == '__main__':
    model = initialize_model()
