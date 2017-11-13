import urllib

num_training_steps = 5000
inception_model_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
inception_model_filename = 'inceptionV3.tgz'

'''
Download and initialize the inception model
'''
def initialize_model():
    urllib.urlretrieve(inception_model_url, inception_model_filename)

    return None

'''
Initialize a softmax layer to the inception model
'''
def add_final_layer(model):
    pass

'''
Evaluation step is used to evaluate the accuracy of the model
'''
def add_evaluation_step(model):
    pass

'''
Get arrays of training and testing images
'''
def get_images():
    pass

if __name__ == '__main__':
    model = initialize_model()
    add_final_layer(model)
    add_evaluation_step(model)

    images = get_images()

    for _ in xrange(num_training_steps):
        # train model
        # test model every i steps
        # save results to graph
        pass
