import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def get_bilinear_filter(filter_shape, upscale_factor):
    ##filter_shape is [width, height, num_in_channels, num_out_channels]
    kernel_size = filter_shape[1]
    ### Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            ##Interpolation Calculation
            value = (1 - abs((x - centre_location)/ upscale_factor)) * (1 - abs((y - centre_location)/ upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights, dtype=tf.float32)

    bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init, shape=weights.shape)
    return bilinear_weights


# FROM: http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/
def upsample_layer(in_layer, num_classes, name, upscale_factor):
    kernel_size = 2*upscale_factor - upscale_factor%2
    stride = upscale_factor
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_shape = tf.shape(in_layer)
        in_h = in_shape[1]
        in_w = in_shape[2]
        in_batchSize = in_shape[0]
        # Shape of the in_layer tensor
        # in_batchSize = int(in_layer.get_shape()[0])
        # in_w = int(in_layer.get_shape()[1])
        # in_h = int(in_layer.get_shape()[2])
        #in_shape = tf.shape(in_layer)

        h = in_h * stride
        w = in_w * stride
        new_shape = [in_batchSize, h, w, num_classes]
        output_shape = tf.stack(new_shape)

        filter_shape = [kernel_size, kernel_size, num_classes, num_classes]

        weights = get_bilinear_filter(filter_shape,upscale_factor)
        deconv = tf.nn.conv2d_transpose(in_layer, weights, output_shape, strides=strides, padding='SAME')
    return deconv

def reshape_and_upsample_layer(in_layer, num_classes, scale_fac, name):
    in_classes = int(in_layer.get_shape()[3])
    convW = tf.Variable(tf.truncated_normal(shape=[1,1,in_classes,num_classes], stddev=0.1), name=name+'_weights')
    convB = tf.Variable(tf.truncated_normal(shape=[num_classes], stddev=0.1), name=name+'_bias')
    conv_reshape = tf.nn.conv2d(in_layer, convW, strides = [1,1,1,1], padding='VALID', name=name+'_resized')
    conv_reshape = tf.nn.bias_add(conv_reshape, convB, name=name+'_resized_addBias')

    conv_upsampled = upsample_layer(conv_reshape, num_classes, name+'_upsampled', scale_fac)

    return conv_upsampled

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    vgg_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    # to visualize/check graph of provided VGG16 model
    #file_writer = tf.summary.FileWriter('data/vgg', sess.graph)
        
    return vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out
#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    scale_fac = 32
    upsample_ly7 = reshape_and_upsample_layer(vgg_layer7_out, num_classes, scale_fac, 'ly7_upsample')
    
    scale_fac = 16
    upsample_ly4 = reshape_and_upsample_layer(vgg_layer4_out, num_classes, scale_fac, 'ly4_upsample')
    
    scale_fac = 8
    upsample_ly3 = reshape_and_upsample_layer(vgg_layer3_out, num_classes, scale_fac, 'ly3_upsample')

    # add layers according to paper for FCN8 (fig. 3)
    skip_conn = tf.add(upsample_ly3, tf.add(2*upsample_ly4, 4*upsample_ly7))

    # print('layer 7 upsample shape: {}'.format(upsample_ly7.get_shape()))
    # print('layer 4 upsample shape: {}'.format(upsample_ly4.get_shape()))
    # print('layer 3 upsample shape: {}'.format(upsample_ly3.get_shape()))

    return skip_conn
#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    return None, None, None
#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    pass
#tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        #- subtract mean pixel value
        #- random transform

        # TODO: Build NN using load_vgg, layers, and optimize function
        vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, 'data/vgg')
        last_out = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        '''
        # TEST
        image, labels = next(get_batches_fn(1))
        print('img shape: {}'.format(image.shape))
        sess.run(tf.global_variables_initializer())
        debug = sess.run(last_out, feed_dict={vgg_input: image, vgg_keep_prob: 1.0})
        print('########################################')
        print('########################################')
        print('########################################')
        print('FINAL layer 7: {}'.format(debug.shape))
        '''
        #file_writer = tf.summary.FileWriter('data/vgg', sess.graph)

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
