import os.path
import tensorflow as tf
import numpy as np

# FROM: http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/
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
def upsample_layer(in_layer, n_channels, name, upscale_factor):
    kernel_size = 2*upscale_factor - upscale_factor%2
    stride = upscale_factor
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        # Shape of the in_layer tensor
        in_batchSize = int(in_layer.get_shape()[0])
        in_w = int(in_layer.get_shape()[1])
        in_h = int(in_layer.get_shape()[2])
        #in_shape = tf.shape(in_layer)

        h = in_w * stride
        w = in_h * stride
        new_shape = [in_batchSize, h, w, n_channels]
        output_shape = tf.stack(new_shape)

        filter_shape = [kernel_size, kernel_size, n_channels, n_channels]

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


sess = tf.Session()
batch_size = 3
output_shape = [batch_size, 8, 8, 128]
strides = [1, 2, 2, 1]

l = tf.constant(0.1, shape=[batch_size, 5, 18, 4096])
w = tf.constant(0.1, shape=[7, 7, 128, 4])
num_classes = 2;
#upsample_ly7 = tf.layers.conv2d_transpose(conv_1x1, num_classes, kernel_size=4, strides=(2,2))
conv_out = tf.layers.conv2d(l, 2, 1, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
#h1 = tf.layers.conv2d_transpose(conv_out, 2, 4, 32, 'SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

scale_fac = 32
# kernel_size = 2*scale_fac - scale_fac%2
# in_shape = tf.shape(l)
#filter_shape = [kernel_size, kernel_size, num_classes, num_classes]
#weights_bilin_filter = get_bilinear_filter(filter_shape, scale_fac)
#h1 = tf.nn.conv2d_transpose(l, weights_bilin_filter, output_shape=[in_shape[0], in_shape[1] * scale_fac, in_shape[2] * scale_fac, num_classes], strides=[1, scale_fac, scale_fac, 1], padding='SAME')
# resize and upsample
h1 = reshape_and_upsample_layer(l, num_classes, scale_fac, 'l_out')
#print('shape: {}'.format(h1.get_shape()))

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

print(l.get_shape())
print(conv_out.get_shape())
print(h1.get_shape())
sess.run(h1)