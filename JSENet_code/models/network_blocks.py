#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Definition of networks models with blocks
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import sys
import numpy as np
import tensorflow as tf
import kernels.convolution_ops as conv_ops


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utilities
#       \***************/
#

def weight_variable(shape):
    # tf.set_random_seed(42)
    initial = tf.truncated_normal(shape, stddev=np.sqrt(2 / shape[-1]))
    initial = tf.round(initial * tf.constant(1000, dtype=tf.float32)) / tf.constant(1000, dtype=tf.float32)
    return tf.Variable(initial, name='weights')


def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial, name='bias')


def ind_max_pool(x, inds):
    """
    This tensorflow operation compute a maxpooling according to the list of indices 'inds'.
    > x = [n1, d] features matrix
    > inds = [n2, max_num] each row of this tensor is a list of indices of features to be pooled together
    >> output = [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = tf.concat([x, tf.reduce_min(x, axis=0, keep_dims=True)], axis=0)

    # Get features for each pooling cell [n2, max_num, d]
    pool_features = tf.gather(x, inds, axis=0)

    # Pool the maximum
    return tf.reduce_max(pool_features, axis=1)


def closest_pool(x, inds):
    """
    This tensorflow operation compute a pooling according to the list of indices 'inds'.
    > x = [n1, d] features matrix
    > inds = [n2, max_num] We only use the first column of this which should be the closest points too pooled positions
    >> output = [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = tf.concat([x, tf.zeros((1, int(x.shape[1])), x.dtype)], axis=0)

    # Get features for each pooling cell [n2, d]
    pool_features = tf.gather(x, inds[:, 0], axis=0)

    return pool_features


def KPConv(query_points, support_points, neighbors_indices, features, K_values, radius, config):
    """
    Returns the output features of a KPConv
    """

    # Get KP extent from current radius and config density
    extent = config.KP_extent * radius / config.density_parameter

    # Convolution
    return conv_ops.KPConv(query_points,
                           support_points,
                           neighbors_indices,
                           features,
                           K_values,
                           fixed=config.fixed_kernel_points,
                           KP_extent=extent,
                           KP_influence=config.KP_influence,
                           aggregation_mode=config.convolution_mode,)


def batch_norm(x, use_batch_norm=True, momentum=0.99, training=True):
    """
    This tensorflow operation compute a batch normalization.
    > x = [n1, d] features matrix
    >> output = [n1, d] normalized, scaled, offset features matrix
    """

    if use_batch_norm:
        return tf.layers.batch_normalization(x,
                                             momentum=momentum,
                                             epsilon=1e-6,
                                             training=training)

    else:
        # Just add biases
        beta = tf.Variable(tf.zeros([x.shape[-1]]), name='offset')
        return x + beta


def leaky_relu(features, alpha=0.2):
    return tf.nn.leaky_relu(features, alpha=alpha, name=None)

# ----------------------------------------------------------------------------------------------------------------------
#
#           Convolution blocks
#       \************************/
#


def unary_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a simple 1x1 convolution
    """

    w = weight_variable([int(features.shape[1]), fdim])
    x = conv_ops.unary_convolution(features, w)
    x = leaky_relu(batch_norm(x,
                              config.use_batch_norm,
                              config.batch_norm_momentum,
                              training))

    return x


def simple_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a simple convolution
    """

    # Weights
    w = weight_variable([config.num_kernel_points, int(features.shape[1]), fdim])

    # Convolution
    x = KPConv(inputs['points'][layer_ind],
               inputs['points'][layer_ind],
               inputs['neighbors'][layer_ind],
               features,
               w,
               radius,
               config)

    x = leaky_relu(batch_norm(x,
                              config.use_batch_norm,
                              config.batch_norm_momentum,
                              training))

    return x


def resnetb_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a resnet bottleneck convolution (1conv > KPconv > 1conv + shortcut)
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([int(features.shape[1]), fdim // 2])
        x = conv_ops.unary_convolution(features, w)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim // 2])
        x = KPConv(inputs['points'][layer_ind],
                   inputs['points'][layer_ind],
                   inputs['neighbors'][layer_ind],
                   x,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):
        if int(features.shape[1]) != 2 * fdim:
            w = weight_variable([int(features.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(features, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)
        else:
            shortcut = features

    return leaky_relu(x + shortcut)


def resnetb_strided_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a strided resnet bottleneck convolution (shortcut is a maxpooling)
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([int(features.shape[1]), fdim // 2])
        x = conv_ops.unary_convolution(features, w)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim // 2])
        x = KPConv(inputs['points'][layer_ind + 1],
                   inputs['points'][layer_ind],
                   inputs['pools'][layer_ind],
                   x,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):

        # Pool shortcuts to strided points TODO: max_pool or closest_pool ?
        shortcut = ind_max_pool(features, inputs['pools'][layer_ind])
        # shortcut = closest_pool(features, neighbors_indices)

        # Regular upsample of the features if not the same dimension
        if int(shortcut.shape[1]) != 2 * fdim:
            w = weight_variable([int(shortcut.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(shortcut, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)

    return leaky_relu(x + shortcut)


def simple_upsample_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a simple upsampling convolution
    """

    # Weights
    w = weight_variable([config.num_kernel_points, int(features.shape[1]), fdim])

    # Convolution
    x = KPConv(inputs['points'][layer_ind - 1],
               inputs['points'][layer_ind],
               inputs['upsamples'][layer_ind - 1],
               features,
               w,
               radius,
               config)

    x = leaky_relu(batch_norm(x,
                              config.use_batch_norm,
                              config.batch_norm_momentum,
                              training))

    return x


def nearest_upsample_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing an upsampling by nearest interpolation
    """

    with tf.variable_scope('nearest_upsample'):
        upsampled_features = closest_pool(features, inputs['upsamples'][layer_ind - 1])

    return upsampled_features


def get_block_ops(block_name):

    if block_name == 'unary':
        return unary_block

    if block_name == 'simple':
        return simple_block

    elif block_name == 'resnetb':
        return resnetb_block

    elif block_name == 'resnetb_strided':
        return resnetb_strided_block

    elif block_name == 'nearest_upsample':
        return nearest_upsample_block

    elif block_name == 'simple_upsample':
        return simple_upsample_block

    else:
        raise ValueError('Unknown block name in the architecture definition : ' + block_name)




# ----------------------------------------------------------------------------------------------------------------------
#
#           Architectures
#       \*******************/
#


def assemble_CNN_blocks(inputs, config, dropout_prob):
    """
    Definition of all the encoder layers according to config
    :param inputs: dictionary of inputs with keys [points, neighbors, pools, features, batches, labels, boundaries] flat inputs(not B X N X C)((N1 + N2 + ...) X C)
    :param config:
    :param dropout_prob:
    :return:
    """

    # Current radius of convolution and feature dimension
    r = config.first_subsampling_dl * config.density_parameter
    layer = 0
    fdim = config.first_features_dim

    # Input features
    features = inputs['features']
    F = []

    # Boolean of training
    training = dropout_prob < 0.99

    # Loop over consecutive blocks
    block_in_layer = 0
    for block_i, block in enumerate(config.architecture):

        # Detect change to next layer
        if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):

            # Save this layer features
            F += [features]

        # Detect upsampling block to stop
        if 'upsample' in block:
            break

        with tf.variable_scope('layer_{:d}/{:s}_{:d}'.format(layer, block, block_in_layer)):

            # Get the function for this layer
            block_ops = get_block_ops(block)

            # Apply the layer function defining tf ops
            features = block_ops(layer,
                                 inputs,
                                 features,
                                 r,
                                 fdim,
                                 config,
                                 training)

        # Index of block in this layer
        block_in_layer += 1

        # Detect change to a subsampled layer
        if 'pool' in block or 'strided' in block:

            # Update radius and feature dimension for next layer
            layer += 1
            r *= 2
            fdim *= 2
            block_in_layer = 0

        # Save feature vector after global pooling
        if 'global' in block:
            # Save this layer features
            F += [features]

    return F




def assemble_DCNN_blocks_region(extracted_features, inputs, config, dropout_prob):
    """
    Definition of decoder layers of semantic segmentation branch according to config
    :param inputs: dictionary of inputs with keys [points, neighbors, pools, features, batches, labels, boundaries] flat inputs(not B X N X C)((N1 + N2 + ...) X C)
    :param config:
    :param dropout_prob:
    :return:
    """
    # features from encoder
    features = extracted_features[-1]

    # Current radius of convolution and feature dimension
    layer = config.num_layers - 1
    r = config.first_subsampling_dl * config.density_parameter * 2 ** layer
    fdim = config.first_features_dim * 2 ** layer # if you use resnet, fdim is actually 2 times that

    # Boolean of training
    training = dropout_prob < 0.99

    # Find first upsampling block
    start_i = 0
    for block_i, block in enumerate(config.architecture):
        if 'upsample' in block:
            start_i = block_i
            break

    # Loop over upsampling blocks
    block_in_layer = 0
    for block_i, block in enumerate(config.architecture[start_i:]):

        with tf.variable_scope('uplayerregion_{:d}/{:s}_{:d}'.format(layer, block, block_in_layer)):

            # Get the function for this layer
            block_ops = get_block_ops(block)

            # Apply the layer function defining tf ops
            features = block_ops(layer,
                                 inputs,
                                 features,
                                 r,
                                 fdim,
                                 config,
                                 training)

        # Index of block in this layer
        block_in_layer += 1

        # Detect change to a subsampled layer
        if 'upsample' in block:

            # Update radius and feature dimension for next layer
            layer -= 1
            r *= 0.5
            fdim = fdim // 2
            block_in_layer = 0

            # Concatenate with CNN feature map
            features = tf.concat((features, extracted_features[layer]), axis=1)

    return features


def assemble_DCNN_blocks_edge(extracted_features, inputs, config, dropout_prob):
    """
    Definition of decoder layers of semantic edge detection branch according to config
    :param inputs: dictionary of inputs with keys [points, neighbors, pools, features, batches, labels, boundaries] flat inputs(not B X N X C)((N1 + N2 + ...) X C)
    :param config:
    :param dropout_prob:
    :return:
    """
    # features from CNN
    features = extracted_features[-1]

    # Current radius of convolution and feature dimension
    layer = config.num_layers - 1
    r = config.first_subsampling_dl * config.density_parameter * 2 ** layer
    fdim = config.first_features_dim * 2 ** layer # if you use resnet, fdim is actually 2 times that

    # Boolean of training
    training = dropout_prob < 0.99

    # Find first upsampling block
    start_i = 0
    for block_i, block in enumerate(config.architecture):
        if 'upsample' in block:
            start_i = block_i
            break

    # Loop over upsampling blocks
    block_in_layer = 0
    for block_i, block in enumerate(config.architecture[start_i:]):

        with tf.variable_scope('uplayeredge_{:d}/{:s}_{:d}'.format(layer, block, block_in_layer)):

            # Get the function for this layer
            block_ops = get_block_ops(block)

            # Apply the layer function defining tf ops
            features = block_ops(layer,
                                 inputs,
                                 features,
                                 r,
                                 fdim,
                                 config,
                                 training)

        # Index of block in this layer
        block_in_layer += 1

        # Detect change to a subsampled layer
        if 'upsample' in block:

            # Update radius and feature dimension for next layer
            layer -= 1
            r *= 0.5
            fdim = fdim // 2
            block_in_layer = 0

            # Concatenate with CNN feature map
            features = tf.concat((features, extracted_features[layer]), axis=1)

    return features


def feature_fusion(inputs, features, config, training):
    """
    Feature fusion sub-module
    """
    # Conv radius
    r0 = config.first_subsampling_dl * config.density_parameter
    r1 = r0 * 2
    r2 = r1 * 2
    r3 = r2 * 2
    r4 = r3 * 2

    # Light-weight U-net
    refine_in0 = unary_block(0, inputs, features, r0, 32, config, training)

    with tf.variable_scope('resb_0'):
        refine_in1 = resnetb_block(0, inputs, refine_in0, r0, 16, config, training)
    with tf.variable_scope('resb_strided_0'):
        refine_in2 = resnetb_strided_block(0, inputs, refine_in1, r0, 16, config, training)
    
    with tf.variable_scope('resb_1'):
        refine_in3 = resnetb_block(1, inputs, refine_in2, r1, 16, config, training)
    with tf.variable_scope('resb_strided_1'):
        refine_in4 = resnetb_strided_block(1, inputs, refine_in3, r1, 16, config, training)
    
    with tf.variable_scope('resb_2'):
        refine_in5 = resnetb_block(2, inputs, refine_in4, r2, 16, config, training)
    with tf.variable_scope('resb_strided_2'):
        refine_in6 = resnetb_strided_block(2, inputs, refine_in5, r2, 16, config, training)
    
    with tf.variable_scope('resb_3'):
        refine_in7 = resnetb_block(3, inputs, refine_in6, r3, 16, config, training)
    with tf.variable_scope('resb_strided_3'):
        refine_in8 = resnetb_strided_block(3, inputs, refine_in7, r3, 16, config, training)
    
    with tf.variable_scope('resb_4'):
        refine_in9 = resnetb_block(4, inputs, refine_in8, r4, 16, config, training)
    
    refine_up0 = nearest_upsample_block(4, inputs, refine_in9, r4, -1, config, training)
    refine_up1 = tf.concat((refine_up0, refine_in7), axis=1)
    refine_up2 = unary_block(-1, inputs, refine_up1, -1, 32, config, training)
    
    refine_up3 = nearest_upsample_block(3, inputs, refine_up2, r3, -1, config, training)
    refine_up4 = tf.concat((refine_up3, refine_in5), axis=1)
    refine_up5 = unary_block(-1, inputs, refine_up4, -1, 32, config, training)
    
    refine_up6 = nearest_upsample_block(2, inputs, refine_up5, r2, -1, config, training)
    refine_up7 = tf.concat((refine_up6, refine_in3), axis=1)
    refine_up8 = unary_block(-1, inputs, refine_up7, -1, 32, config, training)

    refine_up9 = nearest_upsample_block(1, inputs, refine_up8, r1, -1, config, training)
    refine_up10 = tf.concat((refine_up9, refine_in1), axis=1)
    refine_up11 = unary_block(-1, inputs, refine_up10, -1, 32, config, training)

    refine_feature = tf.concat((refine_up11, refine_in0), axis=1)
    
    return refine_feature


def edge_generation(inputs, logits_region):
    """
    Edge map generation sub-module
    :param inputs: flat_inputs containing neighbor information
    :param logits_region: semantic segmentation output without softmax normalization
    """
    max_ind = tf.math.reduce_max(inputs['in_batches'])
    num_classes = tf.shape(logits_region)[1]
    
    # region_softmax: [num_points, num_class] (0 ~ 1)
    region_softmax = tf.nn.softmax(logits_region)

    # Add a last row with zero activation for shadow filtering [num_points + 1, num_class]
    region_softmax_shadow = tf.concat([region_softmax, tf.zeros((1, num_classes), dtype=tf.float32)], axis=0)

    # Get activations for each point [num_points, num_neighbor(shadow neighbor included), num_class]
    neighbor_activations = tf.gather(region_softmax_shadow, inputs['convert_neighbors'], axis=0)
    
    # neighbor_sums [num_points, num_class]
    neighbor_sums = tf.reduce_sum(neighbor_activations, axis=1)
    
    # number of real neighbors [num_points, 1]
    num_real_neighbors = tf.math.less(inputs['convert_neighbors'], max_ind)
    num_real_neighbors = tf.dtypes.cast(num_real_neighbors, tf.float32)
    num_real_neighbors = tf.reduce_sum(num_real_neighbors, axis=1, keepdims=True)

    # neighbor_avgs [num_points, num_class]
    neighbor_avgs = tf.math.divide(neighbor_sums, num_real_neighbors)

    # edge_map_fr [num_points, num_class]
    edge_map_fr = tf.math.subtract(neighbor_avgs, region_softmax)
    edge_map_fr = tf.math.abs(edge_map_fr)

    return edge_map_fr



# ----------------------------------------------------------------------------------------------------------------------
#
#           Output heads
#       \*******************/
#


def segmentation_head_region(features, config, dropout_prob):
    """
    Logits generation head for semantic segmentation
    :param features: [Point_num, feature_channel]
    :param config
    :param dropout_prob
    :return logits: [Point_num, num_classes] (not normalized)
    """
    training = dropout_prob < 0.99

    # Unary conv (equivalent to fully connected for each pixel)
    with tf.variable_scope('head_unary_conv_region'):
        w = weight_variable([int(features.shape[1]), config.first_features_dim])
        features = conv_ops.unary_convolution(features, w)
        features = leaky_relu(batch_norm(features,
                                         config.use_batch_norm,
                                         config.batch_norm_momentum,
                                         training))

    # Softmax
    with tf.variable_scope('softmax_region'):
        w = weight_variable([config.first_features_dim, config.num_classes])
        b = bias_variable([config.num_classes])
        logits = conv_ops.unary_convolution(features, w) + b

    return logits


def segmentation_head_edge(features, config, dropout_prob):
    """
    Logits generation head for semantic edge detection
    :param features: [Point_num, feature_channel]
    :param config
    :param dropout_prob
    :return logits: [Point_num, num_classes] (not normalized)
    """
    # Boolean of training
    training = dropout_prob < 0.99

    # Unary conv (equivalent to fully connected for each pixel)
    with tf.variable_scope('head_unary_conv_edge'):
        w = weight_variable([int(features.shape[1]), config.first_features_dim])
        features = conv_ops.unary_convolution(features, w)
        features = leaky_relu(batch_norm(features,
                                         config.use_batch_norm,
                                         config.batch_norm_momentum,
                                         training))

    # Softmax
    with tf.variable_scope('softmax_edge'):
        w = weight_variable([config.first_features_dim, config.num_classes])
        b = bias_variable([config.num_classes])
        logits = conv_ops.unary_convolution(features, w) + b

    return logits



def side_0_head(features, config, dropout_prob):
    """
    Logits generation head for side feature 0 (binary edge map)
    :param features: [Point_num, feature_channel]
    :param config
    :param dropout_prob
    :return logits: [Point_num, 1] (not normalized)
    """
    # Boolean of training
    training = dropout_prob < 0.99

    # Unary conv (equivalent to fully connected for each pixel)
    with tf.variable_scope('head_unary_conv_side0'):
        w = weight_variable([int(features.shape[1]), config.num_classes])
        features = conv_ops.unary_convolution(features, w)
        features = leaky_relu(batch_norm(features,
                                         config.use_batch_norm,
                                         config.batch_norm_momentum,
                                         training))

    # Softmax
    with tf.variable_scope('sigmoid_side0'):
        w = weight_variable([config.num_classes, 1])
        b = bias_variable([1])
        logits = conv_ops.unary_convolution(features, w) + b

    return logits


def side_1_head(features, config, dropout_prob):
    """
    Logits generation head for side feature 1 (binary edge map)
    :param features: [Point_num, feature_channel]
    :param config
    :param dropout_prob
    :return logits: [Point_num, 1] (not normalized)
    """
    # Boolean of training
    training = dropout_prob < 0.99

    # Unary conv (equivalent to fully connected for each pixel)
    with tf.variable_scope('head_unary_conv_side1'):
        w = weight_variable([int(features.shape[1]), config.num_classes])
        features = conv_ops.unary_convolution(features, w)
        features = leaky_relu(batch_norm(features,
                                         config.use_batch_norm,
                                         config.batch_norm_momentum,
                                         training))

    # Softmax
    with tf.variable_scope('sigmoid_side1'):
        w = weight_variable([config.num_classes, 1])
        b = bias_variable([1])
        logits = conv_ops.unary_convolution(features, w) + b

    return logits


def side_2_head(features, config, dropout_prob):
    """
    Logits generation head for side feature 2 (binary edge map)
    :param features: [Point_num, feature_channel]
    :param config
    :param dropout_prob
    :return logits: [Point_num, 1] (not normalized)
    """
    # Boolean of training
    training = dropout_prob < 0.99

    # Unary conv (equivalent to fully connected for each pixel)
    with tf.variable_scope('head_unary_conv_side2'):
        w = weight_variable([int(features.shape[1]), config.num_classes])
        features = conv_ops.unary_convolution(features, w)
        features = leaky_relu(batch_norm(features,
                                         config.use_batch_norm,
                                         config.batch_norm_momentum,
                                         training))

    # Softmax
    with tf.variable_scope('sigmoid_side2'):
        w = weight_variable([config.num_classes, 1])
        b = bias_variable([1])
        logits = conv_ops.unary_convolution(features, w) + b

    return logits


def side_3_head(features, config, dropout_prob):
    """
    Logits generation head for side feature 3 (semantic segmentation mask)
    :param features: [Point_num, feature_channel]
    :param config
    :param dropout_prob
    :return logits: [Point_num, num_classes] (not normalized)
    """
    # Boolean of training
    training = dropout_prob < 0.99

    # Unary conv (equivalent to fully connected for each pixel)
    with tf.variable_scope('head_unary_conv_side3'):
        w = weight_variable([int(features.shape[1]), config.num_classes])
        features = conv_ops.unary_convolution(features, w)
        features = leaky_relu(batch_norm(features,
                                         config.use_batch_norm,
                                         config.batch_norm_momentum,
                                         training))

    # Softmax
    with tf.variable_scope('softmax_side3'):
        w = weight_variable([config.num_classes, config.num_classes])
        b = bias_variable([config.num_classes])
        logits = conv_ops.unary_convolution(features, w) + b

    return logits


def side_4_head(features, config, dropout_prob):
    """
    Logits generation head for side feature 4 (semantic segmentation mask)
    :param features: [Point_num, feature_channel]
    :param config
    :param dropout_prob
    :return logits: [Point_num, num_classes] (not normalized)
    """
    # Boolean of training
    training = dropout_prob < 0.99

    # Unary conv (equivalent to fully connected for each pixel)
    with tf.variable_scope('head_unary_conv_side4'):
        w = weight_variable([int(features.shape[1]), config.num_classes])
        features = conv_ops.unary_convolution(features, w)
        features = leaky_relu(batch_norm(features,
                                         config.use_batch_norm,
                                         config.batch_norm_momentum,
                                         training))

    # Softmax
    with tf.variable_scope('softmax_side4'):
        w = weight_variable([config.num_classes, config.num_classes])
        b = bias_variable([config.num_classes])
        logits = conv_ops.unary_convolution(features, w) + b

    return logits



# ----------------------------------------------------------------------------------------------------------------------
#
#           Loss Terms
#       \*******************/
#


def segmentation_loss_edge(logits, inputs, batch_average=False):
    # logist: [Point_num, num_classes] (not normalized)
    # point_b_c_i: [Point_num] store the class for boundary points
        # -1 indicates empty
    point_b_c_0 = inputs['point_b_c_0']
    point_b_c_1 = inputs['point_b_c_1']
    point_b_c_2 = inputs['point_b_c_2']

    num_points = tf.shape(logits)[0]
    num_classes = tf.shape(logits)[1]

    # boundary_target: [Point_num, num_classes] Every column representing the boundary labels for one class
    point_b_c_0_onehot = tf.one_hot(point_b_c_0, num_classes)
    point_b_c_0_onehot = tf.dtypes.cast(point_b_c_0_onehot, dtype=tf.bool)
    point_b_c_1_onehot = tf.one_hot(point_b_c_1, num_classes)
    point_b_c_1_onehot = tf.dtypes.cast(point_b_c_1_onehot, dtype=tf.bool)
    point_b_c_2_onehot = tf.one_hot(point_b_c_2, num_classes)
    point_b_c_2_onehot = tf.dtypes.cast(point_b_c_2_onehot, dtype=tf.bool)
    boundary_target = tf.math.logical_or(point_b_c_0_onehot, point_b_c_1_onehot)
    boundary_target = tf.math.logical_or(boundary_target, point_b_c_2_onehot)
    boundary_target = tf.dtypes.cast(boundary_target, tf.float32)
    
    # weight = (1 - boundary_percentage) / boundary_percentage [num_classes]
    boundary_percentages = tf.reduce_sum(boundary_target, axis=0) / tf.dtypes.cast(num_points, tf.float32)  # [num_classes]
    weight = (1.0 - boundary_percentages) / (boundary_percentages + 1e-6)

    # calculate weighted multi-label loss
    # labels * -log(sigmoid(logits)) * pos_weight + (1 - lables) * -log(1 - sigmoid(logits))
    log_weight = 1 + (weight - 1) * boundary_target
    cross_entropy = (1.0 - boundary_target) * logits + log_weight * (tf.math.log1p(tf.math.exp(-tf.math.abs(logits))) + tf.nn.relu(-logits))

    cross_entropy = tf.reduce_sum(cross_entropy, axis=1)

    if not batch_average:
        # Option 1 : Mean on all points of all batch
        return tf.reduce_mean(cross_entropy, name='ml_xentropy_mean')

    else:
        # Option 2 : First mean on each batch, then mean (correspond to weighted sum with batch proportions)
        stacked_weights = inputs['batch_weights']
        return tf.reduce_mean(stacked_weights * cross_entropy, name='ml_xentropy_mean')


def segmentation_loss_efr(edge_map_fr, inputs, batch_average=False, scannet=False, inds=None):

    # edge_map_fr: [Point_num, num_classes] (not normalized)
    point_labels = inputs['point_labels']
    convert_neighbors = inputs['convert_neighbors']
    max_ind = tf.math.reduce_max(inputs['in_batches'])
    num_classes = tf.shape(edge_map_fr)[1]

    is_boundaries = inputs['point_boundaries']
    
    boundary_percentage = tf.reduce_sum(is_boundaries) / tf.shape(is_boundaries)[0]
    def f1(): return 1.0
    def f2(): return tf.dtypes.cast((1 - boundary_percentage) / boundary_percentage, tf.float32)
    weight = tf.cond(tf.math.equal(boundary_percentage, 0.0), true_fn=f1, false_fn=f2)
    
    if scannet == True:
        # convert array label to onehot matrix [num_points, num_class]
        point_labels_onehot = tf.one_hot(point_labels, num_classes + tf.constant(2, dtype=tf.int32), dtype=tf.float32)

        # delete the first two column with ignored labels
        point_labels_onehot = tf.slice(point_labels_onehot, [0, 2], [-1, num_classes])
    else:
        # convert array label to onehot matrix [num_points, num_class]
        point_labels_onehot = tf.one_hot(point_labels, num_classes, dtype=tf.float32)
    
    # Add a last row with zero activation for shadow filtering [num_points+1, num_class]
    point_labels_onehot_shadow = tf.concat([point_labels_onehot, tf.zeros((1, num_classes), dtype=tf.float32)], axis=0)

    # Get activations for each point [num_points, num_neighbor, num_class]
    neighbor_activations = tf.gather(point_labels_onehot_shadow, convert_neighbors, axis=0)
    
    # neighbor_sums [num_points, num_class]
    neighbor_sums = tf.reduce_sum(neighbor_activations, axis=1)
    
    # number of real neighbors [num_points, 1]
    num_real_neighbors = tf.math.less(convert_neighbors, max_ind)
    num_real_neighbors = tf.dtypes.cast(num_real_neighbors, tf.float32)
    num_real_neighbors = tf.reduce_sum(num_real_neighbors, axis=1, keepdims=True)

    # neighbor_avgs [num_points, num_class]
    neighbor_avgs = tf.math.divide(neighbor_sums, num_real_neighbors)

    # gt_map_fr [num_points, num_class]
    gt_map_fr = tf.math.subtract(neighbor_avgs, point_labels_onehot)
    gt_map_fr = tf.math.abs(gt_map_fr)

    if scannet == True:
        edge_map_fr = tf.gather(edge_map_fr, inds, axis=0)
        gt_map_fr = tf.gather(gt_map_fr, inds, axis=0)
    
    # calculate loss
    l1_loss = tf.math.subtract(edge_map_fr, gt_map_fr)
    l1_loss = tf.math.abs(l1_loss)
    l1_loss = tf.reduce_sum(l1_loss, axis=1)
    l1_loss = l1_loss * weight

    if not batch_average:
        # Option 1 : Mean on all points of all batch
        return tf.reduce_mean(l1_loss, name='efr_l1_mean')

    else:
        # Option 2 : First mean on each batch, then mean (correspond to weighted sum with batch proportions)
        stacked_weights = inputs['batch_weights']
        return tf.reduce_mean(stacked_weights * l1_loss, name='efr_l1_mean')   


def segmentation_loss_region(logits, inputs, batch_average=False):

    # Exclusive Labels cross entropy on each point
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs['point_labels'],
                                                                   logits=logits,
                                                                   name='xentropy')

    if not batch_average:
        # Option 1 : Mean on all points of all batch
        return tf.reduce_mean(cross_entropy, name='xentropy_mean')

    else:
        # Option 2 : First mean on each batch, then mean (correspond to weighted sum with batch proportions)
        stacked_weights = inputs['batch_weights']
        return tf.reduce_mean(stacked_weights * cross_entropy, name='xentropy_mean')



def bce_loss(logits, inputs, batch_average=False):

    is_boundaries = inputs['point_boundaries']
    
    boundary_percentage = tf.reduce_sum(is_boundaries) / tf.shape(is_boundaries)[0]
    def f1(): return 1.0
    def f2(): return tf.dtypes.cast((1 - boundary_percentage) / boundary_percentage, tf.float32)
    weight = tf.cond(tf.math.equal(boundary_percentage, 0.0), true_fn=f1, false_fn=f2)
    
    is_boundaries = tf.dtypes.cast(is_boundaries, tf.float32)
    logits = tf.squeeze(logits, axis=1)

    bce_loss = tf.nn.weighted_cross_entropy_with_logits(targets=is_boundaries,
                                                        logits=logits,
                                                        pos_weight=weight,
                                                        name='b_xentropy')
    if not batch_average:
        # Option 1 : Mean on all points of all batch
        return tf.reduce_mean(bce_loss, name='b_xentropy_mean')

    else:
        # Option 2 : First mean on each batch, then mean (correspond to weighted sum with batch proportions)
        stacked_weights = inputs['batch_weights']
        return tf.reduce_mean(stacked_weights * bce_loss, name='b_xentropy_mean')

