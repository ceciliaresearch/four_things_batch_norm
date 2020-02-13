"""This file gives the code for a normalization layer which includes Batch
Normalization, Ghost Batch Normalization, and Group Normalization as special
cases. It also supports custom weight decay on the scale and shift variables
(beta and gamma) and using a weighted average of example and moving average
statistics during inference.

Example usage:

# Batch Normalization (batch size = 128)
x = normalization_layer(x, channels_per_group=1, examples_per_group=128)

# Ghost Batch Normalization (ghost batch size = 16)
x = normalization_layer(x, channels_per_group=1, examples_per_group=16)

# Group Normalization
x = normalization_layer(x, channel_groups=32, examples_per_group=1)

# Batch/Group Normalization Generalization
x = normalization_layer(x, channel_groups=32, examples_per_group=2)
"""

import tensorflow as tf

def normalization_layer(x, is_training=True, scope='norm', channels_per_group=0,
    channel_groups=0, weight_decay=0., moving_average_decay=0.99,
    examples_per_group=0, eps=1e-5, example_eval_weight=0.):
  with tf.variable_scope(scope) as scope:
    # Assumes this is for a convolutional layer in channels_first format.
    _, channels, height, width = x.get_shape().as_list()
    num_examples = tf.shape(x)[0]

    # Figure out the number of channels per group/number of groups.
    channel_groups, channels_per_group = get_num_channel_groups(
        channels, channels_per_group, channel_groups)
    beta_regularizer, gamma_regularizer = get_bn_regularizers()
    beta = tf.get_variable(
        name='beta',
        initializer=tf.constant(0.0, shape=[1, channels, 1, 1]),
        trainable=True,
        dtype=tf.float32,
        regularizer=beta_regularizer)
    gamma = tf.get_variable(
        name='gamma',
        initializer=tf.constant(1.0, shape=[1, channels, 1, 1]),
        trainable=True,
        dtype=tf.float32,
        regularizer=gamma_regularizer)

    moving_x = tf.get_variable(
        name='moving_x',
        shape=(1, channels, 1, 1),
        initializer=tf.initializers.zeros,
        trainable=False,
        dtype=tf.float32)
    moving_x2 = tf.get_variable(
        name='moving_x2',
        shape=(1, channels, 1, 1),
        initializer=tf.initializers.ones,
        trainable=False,
        dtype=tf.float32)

    # Compute normalization statistics with sufficient_statistics for
    # flexibility and efficiency.
    counts, channel_x, channel_x2, _ = tf.nn.sufficient_statistics(
        x, [2, 3], keep_dims=True)
    channel_x /= counts
    channel_x2 /= counts

    if is_training:
      # Add updates:
      x_update = tf.assign(moving_x,
          moving_average_decay * moving_x + (1. - moving_average_decay) *
          tf.reduce_mean(channel_x, axis=0, keepdims=True))
      x2_update = tf.assign(moving_x2,
          moving_average_decay * moving_x2 + (1. - moving_average_decay) *
          tf.reduce_mean(channel_x2, axis=0, keepdims=True))
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, x_update)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, x2_update)

      # Group by example group and channel group.
      examples_per_group = tf.minimum(examples_per_group, num_examples)
      # Assume that num_examples is always divisible by examples_per_group.
      example_groups = num_examples // examples_per_group
      channel_x = tf.reshape(channel_x,
          [example_groups, examples_per_group,
           channel_groups, channels_per_group, 1, 1])
      channel_x2 = tf.reshape(channel_x2,
          [example_groups, examples_per_group,
           channel_groups, channels_per_group, 1, 1])

      group_mean = tf.reduce_mean(channel_x, axis=[1, 3], keep_dims=True)
      group_x2 = tf.reduce_mean(channel_x2, axis=[1, 3], keep_dims=True)
      group_var = group_x2 - tf.pow(group_mean, 2)

      nc_mean = tf.reshape(
          tf.tile(group_mean,
                  [1, examples_per_group, 1, channels_per_group, 1, 1]),
          [-1, channels, 1, 1])
      nc_var = tf.reshape(
          tf.tile(group_var,
                  [1, examples_per_group, 1, channels_per_group, 1, 1]),
          [-1, channels, 1, 1])

      mult = gamma * tf.rsqrt(nc_var + eps)
      add = -nc_mean * mult + beta
      x = x * mult + add
    else:
      # is_training == False
      channel_x = tf.reshape(channel_x,
          [num_examples, channel_groups, channels_per_group, 1, 1])
      channel_x2 = tf.reshape(channel_x2,
          [num_examples, channel_groups, channels_per_group, 1, 1])
      group_x = tf.reduce_mean(channel_x, axis=[2], keepdims=True)
      group_x2 = tf.reduce_mean(channel_x2, axis=[2], keepdims=True)
      moving_x_group = tf.reduce_mean(tf.reshape(moving_x,
        [1, channel_groups, channels_per_group, 1, 1]), axis=[2], keepdims=True)
      moving_x2_group = tf.reduce_mean(tf.reshape(moving_x2,
        [1, channel_groups, channels_per_group, 1, 1]), axis=[2], keepdims=True)

      norm_x = (1. - example_eval_weight) * moving_x_group + (
          example_eval_weight * group_x)
      norm_x2 = (1. - example_eval_weight) * moving_x2_group + (
          example_eval_weight * group_x2)
      norm_var = norm_x2 - tf.pow(norm_x, 2)

      norm_x = tf.reshape(tf.tile(norm_x, [1, 1, channels_per_group, 1, 1]),
          [num_examples, channels, 1, 1])
      norm_var = tf.reshape(tf.tile(norm_var, [1, 1, channels_per_group, 1, 1]),
          [num_examples, channels, 1, 1])

      mult = gamma * tf.rsqrt(norm_var + eps)
      add = -norm_x * mult + beta
      x = x * mult + add
  return x


def get_num_channel_groups(channels, channels_per_group=0, channel_groups=0):
  if channels_per_group > 0:
    channels_per_group = min(channels_per_group, channels)
  elif channel_groups > 0:
    channels_per_group = max(channels // channel_groups, 1)
  else:
    raise ValueError('Either channels_per_group or channel_groups must be '
                     'provided.')
  channel_groups = channels // channels_per_group
  return (channel_groups, channels_per_group)


def get_bn_regularizers(weight_decay=0.):
  if weight_decay > 0:
    gamma_reg_baseline = 1.
    beta_regularizer = lambda tensor: weight_decay * tf.nn.l2_loss(tensor)
    gamma_regularizer = lambda tensor: weight_decay * tf.nn.l2_loss(
        tensor - gamma_reg_baseline)
  else:
    beta_regularizer = None
    gamma_regularizer = None
  return beta_regularizer, gamma_regularizer