"""Custom layers that constitute building blocks of NN models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any

import numpy as np
import sonnet as snt
import tensorflow as tf


NONLINEARITIES = {
    'relu': tf.nn.relu,
    'exp': tf.exp,
    'identity': tf.identity
}


class Conv1dPeriodic(snt.AbstractModule):
  """1D Convolution module with periodic boundary conditions."""

  def __init__(
      self,
      output_channels: int,
      kernel_shape: int,
      stride: int = 1,
      name: str = 'conv_1d_periodic'):
    """Constructs Conv1dPeriodic moduel.

      Args:
      output_channels: Number of channels in convolution.
      kernel_shape: Convolution kernel sizes.
      stride: Convolution stride.
      name: Name of the module.
    """
    super(Conv1dPeriodic, self).__init__(name=name)
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._stride = stride
    with self._enter_variable_scope():
      self._conv_1d_module = snt.Conv1D(
          output_channels=self._output_channels,
          kernel_shape=self._kernel_shape,
          stride=self._stride, padding=snt.VALID)

  def _pad_input(self, inputs: tf.Tensor) -> tf.Tensor:
    """Pads 'inputs' on left and right side to achieve periodic effect.

    Implements effect of periodic boundary conditions by padding the `inputs`
    tensor on both sides such that VALID padding is equivalent to using periodic
    boundaries.

    Args:
      inputs: Tensor to pad.

    Returns:
      Padded tensor.
    """
    input_size = inputs.get_shape().as_list()[1]
    if self._kernel_shape % 2 == 1:
      pad_size = (self._kernel_shape - 1) // 2
      left_pad = inputs[:, (input_size - pad_size):, ...]
      right_pad = inputs[:, :pad_size, ...]
    else:
      left_pad_size = self._kernel_shape // 2
      right_pad_size = self._kernel_shape // 2 - 1
      left_pad = inputs[:, (input_size - left_pad_size):, ...]
      right_pad = inputs[:, :right_pad_size, ...]
    return tf.concat([left_pad, inputs, right_pad], axis=1)


  def _build(self, inputs: tf.Tensor) -> tf.Tensor:
    """Connects the Conv1dPeridic module into the graph, with input `inputs`.

    Args:
      inputs: Tensor with input values.

    Returns:
      Result of convolving `inputs` with module's kernels with periodic padding.
    """
    return self._conv_1d_module(self._pad_input(inputs))


class Conv2dPeriodic(snt.AbstractModule):
  """2D Convolution module with periodic boundary conditions."""

  def __init__(
      self,
      output_channels: int,
      kernel_shape: Any,
      stride: Any = 1,
      name: str = 'conv_2d_periodic'):
    """Constructs Conv2dPeriodic moduel.

    Args:
      output_channels: Number of output channels.
      kernel_shape: Sequence of kernel sizes (of size 2), or integer that is
          used to define kernel size in all dimensions.
      stride: Sequence of kernel strides (of size 2), or integer that is used to
          define stride in all dimensions.
      name: Name of the module.
    """
    super(Conv2dPeriodic, self).__init__(name=name)
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._stride = stride
    with self._enter_variable_scope():
      self._conv_2d_module = snt.Conv2D(output_channels=self._output_channels,
                                        kernel_shape=self._kernel_shape,
                                        stride=self._stride, padding=snt.VALID)

  def _pad_input(self, inputs: tf.Tensor) -> tf.Tensor:
    """Pads 'inputs' on all sides to achieve periodic effect.

    Implements effect of periodic boundary conditions by padding the `inputs`
    tensor on all sides such that VALID padding is equivalent to using periodic
    boundaries.

    Args:
      inputs: Tensor to pad. dim(inputs) must be >= 3.

    Returns:
      Padded tensor.
    """
    x_input_size = inputs.get_shape().as_list()[2]
    y_input_size = inputs.get_shape().as_list()[1]
    if self._kernel_shape % 2 == 1:
      left_pad_size = (self._kernel_shape - 1) // 2
      right_pad_size = (self._kernel_shape - 1) // 2
      bot_pad_size = (self._kernel_shape - 1) // 2
      top_pad_size = (self._kernel_shape - 1) // 2
    else:
      left_pad_size = self._kernel_shape // 2 - 1
      right_pad_size = self._kernel_shape // 2
      bot_pad_size = self._kernel_shape // 2 - 1
      top_pad_size = self._kernel_shape // 2

    left_pad = inputs[:, :, (x_input_size - left_pad_size):, ...]
    right_pad = inputs[:, :, :right_pad_size, ...]
    width_padded = tf.concat([left_pad, inputs, right_pad], axis=2)
    bot_pad = width_padded[:, (y_input_size - bot_pad_size):, ...]
    top_pad = width_padded[:, :top_pad_size, ...]
    return tf.concat([bot_pad, width_padded, top_pad], axis=1)


  def _build(self, inputs: tf.Tensor) -> tf.Tensor:
    """Connects the Conv1dPeridic module into the graph, with input `inputs`.

    Args:
      inputs: Tensor with input values.

    Returns:
      Result of convolving `inputs` with module's kernels with periodic padding.
    """
    return self._conv_2d_module(self._pad_input(inputs))


class ResBlock2d(snt.AbstractModule):
  """Residual network block for 2D system with periodic boundary conditions.

  A building block of ResidualNetworks. It performs 2 convolutions with Selu
  activations, which are than concatenated together with the input. In this
  implementations we use convolutions that pad the input to produce periodic
  effect. We also omit batch normalization present in the orginal paper:
  https://arxiv.org/pdf/1512.03385.pdf."""

  def __init__(
      self,
      num_filters: int,
      kernel_shape: Any,
      conv_stride: Any = 1,
      projection_shortcut: snt.AbstractModule = None,
      name: str = 'res_block_2d'):
    """Constructs a ResNet block for 1D systems.

    Args:
      num_filters: Number of filters for the convolutions.
      kernel_shape: Shape of the kernel for the convolutions.
      conv_stride: Stride for the convolutions.
      projection_shortcut: The module to apply to shortcuts.
      name: Name of the module.
    """
    super(ResBlock2d, self).__init__(name=name)
    self._num_filters = num_filters
    self._kernel_shape = kernel_shape
    self._conv_stride = conv_stride
    self._projection_shortcut = projection_shortcut
    with self._enter_variable_scope():
      self._output_channels = num_filters
      conv_arguments = {
          'output_channels': self._num_filters,
          'kernel_shape': self._kernel_shape,
          'stride': self._conv_stride,
      }
      self._conv_2d_1 = Conv2dPeriodic(**conv_arguments, name='first_conv')
      self._conv_2d_2 = Conv2dPeriodic(**conv_arguments, name='second_conv')


  def _build(self, inputs: tf.Tensor) -> tf.Tensor:
    """Connects the ResBlock1d module into the graph, with input `inputs`.

    Args:
      inputs: Tensor with input values of shape [batch, size_x, size_y, ...]

    Returns:
      Result of application of ResNetBlock2d to `inputs`.

    Raises:
      ValueError: Inputs shape is not compatable with filters.
    """
    #TODO(dkochkov)  Change convolutions to follow BCWH convention
    num_channels = inputs.shape.as_list()[3]
    if self._output_channels != num_channels:
      raise ValueError('Inputs shape is not compatable with filters.')

    if self._projection_shortcut is None:
      shortcut = inputs
    else:
      shortcut = self._projection_shortcut(inputs)

    components = [self._conv_2d_1, tf.nn.selu, self._conv_2d_2]
    residual_value = snt.Sequential(components)(inputs)
    return residual_value + shortcut


class ResBlock1d(snt.AbstractModule):
  """Residual network block for 1D system with periodic boundary conditions.

  A building block of ResidualNetworks. It performs 2 convolutions with ReLU
  activations, which are than concatenated together with the input. In this
  implementations we use convolutions that pad the input to produce periodic
  effect. We also omit batch normalization comparing to the orginal paper:
  https://arxiv.org/pdf/1512.03385.pdf."""

  def __init__(
      self,
      num_filters: int,
      kernel_shape: Any,
      conv_stride: Any = 1,
      projection_shortcut: snt.AbstractModule = None,
      name: str = 'res_block_1d'):
    """Constructs a ResNet block for 1D systems.

    Args:
      num_filters: Number of filters for the convolutions.
      kernel_shape: Shape of the kernel for the convolutions.
      conv_stride: Stride for the convolutions.
      projection_shortcut: The module to apply to shortcuts.
      name: Name of the module.
    """
    super(ResBlock1d, self).__init__(name=name)
    self._num_filters = num_filters
    self._kernel_shape = kernel_shape
    self._conv_stride = conv_stride
    self._projection_shortcut = projection_shortcut
    with self._enter_variable_scope():
      self._output_channels = num_filters
      conv_arguments = {
          'output_channels': self._num_filters,
          'kernel_shape': self._kernel_shape,
          'stride': self._conv_stride,
      }
      self._conv_1d_1 = Conv1dPeriodic(**conv_arguments, name='first_conv')
      self._conv_1d_2 = Conv1dPeriodic(**conv_arguments, name='second_conv')


  def _build(self, inputs: tf.Tensor) -> tf.Tensor:
    """Connects the ResBlock1d module into the graph, with input `inputs`.

    Args:
      inputs: Tensor with input values of shape [batch, channels, n_sites,...]

    Returns:
      Result of application of ResNetBlock1d to `inputs`.

    Raises:
      ValueError: Inputs shape is not compatable with filters.
    """
    num_channels = inputs.shape.as_list()[2]
    if self._output_channels != num_channels:
      raise ValueError('Inputs shape is not compatable with filters.')

    if self._projection_shortcut is None:
      shortcut = inputs
    else:
      shortcut = self._projection_shortcut(inputs)

    components = [self._conv_1d_1, tf.nn.selu, self._conv_1d_2]
    residual_value = snt.Sequential(components)(inputs)
    return residual_value + shortcut


class BottleneckResBlock1d(snt.AbstractModule):
  """Residual network block for 1D system with periodic boundary conditions.

  In contrast to ResBlock1d this module uses bottleneck to reduce computation.
  For details see reference in `ResBlock1d`.
  """
  def __init__(
      self,
      num_filters: int,
      kernel_shape: Any,
      conv_stride: Any = 1,
      bottleneck_ratio: int = 2,
      projection_shortcut: snt.AbstractModule = None,
      name: str = 'res_block_1d'):
    """Constructs a ResNet block for 1D systems.

    Args:
      num_filters: Number of filters for the convolutions.
      kernel_shape: Shape of the kernel for the convolutions.
      conv_stride: Stride for the convolutions.
      bottleneck_ratio: Ratio of bottleneck compression.
      projection_shortcut: The module to apply to shortcuts.
      name: Name of the module.
    """
    super(BottleneckResBlock1d, self).__init__(name=name)
    self._num_filters = num_filters
    self._kernel_shape = kernel_shape
    self._conv_stride = conv_stride
    self._bottleneck_ratio = bottleneck_ratio
    self._projection_shortcut = projection_shortcut
    with self._enter_variable_scope():
      output_size = self._num_filters * self._bottleneck_ratio
      self._conv_1d_1 = Conv1dPeriodic(self._num_filters, 1, 1)
      self._conv_1d_2 = Conv1dPeriodic(self._num_filters, self._kernel_shape)
      self._conv_1d_3 = Conv1dPeriodic(output_size, 1, 1)


  def _build(self, inputs: tf.Tensor) -> tf.Tensor:
    """Connects the ResBlock1d module into the graph, with input `inputs`.

    Args:
      inputs: Tensor with input values of shape [batch, channels, n_sites,...]

    Returns:
      Result of application of ResNetBlock1d to `inputs`.

    Raises:
      ValueError: Inputs shape is not compatable with filters.
    """
    num_channels = inputs.shape.as_list()[2]
    if self._output_channels != num_channels:
      raise ValueError('Inputs shape is not compatable with filters.')

    if self._projection_shortcut is None:
      shortcut = inputs
    else:
      shortcut = self._projection_shortcut(inputs)

    components = [
        self._conv_1d_1, tf.nn.relu, self._conv_1d_2,
        tf.nn.relu, self._conv_1d_3
    ]
    residual_value = snt.Sequential(components)(inputs)
    return residual_value + shortcut


class MatrixProductUnit(snt.AbstractModule):
  """Matrix product module representing a single MPS cell."""

  def __init__(
      self,
      vertical_bond_dimension: int,
      horizontal_bond_dimension: int,
      physical_dimension: int = 2,
      name: str = 'matrix_product_module'):
    """Constructs MatrixProductUnit module.

    Args:
      vertical_bond_dimension: Number of rows in MPS unit.
      horizontal_bond_dimension: Number of bonds in MPS unit.
      physical_dimension: Number of physical dimensions.
      name: Name of the module.

    Raises:
      ValueError: Provided physical dimension is not supported.
    """
    if physical_dimension != 2:
      raise ValueError('Only physical dimension 2 is currently supported.')

    super(MatrixProductUnit, self).__init__(name=name)
    self._vertical_bond_dimension = vertical_bond_dimension
    self._horizontal_bond_dimension = horizontal_bond_dimension
    self._physical_dimension = physical_dimension
    self._name = name
    with self._enter_variable_scope():
      shape = [
          self._vertical_bond_dimension,
          self._horizontal_bond_dimension,
          self._physical_dimension,
      ]
      self._mps_var = tf.get_variable('M', shape=shape, dtype=tf.float32)

  def _build(self, inputs: tf.Tensor) -> tf.Tensor:
    """Connects the MatrixProductUnit module into the graph, with `inputs`.

    Args:
      inputs: Tensor with input values with shape=[batch] and values +/- 1.

    Returns:
      MPS matrices corresponding to the physical degrees of freedom.
    """
    index_inputs = tf.unstack(tf.cast((inputs + 1) / 2, tf.int32))

    batch_components = [self._mps_var[:, :, index] for index in index_inputs]
    return tf.stack(batch_components, axis=0)


class GraphConvLayer(snt.AbstractModule):
  """GraphConvLayer module with adjacency list"""

  def __init__(
      self,
      output_channels: int,
      adj: np.ndarray,
      name: str = 'graph_conv_layer'):
    """Constructs GraphConvLayer moduel.

    Args:
      output_channels: Number of output channels in convolution.
      adj: Adjacency list of the graph that stores indices of neighbors
          and itself for every site, with shape [n_site, num_neighbor].
      name: Name of the module.
    """
    super(GraphConvLayer, self).__init__(name=name)
    self._output_channels = output_channels
    self._adj = adj
    num_neighbors = np.shape(self._adj)[1]
    kernel_shape = (1, num_neighbors)
    with self._enter_variable_scope():
      self._conv_module = snt.Conv2D(output_channels=output_channels,
                                     kernel_shape=kernel_shape,
                                     padding=snt.VALID)

  def _build(self, inputs: tf.Tensor) -> tf.Tensor:
    """Connects the GraphConvLayer module into the graph, with input `inputs`.

    Args:
      inputs: Tensor with input values.

    Returns:
      Result of convolving `inputs` with adjacency matrix and module's kernels.
    """
    adj_table = tf.gather(inputs, self._adj, axis=1)
    return tf.squeeze(self._conv_module(adj_table), axis=2)
