"""Defines wavefunction inteface and implemens various wavefunction networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import inspect

from typing import Dict, List, Any

import os
import numpy as np
import tensorflow as tf
import sonnet as snt

import layers


class Wavefunction(snt.AbstractModule):
  """Defines wavefunction interface.

  Wavefunction class is an abstraction that decouples training and evaluation
  procedures from the internal structure of the model. Every wavefunction
  instance must implement `_build` method, that adds wavefunction evaluation
  to the graph.

  Base class provides generic methods applicable to all wavefunctions. If
  generic implementation is not applicable to a particular instance, these
  methods can be overwritten by the subclass. The construction is based on
  the Abstract module in the Sonnet framework. To make variable sharing and
  name_spaces work properly every instance must call super() with a `name`
  argument. For more details see https://deepmind.github.io/sonnet/.

  For current implementation of deep_copy for every argument in the constructor
  a corresponding variable with an underscore must be added to the class. For
  proper variable transfer, all wavefunction components should be added to list
  `_sub_wavefunctions` after calling super.
  """
  def __init__(self, name: str = 'wavefunction'):
    """Creates a Wavefunction instance"""
    super(Wavefunction, self).__init__(name=name)
    self._sub_wavefunctions = []
    self._exp_norm_shift = None  # scale shift variable

  def _build(self, inputs: tf.Tensor) -> tf.Tensor:
    """Builds computational graph evaluating the wavefunction on inputs.

    Args:
      inputs: Input tensor, must have shape (batch, num_sites, ...).

    Returns:
      Tensor holding values of the wavefunction on `inputs`.

    Raises:
      ValueError: Input tensor has wrong shape.
    """
    raise NotImplementedError

  def __add__(self, other: 'Wavefunction') -> 'Wavefunction':
    """Generates new wavefunction that equals to `self` + `other`."""

    class SumOfWavefunctions(Wavefunction):
      """Wavefunction wrapper of the sum of two wavefunctions."""
      def __init__(
          self,
          wf_a: Wavefunction,
          wf_b: Wavefunction,
          name: str = 'sum_of_wavefunctions'
      ):
        """Creates an WavefunctionSum module."""
        # pylint: disable=protected-access
        name = '_plus_'.join([wf_a._unique_name, wf_b._unique_name])
        super(SumOfWavefunctions, self).__init__(name=name)
        self._wf_a = wf_a
        self._wf_b = wf_b
        self._sub_wavefunctions += [self._wf_a, self._wf_b]

      def _build(self, inputs: tf.Tensor) -> tf.Tensor:
        """Builds computational graph evaluating the wavefunction on inputs.

        Args:
          inputs: Input tensor, must have shape (batch, num_sites, ...).

        Returns:
          Tensor holding values of the wavefunction on `inputs`.

        Raises:
          ValueError: Input tensor has wrong shape.
        """
        return self._wf_a(inputs) + self._wf_b(inputs)

      @classmethod
      def from_hparams(cls, hparams: tf.contrib.training.HParams):
        """Constructs an instance of a class from hparams."""
        raise ValueError('Hparams initialization is not supported for sum.')

    return SumOfWavefunctions(self, other)

  def __mul__(self, other: Any) -> 'Wavefunction':
    """Generates new wavefunction that equals `self` * `other`.

    Args:
      other: Wavefunction by which the current wavefunction is multiplied.
          Could be a Wavefunction object or Tensor.

    Returns:
      Wavefunction representing the product of `self` and `other`.
    """

    class ProductOfWavefunctions(Wavefunction):
      """Wavefunction wrapper that represents a product wavefunctions."""
      def __init__(
          self,
          wf_a: 'Wavefunction',
          wf_b: Any,
          name: str = 'product_of_wavefunctions'
      ):
        """Creates an WavefunctionSum module."""
        self._wf_a = wf_a
        self._wf_b = wf_b
        components = [self._wf_a]
        if isinstance(self._wf_b, Wavefunction):
          # pylint: disable=protected-access
          name = '_times_'.join([wf_b._unique_name, wf_a._unique_name])
          components += [self._wf_b]
        elif isinstance(self._wf_b, tf.Tensor):
          # pylint: disable=protected-access
          name = '_times_'.join([wf_b.name.rstrip(':0'), wf_a._unique_name])
        elif isinstance(self._wf_b, float):
          factor_name = str(wf_b).replace('-', 'neg_')
          # pylint: disable=protected-access
          name = '_times_'.join([factor_name, wf_a._unique_name])
        super(ProductOfWavefunctions, self).__init__(name=name)
        self._sub_wavefunctions += components

      def _build(self, inputs: tf.Tensor) -> tf.Tensor:
        """Builds computational graph evaluating the wavefunction on inputs.

        Args:
          inputs: Input tensor, must have shape (batch, num_sites, ...).

        Returns:
          Tensor holding values of the wavefunction on `inputs`.

        Raises:
          ValueError: Input tensor has wrong shape.
        """
        if isinstance(self._wf_b, Wavefunction):
          return self._wf_a(inputs) * self._wf_b(inputs)
        if isinstance(self._wf_b, (float, tf.Tensor)):
          return self._wf_a(inputs) * self._wf_b
        raise ValueError('Type of other is not supported.')

      @classmethod
      def from_hparams(cls, hparams: tf.contrib.training.HParams):
        """Constructs an instance of a class from hparams."""
        raise ValueError('Hparams initialization is not supported for product.')

    return ProductOfWavefunctions(self, other)

  def __sub__(self, other: 'Wavefunction') -> 'Wavefunction':
    """Generates new wavefunction that equals `self` - `other`."""
    return self.__add__(other * -1.)

  def get_trainable_variables(self):
    """Returns a list of trainable variables in this wavefunction."""
    trainable_variables = []
    # pylint: disable=protected-access
    trainable_variables += tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._unique_name + '/')
    for sub_wavefunction in self._sub_wavefunctions:
      trainable_variables += sub_wavefunction.get_trainable_variables()
    return trainable_variables

  def __deepcopy__(self, memo: Dict[int, Any]) -> 'Wavefunction':
    """Implements deepcopy method to replicate wavefunction ansatz.

    This function must be avoided via direct implementation of __deepcopy__
    by the Wavefunction instances. This implementation is provided for testing
    and is likely to cause mysterious bugs. You have been warned.

    Args:
      memo: Dictionary that keeps track of objects that were already copied.

    Returns:
      A deep copy of the current module.
    """
    id_self = id(self)
    _copy = memo.get(id_self)
    if _copy is not None:
      return _copy
    init_args = inspect.getfullargspec(self.__init__)[0]  # get all arguments
    init_args.remove('self')  # this is provided automatically
    init_args.remove('name')  # will provide custom name
    init_values = {
        arg: copy.deepcopy(getattr(self, '_{}'.format(arg)), memo)
        for arg in init_args
    }
    init_values['name'] = 'dc_{}'.format(getattr(self, '_unique_name'))
    _copy = type(self)(**init_values)
    memo[id_self] = _copy
    return _copy

  def add_exp_normalization(
      self,
      pre_exp_tensor: tf.Tensor,
      initial_exp_norm_shift: np.float32 = -10.
  ) -> tf.Tensor:
    """Adds normalization constant to log of psi, prior to exponentiation.

    Adds a constant non-trainable shift to the `pre_exp_tensor` that can be
    adjusted throughout training to avoid overflows. This method will be
    depricated once wavefunction interface is moved to log + phase.

    Args:
      pre_exp_tensor: Tensor of shape [batch_size] to be shifted before exp.
      initial_exp_norm_shift: Initial value of the shift.

    Returns:
      Tensor with values `pre_exp_tensor` shifted by `exp_norm_shift`.
    """
    # TODO(kochkov92) Remove when move to log(psi) + phase(psi)
    self._exp_norm_shift = tf.get_variable(
        name='exp_norm_shift',
        shape=[],
        initializer=tf.constant_initializer(initial_exp_norm_shift),
        dtype=tf.float32,
        trainable=False
    )
    return pre_exp_tensor - self._exp_norm_shift

  def normalize_batch(
      self,
      batch_of_amplitudes: tf.Tensor,
      max_value: np.float32 = 1e10,
  ) -> Any:
    """If output_activation is tf.exp, builds operation that normalize psi.

    Build an operation that changes the normalization variable such that
    values of batch_of_amplitudes are mapped on (0, max_value) interval.
    If mode does not use tf.exp as output activation, returns None.
    Note that this procedure does not guarantee that arbitrary psi(inputs)
    is <= max_value on all inputs. This method can be overriden by other
    wavefunctions if needed.

    Args:
      batch_of_amplitudes: Batch of amplitudes.
      max_value: Upper bound on batch of amplitudes.

    Returns:
      Operation that updates normalization or None if activation is not 'exp'.
    """
    if self._exp_norm_shift is None:
      return None
    max_amplitude_tensor = tf.reduce_max(batch_of_amplitudes)
    log_max = tf.log(max_amplitude_tensor)
    return tf.assign_add(self._exp_norm_shift, log_max - np.log(max_value))

  def update_norm(
      self,
      batch_of_amplitudes: tf.Tensor,
      max_value: np.float32 = 1e10,
  ) -> Any:
    """Builds operation that update wf values in batch not to exceed max_value.

    Similar to normalize batch, returns None if output_activation of the model
    is not tf.exp. In contrast to normalize_batch, this operation changes the
    normalization only if the value of the amplitude in the batch exceeds
    max_value.

    Args:
      batch_of_amplitudes: Batch of amplitudes.
      max_value: Desired upper bound on batch of amplitudes.

    Returns:
      Operation that updates normalization variable.
    """
    if self._exp_norm_shift is None:
      return None
    max_log = np.log(max_value)
    max_amplitude_tensor = tf.reduce_max(batch_of_amplitudes)
    log_max = tf.log(max_amplitude_tensor)
    update_norm = lambda: tf.assign_add(self._exp_norm_shift, log_max - max_log)
    keep_norm = lambda: tf.assign_add(self._exp_norm_shift, 0.)
    normalize = tf.cond(tf.greater(log_max, max_log), update_norm, keep_norm)
    return normalize

  @classmethod
  def from_hparams(
      cls,
      hparams: tf.contrib.training.HParams,
      name: str = ''
  ) -> 'Wavefunction':
    """Constructs an instance of a class from hparams."""
    raise NotImplementedError


def module_transfer_ops(
    source_module: Wavefunction,
    target_module: Wavefunction,
) -> tf.Tensor:
  """Builds operations that copy variables values from source to target.

  Args:
    source_module: Wavefunction whose variables are being copied to target.
    target_module: Wavefunction whose variables are assigned. Must be a deep
        copy of source_module.

  Returns:
    A Tensor operation that assigns all variables of target to be equal to
    those of source.

  Raises:
    ValueError: `target_module` is does not have the same structure as srouce.
  """
  source_variables = source_module.get_trainable_variables()
  target_variables = target_module.get_trainable_variables()
  assign_ops = []
  for source_var, target_var in zip(source_variables, target_variables):
    assign_ops.append(tf.assign(target_var, source_var))
  if hasattr(target_module, 'norm'):
    assign_ops.append(tf.assign(target_module.norm, source_module.norm))
  return tf.group(*assign_ops)


class FullyConnectedNetwork(Wavefunction):
  """Implementation of a wavefunction as fully connected neural network."""

  def __init__(
      self,
      num_layers: int,
      layer_size: int,
      nonlinearity: tf.Tensor = tf.nn.relu,
      output_activation: tf.Tensor = tf.exp,
      name: str = 'fully_connected_network'
  ):
    """Creates an instance of a class."""
    super(FullyConnectedNetwork, self).__init__(name=name)
    self._num_layers = num_layers
    self._layer_size = layer_size
    self._nonlinearity = nonlinearity
    self._output_activation = output_activation
    with self._enter_variable_scope():
      self._components = []
      for _ in range(num_layers):
        self._components += [snt.Linear(output_size=layer_size), nonlinearity]
      self._components += [snt.Linear(output_size=1), tf.squeeze]
      if output_activation == tf.exp:
        self._components += [self.add_exp_normalization, tf.exp]
      else:
        self._components += [output_activation]

  def _build(
      self,
      inputs: tf.Tensor,
  ) -> tf.Tensor:
    """Builds computational graph evaluating the wavefunction on inputs.

    Args:
      inputs: Input tensor, must have shape (batch, num_sites, ...).

    Returns:
      Tensor holding values of the wavefunction on `inputs`.

    Raises:
      ValueError: Input tensor has wrong shape.
    """
    module = snt.Sequential(self._components)
    return module(inputs)

  @classmethod
  def from_hparams(
      cls,
      hparams: tf.contrib.training.HParams,
      name: str = ''
  ) -> 'Wavefunction':
    """Constructs an instance of a class from hparams."""
    fcnn_params = {
        'num_layers': hparams.num_fc_layers,
        'layer_size': hparams.fc_layer_size,
        'output_activation': layers.NONLINEARITIES[hparams.output_activation],
        'nonlinearity': layers.NONLINEARITIES[hparams.nonlinearity],
    }
    if name:
      fcnn_params['name'] = name
    return cls(**fcnn_params)


class RestrictedBoltzmannNetwork(Wavefunction):
  """Implementation of an extended Restricted Boltzmann machine.

  This wavefunction implements RBM wavefunction whose inputs are modified by
  a feed forward neural network with nonlinearities.
  """

  def __init__(
      self,
      num_layers: int,
      layer_size: int,
      nonlinearity: tf.Tensor = tf.nn.relu,
      name: str = 'restricted_boltzmann_network'
  ):
    """Creates an instance of a class."""
    super(RestrictedBoltzmannNetwork, self).__init__(name=name)
    self._num_layers = num_layers
    self._layer_size = layer_size
    self._nonlinearity = nonlinearity
    reduction = functools.partial(tf.reduce_sum, axis=1)
    with self._enter_variable_scope():
      self._components = []
      for _ in range(num_layers):
        self._components += [snt.Linear(output_size=layer_size), nonlinearity]
      self._components += [snt.Linear(layer_size), tf.cosh, tf.log]
      self._components += [reduction, self.add_exp_normalization]
      self._onsite_layer = snt.Linear(output_size=1)

  def _build(
      self,
      inputs: tf.Tensor,
  ) -> tf.Tensor:
    """Builds computational graph evaluating the wavefunction on inputs.

    Args:
      inputs: Input tensor, must have shape (batch, num_sites, ...).

    Returns:
      Tensor holding values of the wavefunction on `inputs`.

    Raises:
      ValueError: Input tensor has wrong shape.
    """
    onsite_component = tf.squeeze(self._onsite_layer(inputs))
    cosh_component = snt.Sequential(self._components)(inputs)
    return tf.exp(onsite_component + cosh_component)

  @classmethod
  def from_hparams(
      cls,
      hparams: tf.contrib.training.HParams,
      name: str = ''
  ) -> 'Wavefunction':
    """Constructs an instance of a class from hparams."""
    rbm_params = {
        'num_layers': hparams.num_fc_layers,
        'layer_size': hparams.fc_layer_size,
        'nonlinearity': layers.NONLINEARITIES[hparams.nonlinearity],
    }
    if name:
      rbm_params['name'] = name
    return cls(**rbm_params)

class Conv1DNetwork(Wavefunction):
  """Implementation of wavefunction as convolutional neural network."""

  def __init__(
      self,
      num_layers: int,
      num_filters: int,
      kernel_size: int,
      nonlinearity: tf.Tensor = tf.nn.relu,
      output_activation: tf.Tensor = tf.exp,
      name: str = 'conv_1d_network',
  ):
    """Creates an instance of a class.

    Args:
      num_layers: Number of convolutional layers.
      num_filters: Number of convolutional filters in each layer.
      kernel_size: Size of convolutional kernels.
      nonlinearity: Nonlinearity to use between hidden layers.
      output_activation: Wavefunction amplitude activation function.
      name: Name of the wave-function.
    """
    super(Conv1DNetwork, self).__init__(name=name)
    self._num_layers = num_layers
    self._num_filters = num_filters
    self._kernel_size = kernel_size
    self._nonlinearity = nonlinearity
    self._output_activation = output_activation

    reduction = functools.partial(tf.reduce_sum, axis=[1, 2])
    self._components = []
    with self._enter_variable_scope():
      for layer in range(num_layers):
        self._components.append(layers.Conv1dPeriodic(num_filters, kernel_size))
        if layer + 1 != num_layers:
          self._components.append(nonlinearity)
      if output_activation == tf.exp:
        self._components += [reduction, self.add_exp_normalization, tf.exp]
      else:
        self._components += [reduction, output_activation]

  def _build(
      self,
      inputs: tf.Tensor,
  ) -> tf.Tensor:
    """Builds computational graph evaluating the wavefunction on inputs.

    Args:
      inputs: Input tensor, must have shape (batch, num_sites, ...).

    Returns:
      Tensor holding values of the wavefunction on `inputs`.

    Raises:
      ValueError: Input tensor has wrong shape.
    """
    return snt.Sequential(self._components)(tf.expand_dims(inputs, 2))

  @classmethod
  def from_hparams(
      cls,
      hparams: tf.contrib.training.HParams,
      name: str = ''
  ) -> 'Wavefunction':
    """Constructs an instance of a class from hparams."""
    conv_1d_params = {
        'num_layers': hparams.num_conv_layers,
        'num_filters': hparams.num_conv_filters,
        'kernel_size': hparams.kernel_size,
        'output_activation': layers.NONLINEARITIES[hparams.output_activation],
        'nonlinearity': layers.NONLINEARITIES[hparams.nonlinearity],
    }
    if name:
      conv_1d_params['name'] = name
    return cls(**conv_1d_params)


class Conv2DNetwork(Wavefunction):
  """Implementation of wavefunction as convolutional neural network."""

  def __init__(
      self,
      num_layers: int,
      num_filters: int,
      kernel_size: int,
      size_x: int,
      size_y: int,
      nonlinearity: tf.Tensor = tf.nn.relu,
      output_activation: tf.Tensor = tf.exp,
      name: str = 'conv_2d_network'
  ):
    """Creates an instance of a class.

    Args:
      num_layers: Number of convolutional layers.
      num_filters: Number of convolutional filters in each layer.
      kernel_size: Size of convolutional kernels.
      size_x: Number of sites along x direction of the system.
      size_y: Number of sites along y direction of the system.
      nonlinearity: Nonlinearity to use between hidden layers.
      output_activation: Wavefunction amplitude activation function.
      name: Name of the wave-function.
    """
    super(Conv2DNetwork, self).__init__(name=name)
    self._num_layers = num_layers
    self._num_filters = num_filters
    self._kernel_size = kernel_size
    self._nonlinearity = nonlinearity
    self._output_activation = output_activation
    self._size_x = size_x
    self._size_y = size_y

    reduction = functools.partial(tf.reduce_sum, axis=[1, 2, 3])
    self._components = []
    with self._enter_variable_scope():
      for layer in range(num_layers):
        self._components.append(layers.Conv2dPeriodic(num_filters, kernel_size))
        if layer + 1 != num_layers:
          self._components.append(nonlinearity)
      if output_activation == tf.exp:
        self._components += [reduction, self.add_exp_normalization, tf.exp]
      else:
        self._components += [reduction, output_activation]

  def _build(
      self,
      inputs: tf.Tensor,
  ) -> tf.Tensor:
    """Builds computational graph evaluating the wavefunction on inputs.

    Args:
      inputs: Input tensor, must have shape (batch, num_sites, ...).

    Returns:
      Tensor holding values of the wavefunction on `inputs`.

    Raises:
      ValueError: Input tensor has wrong shape.
    """
    inputs_new_shape = [-1, self._size_x, self._size_y, 1]
    inputs = tf.reshape(inputs, shape=inputs_new_shape)
    return snt.Sequential(self._components)(inputs)

  @classmethod
  def from_hparams(
      cls,
      hparams: tf.contrib.training.HParams,
      name: str = ''
  ) -> 'Wavefunction':
    """Constructs an instance of a class from hparams."""
    conv_2d_params = {
        'num_layers': hparams.num_conv_layers,
        'num_filters': hparams.num_conv_filters,
        'kernel_size': hparams.kernel_size,
        'size_x': hparams.size_x,
        'size_y': hparams.size_y,
        'output_activation': layers.NONLINEARITIES[hparams.output_activation],
        'nonlinearity': layers.NONLINEARITIES[hparams.nonlinearity],
    }
    if name:
      conv_2d_params['name'] = name
    return cls(**conv_2d_params)

class ResNet1D(Wavefunction):
  """Implementation of wavefunction as residual neural network.

  This network borrows the basis from the Residual networks used for image
  classification. In this version we maintain constant filter representation."""

  def __init__(
      self,
      num_blocks: int,
      num_filters: int,
      kernel_size: int,
      conv_stride: int,
      bottleneck: bool = False,
      output_activation: tf.Tensor = tf.exp,
      name: str = 'res_net_1d'
  ):
    """Creates an instance of a class.

    Args:
      num_blocks: Number of ResNet blocks.
      num_filters: Number of convolutional filters in ResNet blocks.
      kernel_size: Size of convolutional kernels.
      conv_stride: Strides in ResNet blocks.
      bottleneck: Use regular blocks or bottleneck blocks.
      output_activation: Activation function on the final output.
      name: Name of the wave-function.
    """
    super(ResNet1D, self).__init__(name=name)
    self._num_blocks = num_blocks
    self._num_filters = num_filters
    self._kernel_size = kernel_size
    self._conv_stride = conv_stride
    self._bottleneck = bottleneck
    self._output_activation = output_activation

    reduction = functools.partial(tf.reduce_sum, axis=[1, 2])
    block_args = {
        'num_filters': num_filters,
        'kernel_shape': kernel_size,
        'conv_stride': conv_stride,
    }
    res_blocks = []

    with self._enter_variable_scope():
      initial_conv = layers.Conv1dPeriodic(num_filters, kernel_size)
      for _ in range(num_blocks):
        if bottleneck:
          res_blocks.append(layers.BottleneckResBlock1d(**block_args))
        else:
          res_blocks.append(layers.ResBlock1d(**block_args))
      self._components = [initial_conv,] + res_blocks
      if output_activation == tf.exp:
        self._components += [reduction, self.add_exp_normalization, tf.exp]
      else:
        self._components += [reduction, output_activation]


  def _build(
      self,
      inputs: tf.Tensor,
  ) -> tf.Tensor:
    """Builds computational graph evaluating the wavefunction on inputs.

    Args:
      inputs: Input tensor, must have shape (batch, num_sites, ...).

    Returns:
      Tensor holding values of the wavefunction on `inputs`.

    Raises:
      ValueError: Input tensor has wrong shape.
    """
    return snt.Sequential(self._components)(tf.expand_dims(inputs, 2))

  @classmethod
  def from_hparams(
      cls,
      hparams: tf.contrib.training.HParams,
      name: str = ''
  ) -> 'Wavefunction':
    """Constructs an instance of a class from hparams."""
    res_net_1d_params = {
        'num_blocks': hparams.num_resnet_blocks,
        'num_filters': hparams.num_conv_filters,
        'kernel_size': hparams.kernel_size,
        'conv_stride': hparams.conv_strides,
        'output_activation': layers.NONLINEARITIES[hparams.output_activation],
    }
    if name:
      res_net_1d_params['name'] = name
    return cls(**res_net_1d_params)

# pylint: disable=too-many-instance-attributes
class ResNet2D(Wavefunction):
  """Implementation of wavefunction as residual neural network in 2D.

  This network borrows the basis from the Residual networks used for image
  classification. In this version we maintain constant filter size.
  """
  # pylint: disable=too-many-arguments
  def __init__(
      self,
      num_blocks: int,
      num_filters: int,
      kernel_size: int,
      conv_stride: int,
      size_x: int,
      size_y: int,
      bottleneck: bool = False,
      output_activation: tf.Tensor = tf.exp,
      name: str = 'res_net_2d'
  ):
    """Creates an instance of a class.

    Args:
      num_blocks: Number of ResNet blocks.
      num_filters: Number of convolutional filters in ResNet blocks.
      kernel_size: Size of convolutional kernels.
      conv_stride: Strides in ResNet blocks.
      size_x: Number of sites along x direction of the system.
      size_y: Number of sites along y direction of the system.
      bottleneck: Use regular blocks or bottleneck blocks.
      output_activation: Activation function on the final output.
      name: Name of the wave-function.
    """
    super(ResNet2D, self).__init__(name=name)
    self._num_blocks = num_blocks
    self._num_filters = num_filters
    self._kernel_size = kernel_size
    self._conv_stride = conv_stride
    self._size_x = size_x
    self._size_y = size_y
    self._bottleneck = bottleneck
    self._output_activation = output_activation

    reduction = functools.partial(tf.reduce_sum, axis=[1, 2, 3])
    res_blocks = []
    block_args = {
        'num_filters': num_filters,
        'kernel_shape': kernel_size,
        'conv_stride': conv_stride,
    }
    with self._enter_variable_scope():
      initial_conv = layers.Conv2dPeriodic(num_filters, kernel_size)
      for _ in range(num_blocks):
        if bottleneck:
          res_blocks.append(layers.BottleneckResBlock2d(**block_args))
        else:
          res_blocks.append(layers.ResBlock2d(**block_args))
      self._components = [initial_conv,] + res_blocks
      if output_activation == tf.exp:
        self._components += [reduction, self.add_exp_normalization, tf.exp]
      else:
        self._components += [reduction, output_activation]

  def _build(
      self,
      inputs: tf.Tensor,
  ) -> tf.Tensor:
    """Builds computational graph evaluating the wavefunction on inputs.

    Args:
      inputs: Input tensor, must have shape (batch, size_x, size_y, ...).

    Returns:
      Tensor holding values of the wavefunction on `inputs`.

    Raises:
      ValueError: Input tensor has wrong shape.
    """
    inputs_new_shape = [-1, self._size_x, self._size_y, 1]
    inputs = tf.reshape(inputs, shape=inputs_new_shape)
    return snt.Sequential(self._components)(inputs)

  @classmethod
  def from_hparams(
      cls,
      hparams: tf.contrib.training.HParams,
      name: str = ''
  ) -> 'Wavefunction':
    """Constructs an instance of a class from hparams."""
    res_net_2d_params = {
        'num_blocks': hparams.num_resnet_blocks,
        'num_filters': hparams.num_conv_filters,
        'kernel_size': hparams.kernel_size,
        'conv_stride': hparams.conv_strides,
        'size_x': hparams.size_x,
        'size_y': hparams.size_y,
        'output_activation': layers.NONLINEARITIES[hparams.output_activation],
    }
    if name:
      res_net_2d_params['name'] = name
    return cls(**res_net_2d_params)


class MatrixProductState(Wavefunction):
  """Implementation of a wavefunction as matrix product state.

  Current implementation seems to be extremely memory inefficient, possibly
  due to bad contraction order.
  TODO(kochkov92) Fix this.
  """

  def __init__(
      self,
      num_sites: int,
      bond_dimension: int,
      name: str = 'matrix_product_state'
  ):
    """Creates an instance of a class."""
    super(MatrixProductState, self).__init__(name=name)
    self._num_sites = num_sites
    self._bond_dimension = bond_dimension
    self._name = name

    with self._enter_variable_scope():
      self._mp_units = [layers.MatrixProductUnit(1, bond_dimension)]
      for _ in range(self._num_sites - 2):
        self._mp_units.append(
            layers.MatrixProductUnit(bond_dimension, bond_dimension))
      self._mp_units.append(layers.MatrixProductUnit(bond_dimension, 1))

  def _build(
      self,
      inputs: tf.Tensor,
  ) -> tf.Tensor:
    """Connects MatrixProductState module into the graph with input `inputs.

    Args:
      inputs: Input configurations on which to evaluate a wavefunction.

    Returns:
      Tensor representing the probability amplitude of the wavefunction.
    """
    site_inputs = tf.unstack(inputs, axis=1)
    units_and_inputs = zip(self._mp_units, site_inputs)
    mps_matrices = [mpu(site_input) for mpu, site_input in units_and_inputs]

    result = tf.constant(1., shape=[inputs.shape[0], 1, 1], dtype=tf.float32)
    for mpu in mps_matrices:
      result = tf.einsum('bij,bjk->bik', result, mpu)
    return tf.squeeze(result)

  @classmethod
  def from_hparams(
      cls,
      hparams: tf.contrib.training.HParams,
      name: str = ''
  ) -> 'Wavefunction':
    """Constructs an instance of a class from hparams."""
    mps_params = {
        'num_sites': hparams.num_sites,
        'bond_dimension': hparams.bond_dimension,
    }
    if name:
      mps_params['name'] = name
    return cls(**mps_params)


class ProjectedBDG(Wavefunction):
  """P-BDG module."""

  def __init__(
      self,
      num_sites: int,
      name: str = 'projected_bdg'):
    """Constructs a projected BDG module.

    Args:
      num_sites: Number of sites.
      name: Name of the module.
    """
    super(ProjectedBDG, self).__init__(name=name)
    self._num_sites = num_sites
    with self._enter_variable_scope():
      self._pairing_matrix = tf.get_variable(
          'pairing_matrix', shape=[1, num_sites, num_sites], dtype=tf.float32)

  def _build(self, inputs: tf.Tensor) -> tf.Tensor:
    """Connects the P-BDG module into the graph with input `inputs`.

    Args:
      inputs: Tensor with input values of shape=[batch] and values +/- 1.

    Returns:
      Wave-function amplitudes of shape=[batch].
    """
    batch_size = inputs.shape[0]
    n_sites = self._num_sites
    mask = tf.einsum('ij,ik->ijk', tf.nn.relu(inputs), tf.nn.relu(-inputs))
    bool_mask = tf.greater(mask, tf.zeros([batch_size, n_sites, n_sites]))
    tiled_pairing = tf.tile(self._pairing_matrix, [batch_size, 1, 1])
    det_size = [batch_size, n_sites // 2, n_sites // 2]
    pre_det = tf.reshape(tf.boolean_mask(tiled_pairing, bool_mask), det_size)

    sign, ldet = tf.linalg.slogdet(pre_det)
    det_value = tf.exp(self.add_exp_normalization(ldet))
    return sign * det_value

  @classmethod
  def from_hparams(
      cls,
      hparams: tf.contrib.training.HParams,
      name: str = ''
  ) -> 'Wavefunction':
    """Constructs an instance of a class from hparams."""
    pbdg_params = {
        'num_sites': hparams.num_sites,
    }
    if name:
      pbdg_params['name'] = name
    return cls(**pbdg_params)


class FullyConnectedNNB(Wavefunction):
  """BCS neural network backflow module."""

  def __init__(
      self,
      num_sites: int,
      num_layers: int,
      layer_sizes: List[int],
      name: str = 'fully_connected_nnb'):
    """Constructs a neural net backflow module.

    Args:
      num_sites: Number of sites in the system.
      num_layers: Number of neural networks.
      layer_sizes: Sizes of fully connected networks.
      name: Name of the module.
    """
    super(FullyConnectedNNB, self).__init__(name=name)
    self._num_sites = num_sites
    self._num_layers = num_layers
    self._layer_sizes = layer_sizes

    nonlinearity = tf.nn.relu

    self._components = []
    with self._enter_variable_scope():
      for _, layer_size in zip(range(num_layers), layer_sizes):
        self._components += [snt.Linear(output_size=layer_size), nonlinearity]
      self._components += [snt.Linear(output_size=num_sites * num_sites)]

  def _build(self, inputs: tf.Tensor) -> tf.Tensor:
    """Connects the FC-NNB module into the graph with input `inputs`.

    Args:
      inputs: Tensor with input values of shape=[batch] and values +/- 1.

    Returns:
      Wave-function amplitudes of shape=[batch].
    """
    batch_size = inputs.shape[0]
    n_sites = self._num_sites

    pairing_shape = [batch_size, n_sites, n_sites]
    pairing = snt.Sequential(self._components)(inputs)
    pairing = tf.reshape(pairing, pairing_shape)

    mask = tf.einsum('ij,ik->ijk', tf.nn.relu(inputs), tf.nn.relu(-inputs))
    mask = tf.greater(mask, tf.zeros([batch_size, n_sites, n_sites]))
    det_size = [batch_size, n_sites // 2, n_sites // 2]
    pre_determinant = tf.reshape(tf.boolean_mask(pairing, mask), det_size)
    return tf.linalg.det(pre_determinant)

  @classmethod
  def from_hparams(
      cls,
      hparams: tf.contrib.training.HParams,
      name: str = ''
  ) -> 'Wavefunction':
    """Constructs an instance of a class from hparams."""
    num_layers = hparams.num_fc_layers
    fcnnb_params = {
        'num_sites': hparams.num_sites,
        'num_layers': num_layers,
        'layer_sizes': [hparams.fc_layer_size for _ in range(num_layers)]
    }
    if name:
      fcnnb_params['name'] = name
    return cls(**fcnnb_params)


class FullVector(Wavefunction):
  """Implementation of a wavefunction as ED vector in fixed Sz sector."""

  def __init__(
      self,
      num_sites: int,
      top_lin_table: np.array,
      bot_lin_table: np.array,
      initial_vector: np.array,
      name: str = 'full_vector'):
    """Constructs a full ED-like vector module.

    Args:
      num_sites: Number of sites in the system.
      top_lin_table: Higher bits index table, see "Lin, H. Q. 1990."
      bot_lin_table: Lower bits index table.
      initial_vector: Initial values of the psi vector.
      name: Name of the module.
    """
    super(FullVector, self).__init__(name=name)
    self._num_sites = num_sites
    self._top_lin_table = top_lin_table
    self._bot_lin_table = bot_lin_table
    self._initial_vector = initial_vector
    self._lin_mask = np.array(
        [2 ** (i) for i in range(0, num_sites // 2)],
        dtype=int,
    )
    with self._enter_variable_scope():
      self._ed_vector = tf.get_variable(
          name='ed_vector',
          initializer=initial_vector,
          trainable=True,
      )

  def _build(self, inputs: tf.Tensor) -> tf.Tensor:
    """Connects the FullVector module into the graph with input `inputs`.

    Args:
      inputs: Tensor with input values of shape=[batch] and values +/- 1.

    Returns:
      Wave-function amplitudes of shape=[batch].
    """
    n_sites = self._num_sites
    top_bits = tf.cast(inputs[:, (n_sites // 2):], tf.int32)
    bot_bits = tf.cast(inputs[:, :(n_sites // 2)], tf.int32)

    top_indices = tf.reduce_sum(tf.nn.relu(top_bits) * self._lin_mask, axis=1)
    bot_indices = tf.reduce_sum(tf.nn.relu(bot_bits) * self._lin_mask, axis=1)
    top_lin_indices = tf.gather(self._top_lin_table, top_indices)
    bot_lin_indices = tf.gather(self._bot_lin_table, bot_indices)

    ed_indices = top_lin_indices + bot_lin_indices
    return tf.gather(self._ed_vector, ed_indices)

  @classmethod
  def from_hparams(
      cls,
      hparams: tf.contrib.training.HParams,
      name: str = ''
  ) -> 'Wavefunction':
    """Constructs an instance of a class from hparams."""
    dir_path = hparams.checkpoint_dir
    top_lin_table_path = os.path.join(dir_path, hparams.top_lin_table_file)
    bot_lin_table_path = os.path.join(dir_path, hparams.bot_lin_table_file)
    ed_vector_path = os.path.join(dir_path, hparams.ed_vector_file)
    top_lin_table = np.genfromtxt(top_lin_table_path, dtype=np.int32)
    bot_lin_table = np.genfromtxt(bot_lin_table_path, dtype=np.int32)
    ed_vector = np.genfromtxt(ed_vector_path, dtype=np.float32)

    full_vector_params = {
        'num_sites': hparams.num_sites,
        'top_lin_table': top_lin_table,
        'bot_lin_table': bot_lin_table,
        'initial_vector': ed_vector,
    }
    if name:
      full_vector_params['name'] = name
    return cls(**full_vector_params)


class GNN(Wavefunction):
  """Implementation of wavefunction as graph neural network."""

  def __init__(
      self,
      num_layers: int,
      num_filters: int,
      adj: np.array,
      nonlinearity: tf.Tensor = tf.nn.relu,
      output_activation: tf.Tensor = tf.exp,
      name: str = 'graph_neural_network',
  ):
    """Creates an instance of a class.

    Args:
      num_layers: Number of convolutional layers.
      num_filters: Number of convolutional filters in each layer.
      adj: adjacency list of the graph.
      nonlinearity: Nonlinearity to use between hidden layers.
      output_activation: Wavefunction amplitude activation function.
      name: Name of the wave-function.
    """
    super(GNN, self).__init__(name=name)
    self._num_layers = num_layers
    self._num_filters = num_filters
    self._adj = adj
    self._nonlinearity = nonlinearity
    self._output_activation = output_activation

    reduction = functools.partial(tf.reduce_sum, axis=[1, 2])
    self._components = []
    with self._enter_variable_scope():
      for layer in range(num_layers):
        self._components.append(layers.GraphConvLayer(num_filters, adj))
        if layer + 1 != num_layers:
          self._components.append(nonlinearity)
      if output_activation == tf.exp:
        self._components += [reduction, self.add_exp_normalization, tf.exp]
      else:
        self._components += [reduction, output_activation]

  def _build(
      self,
      inputs: tf.Tensor,
  ) -> tf.Tensor:
    """Builds computational graph evaluating the wavefunction on inputs.

    Args:
      inputs: Input tensor, must have shape (batch, num_sites, ...).

    Returns:
      Tensor holding values of the wavefunction on `inputs`.

    Raises:
      ValueError: Input tensor has wrong shape.
    """
    return snt.Sequential(self._components)(tf.expand_dims(inputs, 2))

  @classmethod
  def from_hparams(
      cls,
      hparams: tf.contrib.training.HParams,
      name: str = ''
  ) -> 'Wavefunction':
    """Constructs an instance of a class from hparams."""
    gnn_params = {
        'num_layers': hparams.num_conv_layers,
        'num_filters': hparams.num_conv_filters,
        'adj': np.genfromtxt(hparams.adj_list, dtype=int),
        'output_activation': layers.NONLINEARITIES[hparams.output_activation],
        'nonlinearity': layers.NONLINEARITIES[hparams.nonlinearity],
    }
    if name:
      gnn_params['name'] = name
    return cls(**gnn)




def build_wavefunction(
    hparams: tf.contrib.training.HParams,
) -> 'Wavefunction':
  """Returns a Wavefunction object based on the requested type and hparams.

  Creates a Wavefunction instance corresponding to `wavefunction_type` and uses
  parameters from hparams to initialize it.

  Args:
    hparams: Class holding hyperparameters of the wavefunction ansatzs.

  Returns:
    A Wavefunction instance corresponding to `wavefunction_type`.

  Raises:
    ValueError: Provided `wavefunction_type` is not registered.
  """
  wavefunction_type = hparams.wavefunction_type
  if wavefunction_type in WAVEFUNCTION_TYPES:
    return WAVEFUNCTION_TYPES[wavefunction_type].from_hparams(hparams)

  if hparams.wavefunction_type == 'fc_sum':
    wavefunction_type = 'fully_connected'
    fcnn_1 = WAVEFUNCTION_TYPES[wavefunction_type].from_hparams(hparams, 'fc_1')
    fcnn_2 = WAVEFUNCTION_TYPES[wavefunction_type].from_hparams(hparams, 'fc_2')
    return fcnn_1 + fcnn_2

  if hparams.wavefunction_type == 'rbm_sum':
    wavefunction_type = 'rbm'
    rbm_1 = WAVEFUNCTION_TYPES[wavefunction_type].from_hparams(hparams, 'rbm_1')
    rbm_2 = WAVEFUNCTION_TYPES[wavefunction_type].from_hparams(hparams, 'rbm_2')
    return rbm_1 + rbm_2

  if hparams.wavefunction_type == 'fc_diff':
    wavefunction_type = 'fully_connected'
    fcnn_1 = WAVEFUNCTION_TYPES[wavefunction_type].from_hparams(hparams, 'fc_1')
    fcnn_2 = WAVEFUNCTION_TYPES[wavefunction_type].from_hparams(hparams, 'fc_2')
    return fcnn_1 - fcnn_2

  if hparams.wavefunction_type == 'rbm_diff':
    wavefunction_type = 'rbm'
    rbm_1 = WAVEFUNCTION_TYPES[wavefunction_type].from_hparams(hparams, 'rbm_1')
    rbm_2 = WAVEFUNCTION_TYPES[wavefunction_type].from_hparams(hparams, 'rbm_2')
    return rbm_1 - rbm_2

  raise ValueError('Provided wavefunction_type is not registered.')


WAVEFUNCTION_TYPES = {
    'fully_connected': FullyConnectedNetwork,
    'rbm': RestrictedBoltzmannNetwork,
    'conv_1d': Conv1DNetwork,
    'conv_2d': Conv2DNetwork,
    'mps': MatrixProductState,
    'pbdg': ProjectedBDG,
    'fully_connected_nnb': FullyConnectedNNB,
    'res_net_1d': ResNet1D,
    'res_net_2d': ResNet2D,
    'ed_vector': FullVector,
    'gnn': GNN,
}
