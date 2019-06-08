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

  def complex(self, other: 'Wavefunction') -> 'Wavefunction':
    """Generates new wavefunction that equals to `self` + `other`."""

    class ComplexWavefunctions(Wavefunction):
      """Wavefunction wrapper of the ampltitude-angle representation of a complex wavefunction."""
      def __init__(
          self,
          wf_a: Wavefunction,
          wf_b: Wavefunction,
          name: str = 'complex_wavefunctions'
      ):
        """Creates an WavefunctionSum module."""
        # pylint: disable=protected-access
        name = '_plus_'.join([wf_a._unique_name, wf_b._unique_name])
        super(ComplexWavefunctions, self).__init__(name=name)
        self._wf_a = wf_a
        self._wf_b = wf_b
        self._sub_wavefunctions += [self._wf_a, self._wf_b]

      def _build(self, inputs: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """Builds computational graph evaluating the wavefunction on inputs.

        Args:
          inputs: Input tensor, must have shape (batch, num_sites, ...).

        Returns:
          Tensor holding values of the wavefunction on `inputs`.

        Raises:
          ValueError: Input tensor has wrong shape.
        """
        return (self._wf_a(inputs), self._wf_b(inputs))

      @classmethod
      def from_hparams(cls, hparams: tf.contrib.training.HParams):
        """Constructs an instance of a class from hparams."""
        raise ValueError('Hparams initialization is not supported for sum.')

    return ComplexWavefunctions(self, other)

  def get_trainable_variables(self):
    """Returns a list of trainable variables in this wavefunction."""
    trainable_variables = []
    # pylint: disable=protected-access
    trainable_variables += tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._unique_name + '/')
    for sub_wavefunction in self._sub_wavefunctions:
      trainable_variables += sub_wavefunction.get_trainable_variables()
    return trainable_variables

  def get_trainable_sub_variables(self):
    """Returns a list of trainable variables in this wavefunction."""
    trainable_variables = []
    # pylint: disable=protected-access
    for sub_wavefunction in self._sub_wavefunctions:
      trainable_variables.append(sub_wavefunction.get_trainable_variables())
    return trainable_variables

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
      output_activation: tf.Tensor = tf.identity,
      name: str = 'fully_connected_network'
  ):
    """Creates an instance of a class."""
    super(FullyConnectedNetwork, self).__init__(name=name)
    self._num_layers = num_layers
    self._layer_size = layer_size
    self._nonlinearity = nonlinearity
    self._output_activation = output_activation
    with self._enter_variable_scope():
      #self._amplitude_components = []
      #self._angle_components = []
      #for _ in range(num_layers):
      #  self._amplitude_components += [snt.Linear(output_size=layer_size), nonlinearity]
      #  self._angle_components += [snt.Linear(output_size=layer_size), nonlinearity]
      #self._amplitude_components += [snt.Linear(output_size=1), tf.squeeze]
      #self._angle_components += [snt.Linear(output_size=1), tf.squeeze]
      #self._amplitude_components += [output_activation]
      #self._angle_components += [output_activation]
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

    #amplitude_module = snt.Sequential(self._amplitude_components, name='amplitude')
    #angle_module = snt.Sequential(self._angle_components, name='angle')
    #return (amplitude_module(inputs), angle_module(inputs))
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

  if hparams.wavefunction_type in ('complex'):
    wf_type_a, wf_type_b = hparams.composite_wavefunction_types
    activation_a, activation_b = hparams.composite_output_activations
    wf_a_hparams = copy.copy(hparams)
    wf_b_hparams = copy.copy(hparams)
    wf_a_hparams.set_hparam('output_activation', activation_a)
    wf_a_hparams.set_hparam('wavefunction_type', wf_type_a)
    wf_b_hparams.set_hparam('output_activation', activation_b)
    wf_b_hparams.set_hparam('wavefunction_type', wf_type_b)
    wf_a = WAVEFUNCTION_TYPES[wf_type_a].from_hparams(wf_a_hparams)
    wf_b = WAVEFUNCTION_TYPES[wf_type_b].from_hparams(wf_b_hparams)
    if hparams.wavefunction_type == 'complex':
      return Wavefunction.complex(wf_a, wf_b)

  raise ValueError('Provided wavefunction_type is not registered.')


WAVEFUNCTION_TYPES = {
    'fully_connected': FullyConnectedNetwork,
}
