"""Defines wavefunction inteface and implemens various wavefunction networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import copy
import functools
import inspect

from typing import Dict, List, Any

import os
import numpy as np
import tensorflow.keras as keras
import layers



class Wavefunction(keras.Model):

  def __init__(self, name: str = 'wavefunction'):
    """Creates a Wavefunction instance"""
    super(Wavefunction, self).__init__(name=name)

  def call(self, inputs: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """Builds computational graph evaluating the wavefunction on inputs.

    Args:
      inputs: Input tensor, must have shape (batch, num_sites, ...).

    Returns:
      Tensor holding values of the wavefunction on `inputs`.

    Raises:
      ValueError: Input tensor has wrong shape.
    """
    raise NotImplementedError


  def get_trainable_variables(self):
    """Returns a list of trainable variables in this wavefunction."""
    #trainable_variables = []
    ## pylint: disable=protected-access
    #trainable_variables += self.get_weights()
    trainable_variables = self.get_weights()
    return trainable_variables


  @classmethod
  def from_hparams(
      cls,
      hparams: tf.contrib.training.HParams,
      name: str = ''
  ) -> 'Wavefunction':
    """Constructs an instance of a class from hparams."""
    raise NotImplementedError



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

    #with self._enter_variable_scope():
    self._components = []
    for _ in range(num_layers):
      self._components += [keras.layers.Dense(units=layer_size, activation=nonlinearity)]

    self._amplitude = keras.layers.Dense(1, activation=tf.identity)
    self._angle = keras.layers.Dense(1, activation='sigmoid')

  def call(
      self,
      inputs: tf.Tensor,
  ) -> (tf.Tensor, tf.Tensor):
    """Builds computational graph evaluating the wavefunction on inputs.

    Args:
      inputs: Input tensor, must have shape (batch, num_sites, ...).

    Returns:
      Tensor holding values of the wavefunction on `inputs`.

    Raises:
      ValueError: Input tensor has wrong shape.
    """
    #wf_input = keras.Input(shape=inputs.shape)
    #x = self._components[0](wf_input)
    x = self._components[0](inputs)
    for i in range(len(self._components)-1):
      x = self._components[i+1](x)
    amplitude = tf.squeeze(self._amplitude(x))
    angle = tf.squeeze(np.pi*self._angle(x))

    #self._model = keras.Model(inputs=wf_input, outputs=[amplitude, angle])

    #return self._model(inputs)
    return [amplitude, angle]



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

  raise ValueError('Provided wavefunction_type is not registered.')


WAVEFUNCTION_TYPES = {
    'fully_connected': FullyConnectedNetwork,
}


