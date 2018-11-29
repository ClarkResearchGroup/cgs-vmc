"""Useful utilities for computational graph and components construction."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Tuple

import enum
import numpy as np
import tensorflow as tf

import wavefunctions
import utils


class ResourceName(enum.Enum):
  """Type of sharable resources, serves as key in `shared_resources`."""
  CONFIGS = 'CONFIGS'
  TARGET_PSI = 'TARGET_PSI'
  TRAINING_PSI = 'TRAINING_PSI'
  MONTE_CARLO_SAMPLING = 'MONTE_CARLO_SAMPLING'


def get_or_create_num_epochs() -> tf.Tensor:
  """Returns a variable representing number of optimization epochs."""
  with tf.variable_scope("", reuse=tf.AUTO_REUSE):
    num_epochs = tf.get_variable(
        name='num_epochs',
        shape=[],
        initializer=tf.zeros_initializer(dtype=tf.int32),
        dtype=tf.int32,
        trainable=False
    )
    return num_epochs


def build_monte_carlo_sampling(
    inputs: tf.Tensor,
    wavefunction: wavefunctions.Wavefunction,
    psi: tf.Tensor = None
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Connects operations that perform MCMC sampling in `inupts` to the graph.

  Args:
    inputs: Variable holding sampled system configurations.
    wavefunction: Wavefunction object to use for important sampling.
    psi: Tensor representing wavefunction(inputs), or None.

  Returns:
    A tuple of tensors representing operation for updating inputs and evaluating
    acceptance ratio.
  """
  if psi is None:
    psi = wavefunction(inputs)

  batch_size = inputs.get_shape().as_list()[0]

  sites_sample = tf.random_uniform(shape=inputs.shape, dtype=tf.float32)
  swap_choice = tf.multiply(inputs, sites_sample)

  down_spins = tf.stack(
      [np.arange(0, batch_size), tf.argmin(swap_choice, 1)], 1)
  up_spins = tf.stack(
      [np.arange(0, batch_size), tf.argmax(swap_choice, 1)], 1)

  spin_down_update = tf.scatter_nd(
      down_spins, 2 * np.ones(batch_size, dtype=np.float32), inputs.shape)

  spin_up_update = tf.scatter_nd(
      up_spins, -2 * np.ones(batch_size, dtype=np.float32), inputs.shape)

  updated_config = tf.add_n([inputs, spin_down_update, spin_up_update])
  new_config_psi = wavefunction(updated_config)
  ratios = tf.abs(new_config_psi) / tf.abs(psi)
  metropolis_rnd = tf.sqrt(
      tf.random_uniform(shape=(batch_size,), dtype=tf.float32))

  updates_mask = tf.cast(tf.greater(ratios, metropolis_rnd), tf.float32)

  accepted_spin_down_update = tf.scatter_nd(
      down_spins, 2 * updates_mask * np.ones(batch_size), inputs.shape)
  accepted_spin_up_update = tf.scatter_nd(
      up_spins, -2 * updates_mask * np.ones(batch_size), inputs.shape)

  acceptance_count = tf.reduce_sum(updates_mask)
  mc_step = tf.assign_add(
      inputs, accepted_spin_down_update + accepted_spin_up_update)
  return mc_step, acceptance_count


def get_configs(
    shared_resources: Dict[ResourceName, tf.Tensor],
    batch_size: int,
    n_sites: int,
    include: bool = True,
) -> tf.Tensor:
  """Retrieves or creates a variables to hold a batch of configurations.

  Args:
    shared_resources: System resources shared among different modules.
    batch_size: Number of configurations in the batch.
    n_sites: Number of sites in the system.
    include: Boolean indicating whether to update `shared_resources`.

  Returns:
    Non trainable variable of shape [batch_size, n_sites].

  Raises:
    ValueError: Size of existing variable does not match.
  """
  if ResourceName.CONFIGS in shared_resources:
    configs = shared_resources[ResourceName.CONFIGS]
    if configs.shape.as_list() != [batch_size, n_sites]:
      raise ValueError('Size of existing variable does not match.')
    return configs

  init = utils.random_configurations(n_sites, batch_size)
  configs = tf.get_variable(name='configs', initializer=init, trainable=False)
  if include:
    shared_resources[ResourceName.CONFIGS] = configs
  return configs


def get_monte_carlo_sampling(
    shared_resources: Dict[ResourceName, tf.Tensor],
    inputs: tf.Tensor,
    wavefunction: wavefunctions.Wavefunction,
    include: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Returns operations that update inputs based on MCMC sampling.

  Args:
    shared_resources: System resources shared among different modules.
    inputs: Variable holding system configurations.
    wavefunction: Wavefunction object to use for important sampling.
    include: Boolean indicating whether to update `shared_resources`.

  Returns:
    A tuple of tensors representing operation for updating inputs and evaluating
    acceptance ratio.
  """
  if ResourceName.MONTE_CARLO_SAMPLING in shared_resources:
    return shared_resources[ResourceName.MONTE_CARLO_SAMPLING]
  mc_step, acc_rate = build_monte_carlo_sampling(inputs, wavefunction)
  if include:
    shared_resources[ResourceName.MONTE_CARLO_SAMPLING] = (mc_step, acc_rate)
  return mc_step, acc_rate
