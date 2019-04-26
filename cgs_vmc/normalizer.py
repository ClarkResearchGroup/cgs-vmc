"""Implements wavefunction normalization to match ranges and avoid overflows.

This module will be restructured and potentially moved elsewhere once
wavefunctions are migrated to produce amplitudes in log() + sign() format.
When using logs, overflows and underflows are no longer a problem and hence
the only non-depricated utility will be matching the range for some instances
of supervised training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Dict, Any

import tensorflow as tf
import numpy as np

import wavefunctions
import graph_builders


def normalize_wavefunction(
    wavefunction: wavefunctions.Wavefunction,
    system_configs: tf.Tensor,
    update_config: tf.Tensor,
    session: tf.Session,
    value: np.float32 = 1e10,
    normalization_iterations: int = 200,
):
  """Generates and executes operations that the fix range of wf amplitudes.

  Generates set_norm and update_norm operations that are then executed to set
  the range of wavefunction amplitudes to be (0, 10**20).

  Args:
      wavefunction: Wavefunction model to normalize.
      system_configs: Configurations on which system can be evaluated.
      update_config: Operation that samples new `system_configs`.
      session: Active session in which the wavefunction is normalized.
      value: What is the max value to aim for.
      normalization_iterations: Number of batches to process for normalization.
  """
  psi_value = wavefunction(system_configs)
  set_norm_on_batch = wavefunction.normalize_batch(psi_value, value)
  update_norm_on_batch = wavefunction.update_norm(psi_value, value)

  if not set_norm_on_batch:
    return
  session.run(set_norm_on_batch)
  for _ in range(normalization_iterations):
    session.run(update_config)
    session.run(update_norm_on_batch)


def build_normalization_ops(
    wavefunction: wavefunctions.Wavefunction,
    hparams: tf.contrib.training.HParams,
    shared_resources: Dict[graph_builders.ResourceName, tf.Tensor],
) -> Any:
  """Generates operations that the fix range of wf amplitudes.

  Construct computational graph to normalize the wave function magnitude by
  dividing the largest magnitude from monte_carlo sampling.

  Args:
      wavefunction: Wavefunction model to normalize.
      hparams: Class holding hyperparameters of the wavefunction ansatzs.
      shared_resources: Resources sharable among different modules.

  Returns:
      Max magnitude of wave function from monte carlo sampling and monte carlo
      step mc_step.
  """

  batch_size = hparams.batch_size
  n_sites = hparams.num_sites
  configs = graph_builders.get_configs(shared_resources, batch_size, n_sites)
  mc_step = graph_builders.get_monte_carlo_sampling(
      shared_resources, configs, wavefunction)[0]
  psi_value = wavefunction(configs)
  max_value = tf.reduce_max(tf.square(psi_value))

  return max_value, mc_step


def run_normalization_ops(
    max_value: Any,
    mc_step: tf.Tensor,
    wavefunction: wavefunctions.Wavefunction,
    session: tf.Session,
    hparams: tf.contrib.training.HParams,
) -> wavefunctions.Wavefunction:
  """Executes operations that the fix range of wf amplitudes.

  Run computational graph to normalize the wave function magnitude by
  dividing the largest magnitude from monte_carlo sampling.

  Args:
      max_value: Max value of wavefunction magnitude square estimated,
      mc_step: Monte Carlo step,
      wavefunction: Wavefunction model to normalize.
      session: Active session in which the wavefunction is normalized.
      hparams: Class holding hyperparameters of the wavefunction ansatzs.

  Returns:
      Wavefunction normalized by the max magnitude from monte carlo sampling
      and monte carlo.
  """

  n_sites = hparams.num_sites
  normalization = hparams.normalization

  if normalization == 'monte_carlo':
    current_max = 0.
    for _ in range(hparams.num_equilibration_sweeps * n_sites):
      session.run(mc_step)
      current_max = max(current_max, session.run(max_value))
    scale = 1.0 / np.sqrt(current_max)
  elif normalization == 'space_scale':
    scale = np.sqrt(2 ** n_sites)

  return wavefunction * scale
