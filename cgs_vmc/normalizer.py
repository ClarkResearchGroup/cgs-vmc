"""Implements wavefunction normalization to match ranges and avoid overflows.

This module will be restructured and potentially moved elsewhere once
wavefunctions are migrated to produce amplitudes in log() + sign() format.
When using logs, overflows and underflows are no longer a problem and hence
the only non-depricated utility will be matching the range for some instances
of supervised training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import wavefunctions


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
      session: Active session in which th wavefunction is normalized.
      value: What is the max value to aim for.
      normalization_iterations: Number of batches to process for normalization.
  """
  psi_value = wavefunction(system_configs)
  set_norm_on_batch = wavefunction.normalize_batch(psi_value, value)
  update_norm_on_batch = wavefunction.update_norm(psi_value, value)

  session.run(set_norm_on_batch)
  for _ in range(normalization_iterations):
    session.run(update_config)
    session.run(update_norm_on_batch)
