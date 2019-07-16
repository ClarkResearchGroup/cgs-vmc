"""Training routines that optimize neural networks and other ansatzs.

This module provides interface and implementations of Wavefunction optimizer.
It separates the process into connection of the necessary modules into the graph
and their execution in a specified order to perform a single epoch of training.

New optimizers should implement the `WavefunctionOptimizer` interface to be
compatible with training pipelines.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from typing import Dict, NamedTuple, Any

import numpy as np
import scipy
import tensorflow as tf

import wavefunctions
import graph_builders
import operators


# TODO(kochkov92) Move evaluation components to summaries and clear TrainOps.
TrainOpsTraditional = NamedTuple(
    'TrainingOpsTraditional', [
        ('accumulate_gradients', tf.Tensor),
        ('apply_gradients', tf.Tensor),
        ('reset_gradients', tf.Tensor),
        ('mc_step', tf.Tensor),
        ('acc_rate', tf.Tensor),
        ('metrics', tf.Tensor),
        ('epoch_increment', tf.Tensor),
        ('a_grad', tf.Tensor),
        ('p_grad', tf.Tensor),
        ('amplitude', tf.Tensor),
        ('phase', tf.Tensor),
        ('energy_gradients', tf.Tensor),
        ('amplitude_grads', tf.Tensor),
        ('w_amplitude_grads', tf.Tensor),
        ('w_phase_grads', tf.Tensor),
        #('apply_gradients2', tf.Tensor),
    ]
)
"""Organizes operations used to execute a training epoch traditional methods."""

OPTIMIZERS = {
    'adam': tf.train.AdamOptimizer,
    'gradient': tf.train.GradientDescentOptimizer,
    'rms_prop': tf.train.RMSPropOptimizer,
    'momentum': tf.train.MomentumOptimizer
}


def create_sgd_optimizer(hparams: tf.contrib.training.HParams):
  """Creates an optimizer as specified in hparams."""
  num_epochs = graph_builders.get_or_create_num_epochs()
  learning_rates = hparams.learning_rates
  learning_rate_stops = hparams.learning_rate_stops
  learning_rate = tf.train.piecewise_constant(
      num_epochs, learning_rate_stops, learning_rates)
  return OPTIMIZERS[hparams.optimizer](learning_rate, beta2=hparams.beta2)


class WavefunctionOptimizer():
  """Parents class for ground state wavefunction optimizers."""

  def build_opt_ops(
      self,
      wavefunction: wavefunctions.Wavefunction,
      hamiltonian: operators.Operator,
      hparams: tf.contrib.training.HParams,
      shared_resources: Dict[graph_builders.ResourceName, tf.Tensor],
  ) -> NamedTuple:
    """Adds wavefunction optimization ops to the graph.

    Args:
      wavefunction: Wavefunction ansatz to optimize.
      hamiltonian: Hamiltonian whose ground state we are solving for.
      hparams: Hyperparameters of the optimization procedure.
      shared_resources: Resources sharable among different modules.

    Returns:
      NamedTuple holding tensors needed to run a training epoch.
    """
    raise NotImplementedError

  def run_optimization_epoch(
      self,
      train_ops: NamedTuple,
      session: tf.Session,
      hparams: tf.contrib.training.HParams,
      epoch_number: int = 0,
  ):
    """Runs training epoch by executing `train_ops` in `session`.

    Args:
      train_ops: Training operations returned by `build_opt_ops` method.
      session: Active session where to run a training epoch.
      hparams: Hyperparameters of the optimization procedure.
      epoch_number: Number of epoch.
    """
    raise NotImplementedError


class EnergyGradientOptimizer(WavefunctionOptimizer):
  """Implements wave-function optimization based on energy gradient.

  This is an improved standard approach to wave-function optimization based
  on reduced variance energy gradients with respect to variational parameters.
  """
  # pylint: disable=too-many-locals
  def build_opt_ops(
      self,
      wavefunction: wavefunctions.Wavefunction,
      hamiltonian: operators.Operator,
      hparams: tf.contrib.training.HParams,
      shared_resources: Dict[graph_builders.ResourceName, tf.Tensor],
  ) -> NamedTuple:
    """Adds wavefunction optimization ops to the graph.

    Args:
      wavefunction: Wavefunction ansatz to optimize.
      hamiltonian: Hamiltonian whose ground state we are solving for.
      hparams: Hyperparameters of the optimization procedure.
      shared_resources: Resources sharable among different modules.

    Returns:
      NamedTuple holding tensors needed to run a training epoch.
    """
    batch_size = hparams.batch_size
    n_sites = hparams.num_sites

    configs = graph_builders.get_configs(shared_resources, batch_size, n_sites)
    mc_step, acc_rate = graph_builders.get_monte_carlo_sampling(
        shared_resources, configs, wavefunction)
    opt_v = wavefunction.get_trainable_variables()

    psi_amplitude, psi_phase = wavefunction(configs)
    local_energy_real, local_energy_img = hamiltonian.local_value(
        wavefunction, configs, psi_amplitude, psi_phase)
    local_energy_real = tf.stop_gradient(local_energy_real)
    local_energy_img = tf.stop_gradient(local_energy_img)

    psi_amplitude_raw_grads = tf.gradients(psi_amplitude, opt_v)
    psi_phase_raw_grads = tf.gradients(psi_phase, opt_v)
    psi_amplitude_grads = [
        tf.convert_to_tensor(grad) for grad in psi_amplitude_raw_grads]
    psi_phase_grads = [
        tf.convert_to_tensor(grad) for grad in psi_phase_raw_grads]
    e_psi_amplitude_raw_grads = tf.gradients(
        psi_amplitude * local_energy_real, opt_v)
    e_psi_amplitude_grads = [
        tf.convert_to_tensor(grad) for grad in e_psi_amplitude_raw_grads]
    #e_psi_amplitude_grads = tf.multiply(psi_amplitude_grads, local_energy_real)
    e_psi_phase_raw_grads = tf.gradients(
        psi_phase * local_energy_img, opt_v)
    e_psi_phase_grads = [
        tf.convert_to_tensor(grad) for grad in e_psi_phase_raw_grads]
    #e_psi_phase_grads = tf.multiply(psi_phase_grads, local_energy_img)

    amplitude_grads = [
        tf.metrics.mean_tensor(grad) for grad in psi_amplitude_grads]
    weighted_amplitude_grads = [
        tf.metrics.mean_tensor(grad) for grad in e_psi_amplitude_grads]
    weighted_phase_grads = [
        tf.metrics.mean_tensor(grad) for grad in e_psi_phase_grads]


    mean_energy, update_energy = tf.metrics.mean(local_energy_real)
    mean_energy_img, update_energy_img = tf.metrics.mean(local_energy_img)
    # QA: why need list(map)
    mean_amplitude_grads, update_amplitude_grads = list(
        map(list, zip(*amplitude_grads)))
    mean_weighted_amplitude_grads, update_weighted_amplitude_grads = list(
        map(list, zip(*weighted_amplitude_grads)))
    mean_weighted_phase_grads, update_weighted_phase_grads = list(
        map(list, zip(*weighted_phase_grads)))

    grad_pairs = zip(
        mean_amplitude_grads, mean_weighted_amplitude_grads,
        mean_weighted_phase_grads
    )

    energy_gradients = [
        weighted_amplitude_grad + weighted_phase_grad - mean_energy * grad
        for grad, weighted_amplitude_grad, weighted_phase_grad in grad_pairs
    ]
    grads_and_vars = list(zip(energy_gradients, opt_v))
    optimizer = create_sgd_optimizer(hparams)
    apply_gradients = optimizer.apply_gradients(grads_and_vars)
    reset_gradients = tf.variables_initializer(tf.local_variables())

    all_updates = (
        [update_energy,] + update_amplitude_grads + update_weighted_amplitude_grads + update_weighted_phase_grads
        )
    accumulate_gradients = tf.group(all_updates)

    num_epochs = graph_builders.get_or_create_num_epochs()
    epoch_increment = tf.assign_add(num_epochs, 1)

    train_ops = TrainOpsTraditional(
        accumulate_gradients=accumulate_gradients,
        apply_gradients=apply_gradients,
        reset_gradients=reset_gradients,
        mc_step=mc_step,
        acc_rate=acc_rate,
        metrics=mean_energy,
        epoch_increment=epoch_increment,
        a_grad=psi_amplitude_raw_grads,
        p_grad=psi_phase_raw_grads,
        amplitude=psi_amplitude,
        phase=psi_phase,
        energy_gradients=energy_gradients,
        amplitude_grads=amplitude_grads,
        w_amplitude_grads=weighted_amplitude_grads,
        w_phase_grads=weighted_phase_grads,
        #apply_gradients2=apply_gradients2,
    )
    return train_ops


  def run_optimization_epoch(
      self,
      train_ops: NamedTuple,
      session: tf.Session,
      hparams: tf.contrib.training.HParams,
      epoch_number: int = 0,
  ) -> np.float32:
    """Runs training epoch by executing `train_ops` in `session`.

    Args:
      train_ops: Training operations returned by `build_opt_ops` method.
      session: Active session where to run a training epoch.
      hparams: Hyperparameters of the optimization procedure.
      epoch_number: Number of epoch.

    Returns:
      Current energy estimate.
      #TODO(kochkov92) Move evaluation to summaries and remove return.
    """
    for _ in range(hparams.num_equilibration_sweeps * hparams.num_sites):
      session.run(train_ops.mc_step)

    session.run(train_ops.reset_gradients)
    for _ in range(hparams.num_batches_per_epoch):
      session.run(train_ops.accumulate_gradients)
      for _ in range(hparams.num_monte_carlo_sweeps * hparams.num_sites):
        session.run(train_ops.mc_step)

    #mc_step=session.run(train_ops.mc_step)
    #print('mc_step', mc_step)
    #acc_g = session.run(train_ops.accumulate_gradients)
    #a_grad = session.run(train_ops.a_grad)
    #p_grad = session.run(train_ops.p_grad)
    #amp = session.run(train_ops.amplitude)
    #phase = session.run(train_ops.phase)
    #amplitude_grads = session.run(train_ops.amplitude_grads)
    #w_amplitude_grads = session.run(train_ops.w_amplitude_grads)
    #w_phase_grads = session.run(train_ops.w_phase_grads)
    #energy_gradients = session.run(train_ops.energy_gradients)
    session.run(train_ops.apply_gradients)
    #session.run(train_ops.apply_gradients2)
    energy = session.run(train_ops.metrics)
    #print('apply', apply_g)
    #print('acc', acc_g)
    #print('a_grad', a_grad)
    #print('p_grad', p_grad)
    #print('amp',amp)
    #print('phase',phase)
    #print('amplitude_grads',amplitude_grads)
    #print('w_amplitude_grads',w_amplitude_grads)
    #print('w_phase_grads',w_phase_grads)
    #print('energy_gradients',energy_gradients)
    print('energy',energy)
    session.run(train_ops.reset_gradients)
    session.run(train_ops.epoch_increment)
    return energy


GROUND_STATE_OPTIMIZERS = {
    'EnergyGradient': EnergyGradientOptimizer,
}
