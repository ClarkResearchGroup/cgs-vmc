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
from typing import Dict, NamedTuple

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
        ('update_wf_norm', tf.Tensor),
    ]
)
"""Organizes operations used to execute a training epoch traditional methods."""


TrainOpsSWO = NamedTuple(
    'TrainingOpsSWO', [
        ('train_step', tf.Tensor),
        ('accumulate_gradients', tf.Tensor),
        ('apply_gradients', tf.Tensor),
        ('reset_gradients', tf.Tensor),
        ('mc_step', tf.Tensor),
        ('acc_rate', tf.Tensor),
        ('metrics', tf.Tensor),
        ('energy', tf.Tensor),
        ('update_supervisor', tf.Tensor),
        ('update_normalization', tf.Tensor),
        ('epoch_increment', tf.Tensor),
        ('update_wf_norm', tf.Tensor),
    ]
)
"""Organizes operations used by Supervised Wavefunction Optimizer SWO."""


TrainOpsSupervised = NamedTuple(
    'TrainingOpsSupervised', [
        ('accumulate_gradients', tf.Tensor),
        ('apply_gradients', tf.Tensor),
        ('reset_gradients', tf.Tensor),
        ('mc_step', tf.Tensor),
        ('acc_rate', tf.Tensor),
        ('metrics', tf.Tensor),
        ('epoch_increment', tf.Tensor),
        ('update_wf_norm', tf.Tensor),
    ]
)
"""Organizes operations used by supervised training optimizer."""


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


class SupervisedWavefunctionOptimizer():
  """Supervised wavefunction optimizer.

  Implements SWO with |psi|^2 sampling and adjusted L2 loss as the
  objective function.
  """
  def build_opt_ops(
      self,
      wavefunction: wavefunctions.Wavefunction,
      target_wavefunction: wavefunctions.Wavefunction,
      hparams: tf.contrib.training.HParams,
      shared_resources: Dict[graph_builders.ResourceName, tf.Tensor],
  ) -> NamedTuple:
    """Adds wavefunction optimization ops to the graph.

    Args:
      wavefunction: Wavefunction model to optimize.
      target_wavefunction: Wavefunction model that serves as a target state.
      shared_resources: Resources sharable among different modules.
      hparams: Hyperparameters of the optimization procedure.

    Returns:
      NamedTuple holding tensors needed to run a training epoch.
    """
    batch_size = hparams.batch_size
    n_sites = hparams.num_sites

    configs = graph_builders.get_configs(shared_resources, batch_size, n_sites)
    mc_step, acc_rate = graph_builders.get_monte_carlo_sampling(
        shared_resources, configs, wavefunction)

    psi = wavefunction(configs)
    psi_target = target_wavefunction(configs)

    loss = tf.reduce_mean(
        tf.squared_difference(psi, psi_target * hparams.scale) /
        (tf.square(tf.stop_gradient(psi)))
    )
    opt_v = wavefunction.get_trainable_variables()
    optimizer = create_sgd_optimizer(hparams)
    train_step = optimizer.minimize(loss, var_list=opt_v)
    num_epochs = graph_builders.get_or_create_num_epochs()
    epoch_increment = tf.assign_add(num_epochs, 1)

    train_ops = TrainOpsSupervised(
        accumulate_gradients=None,
        apply_gradients=train_step,
        reset_gradients=None,
        mc_step=mc_step,
        acc_rate=acc_rate,
        metrics=loss,
        update_wf_norm=None,
        epoch_increment=epoch_increment,
    )
    return train_ops


  def run_optimization_epoch(
      self,
      train_ops: NamedTuple,
      session: tf.Session,
      hparams: tf.contrib.training.HParams,
      epoch_number: int,
  )-> np.float32:
    """Runs training epoch by executing `train_ops` in `session`.

    Args:
      train_ops: Training operations returned by `build_opt_ops` method.
      session: Active session where to run a training epoch.
      hparams: Hyperparameters of the optimization procedure.
      epoch_number: Number of epoch.

    Returns:
      Current wavefunctin fidelity loss estimate.
    """
    del epoch_number  # not used by SupervisedWavefunctionOptimizer.
    for _ in range(hparams.num_batches_per_epoch):
      for _ in range(hparams.num_monte_carlo_sweeps * hparams.num_sites):
        session.run(train_ops.mc_step)
      session.run(train_ops.apply_gradients)
    session.run(train_ops.epoch_increment)
    loss = session.run(train_ops.metrics)
    return loss


class BasisIterationSWO():
  """Supervised wavefunction optimizer.

  Implements SWO based on L2 loss and minibatch sampling from the basis dataset.
  """
  def build_opt_ops(
      self,
      wavefunction: wavefunctions.Wavefunction,
      target_wavefunction: wavefunctions.Wavefunction,
      hparams: tf.contrib.training.HParams,
      shared_resources: Dict[graph_builders.ResourceName, tf.Tensor],
  ) -> NamedTuple:
    """Adds wavefunction optimization ops to the graph.

    Args:
      wavefunction: Wavefunction model to optimize.
      target_wavefunction: Wavefunction model that serves as a target state.
      shared_resources: Resources sharable among different modules.
      hparams: Hyperparameters of the optimization procedure.

    Returns:
      NamedTuple holding tensors needed to run a training epoch.
    """
    del shared_resources  # Not used by BasisIterationSWO.
    batch_size = hparams.batch_size
    n_sites = hparams.num_sites

    basis_dataset = tf.contrib.data.CsvDataset(
        hparams.basis_file_path, [tf.float32 for _ in range(n_sites)],
        header=False, field_delim=' ')
    basis_dataset = basis_dataset.map(lambda *x: tf.convert_to_tensor(x))
    shuffle_batch = scipy.special.binom(n_sites, n_sites / 2)
    basis_dataset = basis_dataset.shuffle(shuffle_batch)
    basis_dataset = basis_dataset.batch(batch_size)
    basis_dataset = basis_dataset.repeat()
    config_iterator = basis_dataset.make_one_shot_iterator()
    configs = config_iterator.get_next() * 2. - 1.

    psi = wavefunction(configs)
    psi_target = target_wavefunction(configs)

    loss = tf.reduce_mean(
        tf.squared_difference(psi, psi_target * hparams.scale))
    opt_v = wavefunction.get_trainable_variables()
    optimizer = create_sgd_optimizer(hparams)
    train_step = optimizer.minimize(loss, var_list=opt_v)
    num_epochs = graph_builders.get_or_create_num_epochs()
    epoch_increment = tf.assign_add(num_epochs, 1)

    train_ops = TrainOpsSupervised(
        accumulate_gradients=None,
        apply_gradients=train_step,
        reset_gradients=None,
        mc_step=None,
        acc_rate=None,
        metrics=loss,
        update_wf_norm=None,
        epoch_increment=epoch_increment,
    )
    return train_ops


  def run_optimization_epoch(
      self,
      train_ops: NamedTuple,
      session: tf.Session,
      hparams: tf.contrib.training.HParams,
      epoch_number: int,
  )-> np.float32:
    """Runs training epoch by executing `train_ops` in `session`.

    Args:
      train_ops: Training operations returned by `build_opt_ops` method.
      session: Active session where to run a training epoch.
      hparams: Hyperparameters of the optimization procedure.
      epoch_number: Number of epoch.

    Returns:
      Current wavefunctin fidelity loss estimate.
    """
    del epoch_number  # not used by SupervisedWavefunctionOptimizer.
    for _ in range(hparams.num_batches_per_epoch):
      session.run(train_ops.apply_gradients)
    session.run(train_ops.epoch_increment)
    loss = session.run(train_ops.metrics)
    return loss


class LogOverlapSWO():
  """Supervised wavefunction optimizer based on log(overlap) optimization.

  Implements SWO with |psi|^2 sampling and Log(|<psi|phi>|^{2}) as the
  objective function.
  """
  # pylint: disable=too-many-locals
  def build_opt_ops(
      self,
      wavefunction: wavefunctions.Wavefunction,
      target_wavefunction: wavefunctions.Wavefunction,
      hparams: tf.contrib.training.HParams,
      shared_resources: Dict[graph_builders.ResourceName, tf.Tensor],
  ) -> NamedTuple:
    """Adds wavefunction optimization ops to the graph.

    Args:
      wavefunction: Wavefunction model to optimize.
      target_wavefunction: Wavefunction model that serves as a target state.
      shared_resources: Resources sharable among different modules.
      hparams: Hyperparameters of the optimization procedure.

    Returns:
      NamedTuple holding tensors needed to run a training epoch.
    """
    batch_size = hparams.batch_size
    n_sites = hparams.num_sites
    configs = graph_builders.get_configs(shared_resources, batch_size, n_sites)
    mc_step, acc_rate = graph_builders.get_monte_carlo_sampling(
        shared_resources, configs, wavefunction)

    psi = wavefunction(configs)
    psi_target = target_wavefunction(configs) * hparams.scale
    psi_no_grad = tf.stop_gradient(psi)
    opt_v = wavefunction.get_trainable_variables()
    # Computation of L=log(overlap) gradient with respect to opt_v using
    # "grad(L) = <grad(log(psi))> - <grad(log(psi))*ratio> / <ratio>"
    # where ratio is the ratio of target and curren wave-function.
    ratio = tf.stop_gradient(psi_target / psi)
    log_psi_raw_grads = tf.gradients(psi / psi_no_grad, opt_v)
    log_psi_grads = [tf.convert_to_tensor(grad) for grad in log_psi_raw_grads]
    ratio_log_psi_raw_grads = tf.gradients(ratio * psi / psi_no_grad, opt_v)
    ratio_log_psi_grads = [
        tf.convert_to_tensor(grad) for grad in ratio_log_psi_raw_grads
    ]
    log_grads = [
        tf.metrics.mean_tensor(log_psi_grad) for log_psi_grad in log_psi_grads
    ]
    ratio_grads = [
        tf.metrics.mean_tensor(grad) for grad in ratio_log_psi_grads
    ]
    mean_ratio, accumulate_ratio = tf.metrics.mean(ratio)
    mean_log_grads, accumulate_log_grads = list(map(list, zip(*log_grads)))
    mean_ratio_grads, accumulate_ratio_grads = list(
        map(list, zip(*ratio_grads)))
    all_updates = accumulate_log_grads + accumulate_ratio_grads
    all_updates.append(accumulate_ratio)
    accumulate_gradients = tf.group(all_updates)
    optimizer = create_sgd_optimizer(hparams)
    grad_pairs = zip(mean_log_grads, mean_ratio_grads)
    overlap_gradients = [
        grad - scaled_grad / mean_ratio for grad, scaled_grad in grad_pairs
    ]
    grads_and_vars = list(zip(overlap_gradients, opt_v))
    apply_gradients = optimizer.apply_gradients(grads_and_vars)

    reset_gradients = tf.variables_initializer(tf.local_variables())
    update_wf_norm = wavefunction.update_norm(psi)
    num_epochs = graph_builders.get_or_create_num_epochs()
    epoch_increment = tf.assign_add(num_epochs, 1)

    train_ops = TrainOpsSupervised(
        accumulate_gradients=accumulate_gradients,
        apply_gradients=apply_gradients,
        reset_gradients=reset_gradients,
        mc_step=mc_step,
        acc_rate=acc_rate,
        metrics=None,
        update_wf_norm=update_wf_norm,
        epoch_increment=epoch_increment,
    )
    return train_ops


  def run_optimization_epoch(
      self,
      train_ops: NamedTuple,
      session: tf.Session,
      hparams: tf.contrib.training.HParams,
      epoch_number: int,
  ):
    """Runs training epoch by executing `train_ops` in `session`.

    Args:
      train_ops: Training operations returned by `build_opt_ops` method.
      session: Active session where to run a training epoch.
      hparams: Hyperparameters of the optimization procedure.
      epoch_number: Number of epoch.
    """
    del epoch_number  # not used by SupervisedWavefunctionOptimizer.
    for _ in range(hparams.num_batches_per_epoch):
      for _ in range(hparams.num_monte_carlo_sweeps * hparams.num_sites):
        session.run(train_ops.mc_step)
      session.run(train_ops.reset_gradients)
      session.run(train_ops.accumulate_gradients)
      session.run(train_ops.apply_gradients)
    session.run(train_ops.epoch_increment)


class DualSamplingSWO():
  """Supervised wavefunction optimizer based on SWO optimization.

  Implements SWO with |psi|^2 sampling from both target and current states.
  It uses L2(psi-phi) as the objective function. Currently the sampling bias
  is not accounted for in the gradients. We see more stable performance using
  such parameters for small system sizes.
  """
  # pylint: disable=too-many-locals
  def build_opt_ops(
      self,
      wavefunction: wavefunctions.Wavefunction,
      target_wavefunction: wavefunctions.Wavefunction,
      hparams: tf.contrib.training.HParams,
      shared_resources: Dict[graph_builders.ResourceName, tf.Tensor],
  ) -> NamedTuple:
    """Adds wavefunction optimization ops to the graph.

    Args:
      wavefunction: Wavefunction model to optimize.
      target_wavefunction: Wavefunction model that serves as a target state.
      shared_resources: Resources sharable among different modules.
      hparams: Hyperparameters of the optimization procedure.

    Returns:
      NamedTuple holding tensors needed to run a training epoch.
    """
    batch_size = hparams.batch_size
    n_sites = hparams.num_sites
    psi_configs = graph_builders.get_configs(
        shared_resources, batch_size // 2, n_sites)
    target_configs = graph_builders.get_configs(
        shared_resources, batch_size // 2, n_sites,
        configs_id=graph_builders.ResourceName.TARGET_CONFIGS)

    # Here we use explicit call to build separate MCMC sampling for two wf.
    target_mc_step, target_acc_rate = graph_builders.build_monte_carlo_sampling(
        target_configs, target_wavefunction)
    psi_mc_step, psi_acc_rate = graph_builders.build_monte_carlo_sampling(
        psi_configs, wavefunction)
    mc_step = tf.group([psi_mc_step, target_mc_step])
    acc_rate = tf.group([psi_acc_rate, target_acc_rate])

    configs = tf.concat([psi_configs, target_configs], axis=0)
    psi = wavefunction(configs)
    psi_target = target_wavefunction(configs)

    # # A version of accounting for sampling bias.
    # psi_no_grad = tf.stop_gradient(psi)
    # loss = tf.reduce_mean(
    #     tf.squared_difference(psi, psi_target) /
    #     (tf.square(psi_no_grad) + tf.square(psi_target))
    # )

    loss = tf.reduce_mean(
        tf.squared_difference(psi, psi_target * hparams.scale)
    )
    opt_v = wavefunction.get_trainable_variables()
    optimizer = create_sgd_optimizer(hparams)
    train_step = optimizer.minimize(loss, var_list=opt_v)
    num_epochs = graph_builders.get_or_create_num_epochs()
    epoch_increment = tf.assign_add(num_epochs, 1)

    train_ops = TrainOpsSupervised(
        accumulate_gradients=None,
        apply_gradients=train_step,
        reset_gradients=None,
        mc_step=mc_step,
        acc_rate=acc_rate,
        metrics=loss,
        update_wf_norm=None,
        epoch_increment=epoch_increment,
    )
    return train_ops


  def run_optimization_epoch(
      self,
      train_ops: NamedTuple,
      session: tf.Session,
      hparams: tf.contrib.training.HParams,
      epoch_number: int,
  )-> np.float32:
    """Runs training epoch by executing `train_ops` in `session`.

    Args:
      train_ops: Training operations returned by `build_opt_ops` method.
      session: Active session where to run a training epoch.
      hparams: Hyperparameters of the optimization procedure.
      epoch_number: Number of epoch.

    Returns:
      Current wavefunctin fidelity loss estimate.
    """
    del epoch_number  # not used by SupervisedWavefunctionOptimizer.
    for _ in range(hparams.num_batches_per_epoch):
      for _ in range(hparams.num_monte_carlo_sweeps * hparams.num_sites):
        session.run(train_ops.mc_step)
      session.run(train_ops.apply_gradients)
    session.run(train_ops.epoch_increment)
    loss = session.run(train_ops.metrics)
    return loss


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

    psi = wavefunction(configs)
    psi_no_grad = tf.stop_gradient(psi)
    update_wf_norm = wavefunction.update_norm(psi)
    local_energy = hamiltonian.local_value(wavefunction, configs, psi)
    local_energy = tf.stop_gradient(local_energy)

    log_psi_raw_grads = tf.gradients(psi / psi_no_grad, opt_v)
    log_psi_grads = [tf.convert_to_tensor(grad) for grad in log_psi_raw_grads]
    e_psi_raw_grads = tf.gradients(psi / psi_no_grad * local_energy, opt_v)
    e_psi_grads = [tf.convert_to_tensor(grad) for grad in e_psi_raw_grads]

    grads = [
        tf.metrics.mean_tensor(log_psi_grad) for log_psi_grad in log_psi_grads
    ]
    weighted_grads = [tf.metrics.mean_tensor(grad) for grad in e_psi_grads]

    mean_energy, update_energy = tf.metrics.mean(local_energy)
    mean_pure_grads, update_pure_grads = list(map(list, zip(*grads)))
    mean_scaled_grads, update_scaled_grads = list(
        map(list, zip(*weighted_grads)))

    grad_pairs = zip(mean_pure_grads, mean_scaled_grads)

    energy_gradients = [
        scaled_grad - mean_energy * grad for grad, scaled_grad in grad_pairs
    ]
    grads_and_vars = list(zip(energy_gradients, opt_v))
    optimizer = create_sgd_optimizer(hparams)
    apply_gradients = optimizer.apply_gradients(grads_and_vars)
    reset_gradients = tf.variables_initializer(tf.local_variables())

    all_updates = [update_energy,] + update_pure_grads + update_scaled_grads
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
        update_wf_norm=update_wf_norm,
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

    if train_ops.update_wf_norm is not None:
      session.run(train_ops.update_wf_norm)
    session.run(train_ops.reset_gradients)
    for _ in range(hparams.num_batches_per_epoch):
      session.run(train_ops.accumulate_gradients)
      for _ in range(hparams.num_monte_carlo_sweeps * hparams.num_sites):
        session.run(train_ops.mc_step)

    session.run(train_ops.apply_gradients)
    energy = session.run(train_ops.metrics)
    session.run(train_ops.reset_gradients)
    session.run(train_ops.epoch_increment)
    return energy


class LogOverlapImaginaryTimeSWO(WavefunctionOptimizer):
  """Implements imaginary time SWO based on log overlap gradient formula.

  IT-SWO scheme implemented with gradients evaluated without a loss function.
  In this approach it is not necessary to maintain a normalization constant.
  See arxiv.1808.05232.pdf appendix.
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

    wf_omega = copy.deepcopy(wavefunction)  # building supervisor wavefunction.
    psi = wavefunction(configs)
    psi_omega = wf_omega(configs)
    beta = tf.constant(hparams.time_evolution_beta, dtype=tf.float32)
    h_psi_omega = hamiltonian.apply_in_place(wf_omega, configs, psi_omega)
    h_psi_omega_beta = beta * h_psi_omega
    ite_psi_omega = psi_omega - h_psi_omega_beta
    local_energy = h_psi_omega / psi_omega

    update_wf_norm = wavefunction.update_norm(psi)
    psi_no_grad = tf.stop_gradient(psi)

    ratio = tf.stop_gradient(ite_psi_omega / psi)

    log_psi_raw_grads = tf.gradients(psi / psi_no_grad, opt_v)
    log_psi_grads = [tf.convert_to_tensor(grad) for grad in log_psi_raw_grads]
    ratio_log_psi_raw_grads = tf.gradients(ratio * psi / psi_no_grad, opt_v)
    ratio_log_psi_grads = [
        tf.convert_to_tensor(grad) for grad in ratio_log_psi_raw_grads
    ]

    log_grads = [
        tf.metrics.mean_tensor(log_psi_grad) for log_psi_grad in log_psi_grads
    ]

    ratio_grads = [
        tf.metrics.mean_tensor(grad) for grad in ratio_log_psi_grads
    ]

    mean_energy, accumulate_energy = tf.metrics.mean(local_energy)
    mean_ratio, accumulate_ratio = tf.metrics.mean(ratio)
    mean_log_grads, accumulate_log_grads = list(map(list, zip(*log_grads)))
    mean_ratio_grads, accumulate_ratio_grads = list(
        map(list, zip(*ratio_grads)))

    grad_pairs = zip(mean_log_grads, mean_ratio_grads)

    overlap_gradients = [
        grad - scaled_grad / mean_ratio for grad, scaled_grad in grad_pairs
    ]
    grads_and_vars = list(zip(overlap_gradients, opt_v))

    optimizer = create_sgd_optimizer(hparams)

    apply_gradients = optimizer.apply_gradients(grads_and_vars)
    all_updates = [accumulate_ratio, accumulate_energy]
    all_updates += accumulate_log_grads + accumulate_ratio_grads
    accumulate_gradients = tf.group(all_updates)

    update_network = wavefunctions.module_transfer_ops(wavefunction, wf_omega)
    clear_gradients = tf.variables_initializer(tf.local_variables())

    num_epochs = graph_builders.get_or_create_num_epochs()
    epoch_increment = tf.assign_add(num_epochs, 1)

    train_ops = TrainOpsSWO(
        train_step=None,
        accumulate_gradients=accumulate_gradients,
        apply_gradients=apply_gradients,
        reset_gradients=clear_gradients,
        mc_step=mc_step,
        acc_rate=acc_rate,
        metrics=None,
        energy=mean_energy,
        update_supervisor=update_network,
        update_normalization=None,
        epoch_increment=epoch_increment,
        update_wf_norm=update_wf_norm,
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

    if train_ops.update_wf_norm is not None:
      session.run(train_ops.update_wf_norm)
    session.run(train_ops.update_supervisor)
    for _ in range(hparams.num_batches_per_epoch):
      for _ in range(hparams.num_monte_carlo_sweeps * hparams.num_sites):
        session.run(train_ops.mc_step)
      session.run(train_ops.reset_gradients)
      session.run(train_ops.accumulate_gradients)
      session.run(train_ops.apply_gradients)
    session.run(train_ops.epoch_increment)
    energy = session.run(train_ops.energy)

    # TODO(kochkov92) automate the below behavior to the pipeline
    # for batch in range(hparams.num_batches_per_epoch + 1):
    #   session.run(train_ops.accumulate_gradients)
    #   if epoch_number < 400:
    #     session.run(train_ops.apply_gradients)
    #     session.run(train_ops.reset_gradients)
    #   else:
    #     if batch > 0 and batch % 10 == 0:
    #       session.run(train_ops.apply_gradients)
    #       session.run(train_ops.reset_gradients)
    #   for _ in range(hparams.num_monte_carlo_sweeps * hparams.num_sites):
    #     session.run(train_ops.mc_step)

    return energy


class ImaginaryTimeSWO(WavefunctionOptimizer):
  """Implements imaginary time SWO based L2 loss with normalization."""
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

    wf_omega = copy.deepcopy(wavefunction)  # building supervisor wavefunction.
    beta = tf.constant(hparams.time_evolution_beta, dtype=tf.float32)
    beta2 = tf.constant(hparams.time_evolution_beta ** 2, dtype=tf.float32)
    psi_omega = wf_omega(configs)
    h_psi_omega = hamiltonian.apply_in_place(wf_omega, configs, psi_omega)
    h_psi_omega_beta = h_psi_omega * beta
    ite_psi_omega = psi_omega - h_psi_omega_beta

    local_energy = h_psi_omega / psi_omega
    energy_expectation = tf.reduce_mean(local_energy)
    squared_energy_expectation = tf.reduce_mean(tf.square(local_energy))

    ite_normalization = tf.sqrt(
        1. - 2 * beta * energy_expectation + squared_energy_expectation * beta2
    )

    ite_normalization_var = tf.get_variable(
        name='ite_normalization',
        initializer=tf.ones(shape=[]),
        dtype=tf.float32,
        trainable=False
    )

    num_epochs = graph_builders.get_or_create_num_epochs()
    exp_moving_average = tf.train.ExponentialMovingAverage(0.999, num_epochs)

    accumulate_norm = exp_moving_average.apply([ite_normalization])
    normalization_value = exp_moving_average.average(ite_normalization)
    accumulate_energy = exp_moving_average.apply([energy_expectation])
    energy_value = exp_moving_average.average(energy_expectation)

    update_normalization = tf.assign(ite_normalization_var, normalization_value)

    psi = wavefunction(configs)
    loss = tf.reduce_mean(
        tf.squared_difference(psi, ite_psi_omega / ite_normalization_var)
        / tf.square(tf.stop_gradient(psi))
    )
    optimizer = create_sgd_optimizer(hparams)
    train_step = optimizer.minimize(loss, var_list=opt_v)
    train_step = tf.group([train_step, accumulate_norm, accumulate_energy])

    update_supervisor = wavefunctions.module_transfer_ops(
        wavefunction, wf_omega)
    epoch_increment = tf.assign_add(num_epochs, 1)
    update_wf_norm = wavefunction.update_norm(psi)

    train_ops = TrainOpsSWO(
        train_step=train_step,
        accumulate_gradients=None,
        apply_gradients=None,
        reset_gradients=None,
        mc_step=mc_step,
        acc_rate=acc_rate,
        metrics=loss,
        energy=energy_value,
        update_supervisor=update_supervisor,
        update_normalization=update_normalization,
        epoch_increment=epoch_increment,
        update_wf_norm=update_wf_norm,
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
      hparams: Optimization hyperparameters.
      epoch_number: Number of epoch.

    Returns:
      Current energy estimate.
      #TODO(kochkov92) Move evaluation to summaries and remove return.
    """
    # session.run(train_ops.update_supervisor)
    for _ in range(hparams.num_equilibration_sweeps * hparams.num_sites):
      session.run(train_ops.mc_step)

    if train_ops.update_wf_norm is not None:
      session.run(train_ops.update_wf_norm)
    session.run(train_ops.update_supervisor)

    loss_values = []
    for _ in range(hparams.num_batches_per_epoch):
      for _ in range(hparams.num_monte_carlo_sweeps * hparams.num_sites):
        session.run(train_ops.mc_step)
      _, loss = session.run([train_ops.train_step, train_ops.metrics])
      loss_values.append(loss)
    energy_value = session.run(train_ops.energy)
    session.run(train_ops.update_normalization)
    session.run(train_ops.epoch_increment)
    return energy_value


GROUND_STATE_OPTIMIZERS = {
    'EnergyGradient': EnergyGradientOptimizer,
    'LogOverlapITSWO': LogOverlapImaginaryTimeSWO,
    'ITSWO': ImaginaryTimeSWO,
}


SUPERVISED_OPTIMIZERS = {
    'SWO': SupervisedWavefunctionOptimizer,
    'LogOverlapSWO': LogOverlapSWO,
    'DualSamplingSWO': DualSamplingSWO,
    'BasisIterSWO': BasisIterationSWO,
}
