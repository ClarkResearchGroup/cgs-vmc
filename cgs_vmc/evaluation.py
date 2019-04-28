"""Defines utilities for evaluation of the resulting models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from typing import Dict, List, NamedTuple, Any

import numpy as np
import tensorflow as tf

import graph_builders
import operators
import wavefunctions


EvalOps = NamedTuple(
    'EvaluationOps', [
        ('value', tf.Tensor),
        ('mc_step', tf.Tensor),
        ('acceptance_rate', tf.Tensor),
        ('placeholder_input', tf.Tensor),
        ('wavefunction_value', tf.Tensor),
    ]
)
"""Named tuple of tensors representing evaluation components."""
# TODO(kochkov92) Make separate EvalOps for different evaluators.


class WavefunctionEvaluator():
  """Parents class for wavefunction evaluators."""

  def build_eval_ops(
      self,
      wavefunction: wavefunctions.Wavefunction,
      operator: operators.Operator,
      hparams: tf.contrib.training.HParams,
      shared_resources: Dict[graph_builders.ResourceName, Any],
  ):
    """Adds wavefunction evaluation ops to the graph.

    Args:
      wavefunction: Wavefunction ansatz to evalutate.
      operator: To be replaced with Operator class encoding the observable.
      shared_resources: Resources sharable among different modules.
      hparams: Hyperparameters of the evaluation procedure.

    Returns:
      NamedTuple holding tensors needed to run evaluation.
    """
    raise NotImplementedError

  def run_evaluation(
      self,
      eval_ops: EvalOps,
      session: tf.Session,
      hparams: tf.contrib.training.HParams,
      epoch_num: int,
  ) -> Any:
    """Runs evaluation operations `eval_ops` in `session`.

    Args:
      eval_ops: Tensors in the active graph to be used for evaluation.
      session: Active session.
      hparams: Hyperparameters of the evaluation procedure.
      epoch_num: Epoch number.

    Returns:
      The result of evaluation.
    """
    raise NotImplementedError


class MonteCarloOperatorEvaluator(WavefunctionEvaluator):
  """Implements operator evaluation by running MCMC."""

  def build_eval_ops(
      self,
      wavefunction: wavefunctions.Wavefunction,
      operator: operators.Operator,
      hparams: tf.contrib.training.HParams,
      shared_resources: Dict[graph_builders.ResourceName, Any],
  ) -> EvalOps:
    """Adds wavefunction evaluation ops to the graph.

    Args:
      wavefunction: Wavefunction ansatz to evalutate.
      operator: Operator corresponding to the value we want to evaluate.
      hparams: Hyperparameters of the evaluation procedure.
      shared_resources: System resources shared among different modules.

    Returns:
      NamedTuple holding tensors needed to run evaluation.
    """
    batch_size = hparams.batch_size
    n_sites = hparams.num_sites

    configs = graph_builders.get_configs(shared_resources, batch_size, n_sites)
    mc_step, acc_rate = graph_builders.get_monte_carlo_sampling(
        shared_resources, configs, wavefunction)

    value = tf.reduce_mean(operator.local_value(wavefunction, configs))
    eval_ops = EvalOps(
        value=value,
        mc_step=mc_step,
        acceptance_rate=acc_rate,
        placeholder_input=None,
        wavefunction_value=None,
    )
    return eval_ops


  def run_evaluation(
      self,
      eval_ops: EvalOps,
      session: tf.Session,
      hparams: tf.contrib.training.HParams,
      epoch_num: int,
  ) -> List[float]:
    """Runs evaluation operations `eval_ops` in `session`.

    Args:
      eval_ops: Tensors in the active graph to be used for evaluation.
      session: Active session.
      hparams: Hyperparameters of the evaluation procedure.
      epoch_num: Epoch number.

    Returns:
      List of local values of corresponding operator, has size
      `hparams.num_evaluation_samples`.
    """
    value = eval_ops.value
    mc_step = eval_ops.mc_step
    acceptance_ratio = eval_ops.acceptance_rate
    num_equilibration_sweeps = hparams.num_equilibration_sweeps
    num_evaluation_samples = hparams.num_evaluation_samples
    num_mc_steps = hparams.num_monte_carlo_sweeps * hparams.num_sites
    for _ in range(0, num_equilibration_sweeps * hparams.num_sites):
      session.run(mc_step)
    values = []
    # acceptance_count = 0
    for _ in range(0, num_evaluation_samples):
      values.append(session.run(value))
      for _ in range(0, num_mc_steps):
        session.run([mc_step, acceptance_ratio])
        # TODO(kochkov92) report acceptance ratio somewhere.
        # _, accept = session.run([mc_step, acceptance_ratio])
        # acceptance_count += accept
    # batch_size = hparams.batch_size
    # total_num_moves = num_mc_step * num_evaluation_samples * batch_size
    # A = acceptance_count / total_num_moves
    return values


class VectorWavefunctionEvaluator(WavefunctionEvaluator):
  """Implements the process of evaluation of wavefunction on full basis."""

  def build_eval_ops(
      self,
      wavefunction: wavefunctions.Wavefunction,
      operator: operators.Operator,
      hparams: tf.contrib.training.HParams,
      shared_resources: Dict[graph_builders.ResourceName, Any],
  ) -> EvalOps:
    """Adds wavefunction evaluation ops to the graph.

    Args:
      wavefunction: Wavefunction that will be evaluated.
      operator: Not used by SaveullWavefunction
      shared_resources: Resources sharable among different modules.
      hparams: Hyperparameters of the evaluation procedure.

    Returns:
      NamedTuple holding tensors needed to run evaluation.
    """
    del operator  # not used by VectorWavefunctionEvaluator.
    batch_size = hparams.batch_size
    n_sites = hparams.num_sites

    placeholder_input = tf.placeholder(tf.float32, shape=[batch_size, n_sites])
    wavefunction_value = wavefunction(placeholder_input)

    eval_ops = EvalOps(
        value=None,
        mc_step=None,
        acceptance_rate=None,
        placeholder_input=placeholder_input,
        wavefunction_value=wavefunction_value,
    )
    return eval_ops


  def run_evaluation(
      self,
      eval_ops: EvalOps,
      session: tf.Session,
      hparams: tf.contrib.training.HParams,
      epoch_num: int,
  ):
    """Runs evaluation operations `eval_ops` in `session`.

    Saves a full wave-function under name wavefunction_epoch_{}.txt to
    checkpoint directory based on basis configurations provided in hparams.

    Args:
      eval_ops: Tensors in the active graph to be used for evaluation.
      session: Active session.
      hparams: Hyperparameters of the evaluation procedure.
      epoch_num: Epoch number.

    Raises:
      ValueError: If basis file path is not set.
    """
    batch_size = hparams.batch_size
    n_sites = hparams.num_sites
    basis_file_path = hparams.basis_file_path
    placeholder_input = eval_ops.placeholder_input
    wavefunction_value = eval_ops.wavefunction_value

    if basis_file_path == '':
      raise ValueError('Basis file path is not set.')

    output_file_name = 'wavefunction_epoch_{}.txt'.format(epoch_num)
    output_file_path = os.path.join(hparams.checkpoint_dir, output_file_name)
    out_file = open(output_file_path, 'w')

    config_ind = 0
    config_batch = np.zeros((batch_size, n_sites))
    # TODO(kochkov92) Move to iterating using generator.
    with open(basis_file_path) as basis_file:
      for line in basis_file:
        config_batch[config_ind, :] = np.fromstring(line, dtype=float, sep=' ')
        config_ind += 1
        if config_ind == batch_size:
          config_batch = config_batch * 2. - 1.
          feed_dict = {placeholder_input: config_batch}
          psi = session.run(wavefunction_value, feed_dict=feed_dict)
          config_ind = 0
          for i in range(0, batch_size):
            out_file.write('{} {}\n'.format(psi[i].real, psi[i].imag))
    config_batch = config_batch * 2. - 1.
    feed_dict = {placeholder_input: config_batch}
    psi = session.run(wavefunction_value, feed_dict=feed_dict)
    for i in range(0, config_ind):
      out_file.write('{} {}\n'.format(psi[i].real, psi[i].imag))
    out_file.close()
