"""Defines utilities and hyperparameters for training and evaluation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any

import numpy as np
import tensorflow as tf

from tensorflow.contrib.training.python.training import hparam_pb2
from google.protobuf import text_format


def create_hparams(**kwargs: Any) -> tf.contrib.training.HParams:
  """Creaates Hparams with system and network parameters.

  Directory parameters:
    checkpoint_dir: Directory where the model and metrics is saved.
    supervisor_dir: Directory with supervisor files (in case of direct SWO).
    basis_file_path: Path to the basis file to generate vectors (0 1 format).

  System parameters:
    wavefunction_type: Wavefunction ansatz.
    wavefunction_optimizer_type: Wavefunction optimizer to use.
    num_sites: Number of sites in the system.
    size_x: Dimension along x direction (used for reshaping for conv2d).
    size_y: Dimension along y direction (used for reshaping for conv2d).
    size_z: Dimension along z direction (currently not used).

  Layer parameters (only used if present in the model):
    num_fc_layers: Number of fully connected layers.
    fc_layer_size: Number of neurons per layer.

    num_conv_layers: Number of convolutional layers.
    conv_strides: Convolutional strides.
    kernel_size: Size of convolutional kernels.
    num_conv_filters: Number of convolutional filters.

    num_resnet_blocks: Number of Residual blocks (Deep residual networks paper).

    nonlinearity: Nonlinearity to use for hidden activations.
    output_activation: Nonlinearity to use for the output (if included).

  Monte carlo paramteres:
    num_equilibration_sweeps: Number of markov chain sweeps for equilibration.
    num_monte_carlo_sweeps: Number of markov chain sweeps between samples.

  Training parameters:
    num_epochs: Total number of epochs to run.
    batch_size: Number of configurations in the batch (num of markov chains).
    num_batches_per_epoch: Number of batches sampled per epoch.
    time_evolution_beta: Imaginary time evolution step for ITSWO.
    learning_rates: Learning rates used for SGD
    learning_rate_stops: Epoch stops when learning rate is adjusted.
    optimizer: SGD optimizer used to training.
    beta2: Second momenta for optimizer

  Evaluation parameters:
    num_evaluation_samples: Number of batches to use for evaluation.

  Args:
    **kwargs: Default hyper-parameter values to override.

  Returns:
     HParams object with all hyperparameter values.
  """
  hparams = tf.contrib.training.HParams(
      # Simulation parameters
      checkpoint_dir='',
      supervisor_dir='',
      basis_file_path='',

      wavefunction_type='',
      wavefunction_optimizer_type='',

      # System parameters
      num_sites=40,
      size_x=1,
      size_y=1,
      size_z=1,

      # Fully connected parameters
      num_fc_layers=3,
      fc_layer_size=80,

      # Convolutional parameters
      num_conv_layers=5,
      conv_strides=1,
      kernel_size=5,
      num_conv_filters=16,

      # ResNet parameters
      num_resnet_blocks=2,

      # MPS parameters
      bond_dimension=4,

      nonlinearity='relu',
      output_activation='exp',

      # Monte Carlo parameters
      num_equilibration_sweeps=100,
      num_monte_carlo_sweeps=1,

      # Training parameters
      num_epochs=500,
      batch_size=200,
      num_batches_per_epoch=50,
      time_evolution_beta=0.12,

      learning_rates=[1e-3, 1e-4, 2e-5, 1e-5],
      learning_rate_stops=[300, 600, 1000],
      optimizer='adam',
      beta2=0.99,

      # Evaluation_parameters
      num_evaluation_samples=100,
  )
  hparams.override_from_dict(kwargs)
  return hparams


def load_hparams(hparams_path: str) -> tf.contrib.training.HParams:
  """Reads hparams protobuf from file.

  Args:
    hparams_path: Path to hparams file.

  Returns:
    Hparams object.
  """
  hparam_def = hparam_pb2.HParamDef()
  with tf.gfile.GFile(hparams_path, 'r') as file:
    text_format.Merge(file.read(), hparam_def)
  hparams = tf.contrib.training.HParams(hparam_def)
  return hparams


def random_configurations(
    n_sites: int,
    batch_size: int = 1
) -> np.ndarray:
  """Generates random configurations in Sz=0 sector.

  Generates a random configuration with half spins set to 1 and -1.

  Args:
    n_sites: Number of sites in the state.
    batch_size: Number of configurations to generate.

  Returns:
    Array holding random configurations of dtype=float.
  """
  configurations = np.ones((batch_size, n_sites))
  rnd = np.random.RandomState()
  for i in range(0, batch_size):
    pos = rnd.randint(0, n_sites)
    for _ in range(0, n_sites // 2):
      while (configurations[i, pos] != 1.0):
        pos = rnd.randint(0, n_sites)
      configurations[i, pos] = -1.0
  return configurations.astype(np.float32)
