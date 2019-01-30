"""Runs supervised or unsupervised neural network training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags

import operators
import training
import wavefunctions
import utils

# System parameters
flags.DEFINE_string(
    'checkpoint_dir', '',
    'Full path to the checkpoint directory.')
flags.DEFINE_integer(
    'num_sites', 24,
    'Number of sites in the system.')

# Training parameters
flags.DEFINE_integer(
    'num_epochs', 1000,
    'Total of number of epochs to train on.')
flags.DEFINE_integer(
    'checkpoint_frequency', 1,
    'Number of epochs between checkpoints.')

flags.DEFINE_boolean(
    'resume_training', False,
    'Indicator to resotre variables from the latest checkpoint')

flags.DEFINE_string(
    'wavefunction_type', 'fully_connected',
    'Network architecture to train. Available architectures are listed in '
    'wavefunctions.WAVEFUNCTION_TYPES dict. and '
    'wavefunctions.build_wavefunction() function.')
flags.DEFINE_string(
    'optimizer', 'ITSWO',
    'Ground state optimizer to use. Available options listed in '
    'training.GROUND_STATE_OPTIMIZERS dict.')

flags.DEFINE_string(
    'list_of_evaluators', '',
    'Com-separated list of evaluators to run during training.')

flags.DEFINE_boolean(
    'generate_vectors', False,
    'Indicator generate full wavefunction vectors as a part of evaluation.')
flags.DEFINE_string(
    'basis_file_path', '',
    'Path to the basis file for full wavefunction evaluation.')

flags.DEFINE_string(
    'hparams', '',
    'Override values of hyper-parameters in the form of a '
    'comma-separated list of name=value pairs, e.g., '
    '"num_layers=3,filter_size=64".')

flags.DEFINE_boolean(
    'override', True,
    'Whether to automatically override existing Hparams.')

FLAGS = flags.FLAGS


def main(argv):
  """Runs wavefunction optimization.

  This pipeline optimizes wavefunction specified in flags on a Marshal sign
  included Heisenberg model. Bonds should be specified in the file J.txt in
  checkpoint directory, otherwise will default to 1D PBC system. For other
  tunable parameters see flags description.
  """
  del argv  # Not used.
  n_sites = FLAGS.num_sites
  hparams = utils.create_hparams()
  hparams.set_hparam('checkpoint_dir', FLAGS.checkpoint_dir)
  hparams.set_hparam('basis_file_path', FLAGS.basis_file_path)
  hparams.set_hparam('num_sites', FLAGS.num_sites)
  hparams.set_hparam('num_epochs', FLAGS.num_epochs)
  hparams.set_hparam('wavefunction_type', FLAGS.wavefunction_type)
  hparams.set_hparam('wavefunction_optimizer_type', FLAGS.optimizer)
  hparams.parse(FLAGS.hparams)
  hparams_path = os.path.join(hparams.checkpoint_dir, 'hparams.pbtxt')

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

  if os.path.exists(hparams_path) and not FLAGS.override:
    print('Hparams file already exists')
    exit()

  with tf.gfile.GFile(hparams_path, 'w') as file:
    file.write(str(hparams.to_proto()))

  bonds_file_path = os.path.join(FLAGS.checkpoint_dir, 'J.txt')
  if os.path.exists(bonds_file_path):
    heisenberg_data = np.genfromtxt(bonds_file_path, dtype=int)
    heisenberg_bonds = [[bond[0], bond[1]] for bond in heisenberg_data]
  else:
    heisenberg_bonds = [(i, (i + 1) % n_sites) for i in range(0, n_sites)]

  wavefunction = wavefunctions.build_wavefunction(hparams)
  hamiltonian = operators.HeisenbergHamiltonian(heisenberg_bonds, -1., 1.)

  wavefunction_optimizer = training.GROUND_STATE_OPTIMIZERS[FLAGS.optimizer]()

  # TODO(dkochkov) change the pipeline to avoid adding elements to dictionary
  shared_resources = {}

  graph_building_args = {
      'wavefunction': wavefunction,
      'hamiltonian': hamiltonian,
      'hparams': hparams,
      'shared_resources': shared_resources
  }

  train_ops = wavefunction_optimizer.build_opt_ops(**graph_building_args)

  session = tf.Session()
  init = tf.global_variables_initializer()
  init_l = tf.local_variables_initializer()
  session.run([init, init_l])

  checkpoint_saver = tf.train.Saver(
      wavefunction.get_trainable_variables(), max_to_keep=5)

  if FLAGS.resume_training:
    latest_checkpoint = tf.train.latest_checkpoint(hparams.checkpoint_dir)
    checkpoint_saver.restore(session, latest_checkpoint)

  # TODO(kochkov92) use custom output file.
  training_metrics_file = os.path.join(hparams.checkpoint_dir, 'metrics.txt')
  for epoch_number in range(FLAGS.num_epochs):
    checkpoint_name = 'model_prior_{}_epochs'.format(epoch_number)
    save_path = os.path.join(hparams.checkpoint_dir, checkpoint_name)
    checkpoint_saver.save(session, save_path)

    metrics_record = wavefunction_optimizer.run_optimization_epoch(
        train_ops, session, hparams)

    metrics_file_output = open(training_metrics_file, 'a')
    metrics_file_output.write('{}\n'.format(metrics_record))
    metrics_file_output.close()


if __name__ == '__main__':
  app.run(main)
