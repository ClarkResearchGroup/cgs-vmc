"""Runs evaluation on the optimized wave-function."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags

import operators
import evaluation
import wavefunctions
import utils

# System parameters
flags.DEFINE_float(
    'heisenberg_jx', 1.0,
    'Jx value in Heisenberg Hamiltonian.')

# Wavefunction parameters
flags.DEFINE_string(
    'checkpoint_dir', '',
    'Full path to the checkpoint directory.')

flags.DEFINE_string(
    'bond_file', '',
    'Bond file that specifies the bond connection of the Hamiltonian.')

flags.DEFINE_string(
    'output_file', '',
    'Full path to the file where to save the results.')

flags.DEFINE_string(
    'hparams', '',
    'Override values of hyper-parameters in the form of a '
    'comma-separated list of name=value pairs, e.g., '
    '"num_layers=3,filter_size=64".')


FLAGS = flags.FLAGS


def main(argv):
  """Evaluates energy and prints the result."""
  del argv  # Not used by main.
  hparams_path = os.path.join(FLAGS.checkpoint_dir, 'hparams.pbtxt')
  hparams = utils.load_hparams(hparams_path)
  hparams.parse(FLAGS.hparams)  # optional way to override some hparameters
  n_sites = hparams.num_sites

  # TODO(dkochkov) make a more comprehensive Hamiltonian construction method
  bonds_file_path = os.path.join(FLAGS.checkpoint_dir, FLAGS.bond_file)
  heisenberg_jx = FLAGS.heisenberg_jx
  if os.path.exists(bonds_file_path):
    heisenberg_data = np.genfromtxt(bonds_file_path, dtype=int)
    heisenberg_bonds = [[bond[0], bond[1]] for bond in heisenberg_data]
  else:
    heisenberg_bonds = [(i, (i + 1) % n_sites) for i in range(0, n_sites)]

  wavefunction = wavefunctions.build_wavefunction(hparams)
  hamiltonian = operators.HeisenbergHamiltonian(heisenberg_bonds,
                                                heisenberg_jx, 1.)

  evaluator = evaluation.MonteCarloOperatorEvaluator()

  shared_resources = {}

  graph_building_args = {
      'wavefunction': wavefunction,
      'operator': hamiltonian,
      'hparams': hparams,
      'shared_resources': shared_resources
  }

  evaluation_ops = evaluator.build_eval_ops(**graph_building_args)

  init = tf.global_variables_initializer()
  session = tf.Session()
  session.run(init)

  checkpoint_manager = tf.train.Saver(wavefunction.get_trainable_variables())

  latest_checkpoint = tf.train.latest_checkpoint(hparams.checkpoint_dir)
  checkpoint_manager.restore(session, latest_checkpoint)

  data = evaluator.run_evaluation(evaluation_ops, session, hparams, epoch_num=0)
  mean_energy = np.mean(data)
  uncertainty = np.sqrt(np.std(data)) / len(data)
  print('Energy: {} +/- {}'.format(mean_energy, uncertainty))
  # with open(FLAGS.output_file, 'w') as output_file:
  #   output_file.write('Energy mean: {}\n'.format(mean_energy)
  #   output_file.write('Uncertainty: {}\n'.format(uncertainty))


if __name__ == '__main__':
  app.run(main)
