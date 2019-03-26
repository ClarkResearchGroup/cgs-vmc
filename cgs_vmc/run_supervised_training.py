"""Runs supervised or unsupervised neural network training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from absl import app
from absl import flags

import evaluation
import training
import wavefunctions
import utils

# System parameters
flags.DEFINE_string(
    'checkpoint_dir', '',
    'Full path to the checkpoint directory.')
flags.DEFINE_string(
    'supervisor_dir', '',
    'Full path to the directory with supervisors checkpoints.')

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
    'optimizer', 'SWO',
    'Supervised optimizer to use. Available options listed in '
    'training.SUPERVISED_OPTIMIZERS dict.')

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
  """Runs supervised wavefunction optimization.

  This pipeline optimizes wavefunction by matching amplitudes of a target state.

  """
  del argv  # Not used.

  supervisor_path = os.path.join(FLAGS.supervisor_dir, 'hparams.pbtxt')
  supervisor_hparams = utils.load_hparams(supervisor_path)

  hparams = utils.create_hparams()
  hparams.set_hparam('num_sites', supervisor_hparams.num_sites)
  hparams.set_hparam('checkpoint_dir', FLAGS.checkpoint_dir)
  hparams.set_hparam('supervisor_dir', FLAGS.supervisor_dir)
  hparams.set_hparam('basis_file_path', FLAGS.basis_file_path)
  hparams.set_hparam('num_epochs', FLAGS.num_epochs)
  hparams.set_hparam('wavefunction_type', FLAGS.wavefunction_type)
  hparams.parse(FLAGS.hparams)
  hparams_path = os.path.join(hparams.checkpoint_dir, 'hparams.pbtxt')

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

  if os.path.exists(hparams_path) and not FLAGS.override:
    print('Hparams file already exists')
    exit()

  with tf.gfile.GFile(hparams_path, 'w') as file:
    file.write(str(hparams.to_proto()))

  target_wavefunction = wavefunctions.build_wavefunction(supervisor_hparams)
  wavefunction = wavefunctions.build_wavefunction(hparams)

  wavefunction_optimizer = training.SUPERVISED_OPTIMIZERS[FLAGS.optimizer]()

  shared_resources = {}

  graph_building_args = {
      'wavefunction': wavefunction,
      'target_wavefunction': target_wavefunction,
      'hparams': hparams,
      'shared_resources': shared_resources
  }

  train_ops = wavefunction_optimizer.build_opt_ops(**graph_building_args)

  session = tf.Session()
  init = tf.global_variables_initializer()
  init_l = tf.local_variables_initializer()
  session.run([init, init_l])

  target_saver = tf.train.Saver(target_wavefunction.get_trainable_variables())
  supervisor_checkpoint = tf.train.latest_checkpoint(FLAGS.supervisor_dir)
  target_saver.restore(session, supervisor_checkpoint)
  checkpoint_saver = tf.train.Saver(
      wavefunction.get_trainable_variables(), max_to_keep=5)

  if FLAGS.resume_training:
    latest_checkpoint = tf.train.latest_checkpoint(hparams.checkpoint_dir)
    checkpoint_saver.restore(session, latest_checkpoint)

  for epoch_number in range(FLAGS.num_epochs):
    wavefunction_optimizer.run_optimization_epoch(
        train_ops, session, hparams, epoch_number)
    if epoch_number % FLAGS.checkpoint_frequency == 0:
      checkpoint_name = 'model_after_{}_epochs'.format(epoch_number)
      save_path = os.path.join(hparams.checkpoint_dir, checkpoint_name)
      checkpoint_saver.save(session, save_path)

  if FLAGS.generate_vectors:
    vector_generator = evaluation.VectorWavefunctionEvaluator()
    eval_ops = vector_generator.build_eval_ops(
        wavefunction, None, hparams, shared_resources)
    vector_generator.run_evaluation(
        eval_ops, session, hparams, FLAGS.num_epochs)


if __name__ == '__main__':
  app.run(main)
