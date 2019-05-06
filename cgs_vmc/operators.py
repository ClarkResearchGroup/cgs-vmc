"""Defines Operator interface and particular instances."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, List, Tuple
import tensorflow as tf
import numpy as np

import wavefunctions

class Operator():
  """Defines operators base class.

  Operator class represents operators in quantum mechanics (O). It provides an
  interface for evaluating "local values" <R|O|psi>/<R|psi>, and acting on
  wavefunctions O|psi> = |phi>. The latter can be obtained either in the form
  of a new `Wavefunction` class or as a tensor.
  """

  def build(
      self,
      wavefunction: wavefunctions.Wavefunction,
      inputs: tf.Tensor,
      psi_amplitude: tf.Tensor = None,
      psi_angle: tf.Tensor = None,
  ) -> Tuple[tf.Tensor, ...]:
    """Connects components evaluating O|psi> to the graph.

    Args:
      wavefunction: Wavefunction to which the operator is applied.
      inputs: Input on which components are evaluated.
      psi_amplitude: Optional tensor holding evaluation of wavefunction amplitude on `inputs`.
      psi_angle: Optional tensor holding evaluation of wavefunction angle on `inputs`.

    Returns:
      Tuple containing components of <R|O|psi>, the order is defined by a
      particular instance (e.g. could be Sz matrix element + Sx term).
    """
    raise NotImplementedError

  def local_value(
      self,
      wavefunction: wavefunctions.Wavefunction,
      inputs: tf.Tensor,
      psi_amplitude: tf.Tensor = None,
      psi_angle: tf.Tensor = None,
  ) -> (tf.Tensor, tf.Tensor):
    """Adds tensor holding local values of the operator to the graph.

    In Monte Carlo evaluation expectation values is given by sum_{R} O_local(R),
    where O_local = <R|O|Psi> / <R|Psi>. This function provides implementation
    of O_local.

    Args:
      wavefunction: Wavefunction.
      inputs: Input on which to evaluation O_local.
      psi_amplitude: Optional tensor holding evaluation of wavefunction amplitude on `inputs`.
      psi_angle: Optional tensor holding evaluation of wavefunction angle on `inputs`.

    Returns:
      Tensor of shape [batch_size] holding O_local values.
    """
    raise NotImplementedError


class HeisenbergBond(Operator):
  """Implements S_{i}S_{j} operator."""

  def __init__(self, bond: Tuple[int, int], j_x: np.float32, j_z: np.float32):
    """Creates an instance of a calss."""
    self._bond = bond
    self._j_x = j_x
    self._j_z = j_z

  def build(
      self,
      wavefunction: wavefunctions.Wavefunction,
      inputs: tf.Tensor
  ) -> Tuple[tf.Tensor, ...]:
    """Connects operations evaluating Sz element and Sxy term to the graph.

    This operation computes matrix element <R|S_{i}^{z}S_{j}^{z}|R> and the
    off-diagonal term <R|S_{i}^{x}S_{j}^{x}|psi>.

    Args:
      wavefunction: Wavefunction |psi>.
      inputs: Input <R| on which terms are evaluated.

    Returns:
      Tuple containing SzSz matrix element and SxySxy term of
      amplitude and angle.
    """
    batch_size = inputs.get_shape().as_list()[0]
    i, j = self._bond
    # get i_spins, j_spins configuration
    i_spins = tf.squeeze(tf.slice(inputs, [0, i], [-1, 1]))
    j_spins = tf.squeeze(tf.slice(inputs, [0, j], [-1, 1]))
    i_spin_index = tf.stack(
        [np.arange(0, batch_size), i * np.ones(batch_size).astype(int)], 1)
    j_spin_index = tf.stack(
        [np.arange(0, batch_size), j * np.ones(batch_size).astype(int)], 1)
    spin_i_update = tf.scatter_nd(i_spin_index, j_spins - i_spins, inputs.shape)
    spin_j_update = tf.scatter_nd(j_spin_index, i_spins - j_spins, inputs.shape)
    # updated config is the configuration after SxSx
    updated_config = tf.add_n([inputs, spin_i_update, spin_j_update])
    sz_matrix_element = tf.multiply(i_spins, j_spins)
    # mask is to check whether SxySxy is applicable or not
    mask = tf.less(sz_matrix_element, np.zeros(batch_size))
    mask = tf.cast(mask, sz_matrix_element.dtype)
    # QA: factor[1] has an extra 2 factor!
    factor = tf.constant([0.25 * self._j_z, 0.25 * self._j_x * 2])
    s_perp_amplitude, s_perp_angle = wavefunction(updated_config)
    # QA: whether it is better to use scatter_nd
    # QA: not necessary, but more convenient to get max later
    s_perp_amplitude = tf.multiply(mask, s_perp_amplitude)
    s_perp_angle = tf.multiply(mask, s_perp_angle)
    return sz_matrix_element, s_perp_amplitude, s_perp_angle, factor, mask

  def local_value(
      self,
      wavefunction: wavefunctions.Wavefunction,
      inputs: tf.Tensor,
      psi_amplitude: tf.Tensor = None,
      psi_angle: tf.Tensor = None,
  ) -> (tf.Tensor, tf.Tensor):
    """Adds operations that compute S_{i}S_{j}_local."""
    if psi_amplitude is None and psi_angle is  None:
      psi_amplitude, psi_angle = wavefunction(inputs)
    sz_matrix_element, s_perp_amplitude, s_perp_angle, factor, mask = self.build(wavefunction, inputs)
    sz_matrix_element = factor[0] * sz_matrix_element
    amplitude = s_perp_amplitude - psi_amplitude
    angle = s_perp_angle - psi_angle
    s_perp_real = tf.multiply(tf.exp(amplitude), tf.cos(angle))
    s_perp_real = factor[1] * tf.multiply(mask, s_perp_real)
    s_perp_img = tf.multiply(tf.exp(amplitude), tf.sin(angle))
    s_perp_img = factor[1] * tf.multiply(mask, s_perp_img)
    return (sz_matrix_element + s_perp_real, s_perp_img)


class HeisenbergHamiltonian(Operator):
  """Implements Heisenberg hamiltonian operator."""

  def __init__(
      self,
      bonds: List[Tuple[int, int]],
      j_x: np.float32,
      j_z: np.float32
  ):
    """Creates an instance of a calss."""
    self._bonds_list = bonds
    self._heisenberg_bonds = [HeisenbergBond(bond, j_x, j_z) for bond in bonds]
    self._j_x = j_x
    self._j_z = j_z

  def build(
      self,
      wavefunction: wavefunctions.Wavefunction,
      inputs: tf.Tensor
  ) -> Tuple[tf.Tensor, ...]:
    """Adds operations evaluating diagonal element and off-diagonal terms.

    Args:
      wavefunction: Wavefunction |psi>.
      inputs: Input <R| on which terms are evaluated.

    Returns:
      Tuple containing diagonal matrix element and SxySxy terms.
    """
    sz_elements = []
    s_perp_amplitudes = []
    s_perp_angles = []
    masks_term = []
    for bond in self._heisenberg_bonds:
      sz_term, amplitude, angle, factor, mask = bond.build(wavefunction, inputs)
      sz_elements.append(sz_term)
      s_perp_amplitudes.append(amplitude)
      s_perp_angles.append(angle)
      masks_term.append(mask)
    s_perp_amplitudes = tf.convert_to_tensor(s_perp_amplitudes)
    s_perp_angles = tf.convert_to_tensor(s_perp_angles)
    masks_term = tf.convert_to_tensor(masks_term)

    max_amplitude = tf.reduce_max(s_perp_amplitudes, axis=0)
    amplitudes = s_perp_amplitudes - max_amplitude
    amplitudes = tf.exp(tf.multiply(masks_term, amplitudes))
    amplitudes_real = tf.multiply(amplitudes, tf.cos(s_perp_angles))
    amplitudes_real = tf.reduce_sum(amplitudes_real, axis=0)
    amplitudes_img = tf.multiply(amplitudes, tf.sin(s_perp_angles))
    amplitudes_img = tf.reduce_sum(amplitudes_img, axis=0)
    amplitudes_total = tf.complex(amplitudes_real, amplitudes_img)
    amp = tf.log(tf.abs(amplitudes_total)) + max_amplitude
    angle = tf.angle(amplitudes_total)

    return tf.add_n(sz_elements), amp, angle, factor

  def local_value(
      self,
      wavefunction: wavefunctions.Wavefunction,
      inputs: tf.Tensor,
      psi_amplitude: tf.Tensor = None,
      psi_angle: tf.Tensor = None,
  ) -> (tf.Tensor, tf.Tensor):
    """Adds operations that compute S_{i}S_{j}_local."""
    if psi_amplitude is None and psi_angle is  None:
      psi_amplitude, psi_angle = wavefunction(inputs)
    diagonal, amp, angle, factor = self.build(wavefunction, inputs)
    diagonal = factor[0] * diagonal
    amp = amp - psi_amplitude
    angle = angle - psi_angle
    amplitude_real = factor[1] * tf.multiply(tf.exp(amp), tf.cos(angle))
    amplitude_img = factor[1] * tf.multiply(tf.exp(amp), tf.sin(angle))
    return (diagonal + amplitude_real, amplitude_img)



'''
# test
from test import *
inputs = tf.constant(random_configurations(12, 6), dtype=tf.float32)
batch_size = inputs.get_shape().as_list()[0]
print(batch_size)
i = 1
j = 3
i_spins = tf.squeeze(tf.slice(inputs, [0, i], [-1, 1]))
j_spins = tf.squeeze(tf.slice(inputs, [0, j], [-1, 1]))
i_spin_index = tf.stack(
    [np.arange(0, batch_size), i * np.ones(batch_size).astype(int)], 1)
j_spin_index = tf.stack(
    [np.arange(0, batch_size), j * np.ones(batch_size).astype(int)], 1)
spin_i_update = tf.scatter_nd(i_spin_index, j_spins - i_spins, inputs.shape)
spin_j_update = tf.scatter_nd(j_spin_index, i_spins - j_spins, inputs.shape)
updated_config = tf.add_n([inputs, spin_i_update, spin_j_update])
sz_matrix_element = tf.multiply(i_spins, j_spins)
mask0 = tf.less(sz_matrix_element, np.zeros(batch_size))
mask = tf.cast(mask0, sz_matrix_element.dtype)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print('input\n', sess.run(inputs))
print('i_spins\n',sess.run(i_spins))
print('j_spins\n',sess.run(j_spins))
print('i_spin_index\n', sess.run(i_spin_index))
print('j_spin_index\n', sess.run(j_spin_index))
print('updated_config\n', sess.run(updated_config))
print('mask0\n', sess.run(mask0))
print('mask\n', sess.run(mask))
'''
