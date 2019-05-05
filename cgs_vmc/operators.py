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
      # QA
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
  ) -> tf.Tensor:
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

  def apply_in_place(
      self,
      wavefunction: wavefunctions.Wavefunction,
      inputs: tf.Tensor,
      psi_amplitude: tf.Tensor = None,
      psi_angle: tf.Tensor = None,
  ) -> tf.Tensor:
    """Returns a tensor holding values of <`inputs`|O|`wavefunction`>.

    Args:
      wavefunction:
      inputs:
      psi_amplitude:
      psi_angle:

    Returns:
      Tensor holding values of the wavefunction after application of the
      operator.
    """
    raise NotImplementedError

  def apply(
      self,
      wavefunction: wavefunctions.Wavefunction
  ) -> wavefunctions.Wavefunction:
    """Returns new wavefunction that is equal to O|`wavefunction`>."""
    raise NotImplementedError

# QA
class TransformedWavefunction(wavefunctions.Wavefunction):
  """Class representing wavefunction after application of the operator.

  # TODO(kochkov92) Add description of how this works.
  """

  def __init__(
      self,
      build_function: Callable,
      operator: Operator,
      wavefunction: wavefunctions.Wavefunction,
      name: str = 'transformed_wavefunction'
  ):
    """Creates an WavefunctionSum module."""
    # TODO(kochkov92) change the naming to reflect the application of O.
    super(TransformedWavefunction, self).__init__(name=name)
    # TODO
    self.build_function = build_function
    self._wf = wavefunction
    self._operator = operator
    self._sub_wavefunctions.append(wavefunction)

  def _build(self, inputs: tf.Tensor) -> tf.Tensor:
    """Builds computational graph evaluating the wavefunction on inputs.

    Args:
      inputs: Input tensor, must have shape (batch, num_sites, ...).

    Returns:
      Tensor representing wave-function amplitudes evaluated on `inputs`.
    """
    return self.build_function(self._operator, self._wf, inputs)

  @classmethod
  def from_hparams(cls, hparams: tf.contrib.training.HParams):
    """Constructs an instance of a class from hparams."""
    raise ValueError('Hparams initialization is not supported for this class.')


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
      Tuple containing SzSz matrix element and SxySxy term.
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
    # mask is to check whether SxSx is applicable or not
    mask = tf.less(sz_matrix_element, np.zeros(batch_size))
    mask = tf.cast(mask, sz_matrix_element.dtype)
    s_perp_term = 2. * tf.multiply(mask, wavefunction(updated_config))
    return  0.25 * self._j_z * sz_matrix_element, 0.25 * self._j_x * s_perp_term

  def local_value(
      self,
      wavefunction: wavefunctions.Wavefunction,
      inputs: tf.Tensor,
      psi_amplitude: tf.Tensor = None,
      psi_angle: tf.Tensor = None,
  ) -> tf.Tensor:
    """Adds operations that compute S_{i}S_{j}_local."""
    if psi_amplitude is None and psi_angle is  None:
      psi_amplitude, psi_angle = wavefunction(inputs)
    sz_matrix_element, s_perp_term = self.build(wavefunction, inputs)
    # QA
    return sz_matrix_element + s_perp_term / psi

  def apply_in_place(
      self,
      wavefunction: wavefunctions.Wavefunction,
      inputs: tf.Tensor,
      psi_amplitude: tf.Tensor = None,
      psi_angle: tf.Tensor = None,
  )-> tf.Tensor:
    """Returns matrix element of the operator on `inputs`. See base class."""
    if psi_amplitude is None and psi_angle is  None:
      psi_amplitude, psi_angle = wavefunction(inputs)
    sz_matrix_element, s_perp_term = self.build(wavefunction, inputs)
    # QA
    return sz_matrix_element * psi + s_perp_term

  def apply(
      self,
      wavefunction: wavefunctions.Wavefunction
  )-> wavefunctions.Wavefunction:
    """Applies operator to a wavefunction."""

    def build_function(
        building_operator: Operator,
        wavefunction: wavefunctions.Wavefunction,
        inputs: tf.Tensor
    ) -> tf.Tensor:
      sz_element, s_perp_term = building_operator.build(wavefunction, inputs)
    # QA
      return sz_element * wavefunction(inputs) + s_perp_term

    return TransformedWavefunction(build_function, self, wavefunction)


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
    s_perp_terms = []
    for bond in self._heisenberg_bonds:
      sz_element, s_perp_term = bond.build(wavefunction, inputs)
      sz_elements.append(sz_element)
      s_perp_terms.append(s_perp_term)
    return tf.add_n(sz_elements), tf.add_n(s_perp_terms)

  def local_value(
      self,
      wavefunction: wavefunctions.Wavefunction,
      inputs: tf.Tensor,
      psi_amplitude: tf.Tensor = None,
      psi_angle: tf.Tensor = None,
  ) -> tf.Tensor:
    """Adds operations that compute S_{i}S_{j}_local."""
    if psi_amplitude is None and psi_angle is  None:
      psi_amplitude, psi_angle = wavefunction(inputs)
    diagonal, s_perp_term = self.build(wavefunction, inputs)
    return diagonal + s_perp_term / psi

  def apply_in_place(
      self,
      wavefunction: wavefunctions.Wavefunction,
      inputs: tf.Tensor,
      psi_amplitude: tf.Tensor = None,
      psi_angle: tf.Tensor = None,
  )-> tf.Tensor:
    """Returns matrix element of the operator on `inputs`. See base class."""
    if psi_amplitude is None and psi_angle is  None:
      psi_amplitude, psi_angle = wavefunction(inputs)
    diagonal, s_perp_term = self.build(wavefunction, inputs)
    return diagonal * psi + s_perp_term

  def apply(
      self,
      wavefunction: wavefunctions.Wavefunction
  )-> wavefunctions.Wavefunction:
    """Applies operator to a wavefunction."""

    def build_function(
        q_operator: Operator,
        wavefunction: wavefunctions.Wavefunction,
        inputs: tf.Tensor
    ) -> tf.Tensor:
      sz_matrix_element, s_perp_term = q_operator.build(wavefunction, inputs)
      return wavefunction(inputs) * sz_matrix_element + s_perp_term

    return TransformedWavefunction(build_function, self, wavefunction)


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
