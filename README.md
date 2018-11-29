# Computational Graph States for Variational Monte Carlo

Computational framework that implements Computational Graph States (CGS)
and Supervised Wavefunction Optimization (SWO).


## Installation:
This project uses [Bazel](https://bazel.build/), but can be also executed
directly using python3.

The following libraries are required to run the code:
* [TensorFlow](https://www.tensorflow.org/)
* [Sonnet](https://github.com/deepmind/sonnet)
* [absl-py](https://pypi.org/project/absl-py/)

Can be obtained using pip.

## Running:
```
cd cgs_vmc
bazel run :run_training -- --checkpoint_dir=PATH_TO_CHECKPOINT_DIRECTORY

```

To change optimization methods, variational ansatz etc. Provide corresponding
flags to the run_training call.


Current implementation does not work with $log$ of the wavefunctions and
therefore might experience numerical instabilities for large system sizes. This
can be tuned out by appropriate normalization, but will be removed in future
releases.


For suggestions and comments reach out to
[kochkov92](https://github.com/kochkov92)