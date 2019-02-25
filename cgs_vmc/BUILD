py_library(
    name='layers',
    srcs=[
        'layers.py',
    ],
)

py_library(
    name='utils',
    srcs=[
        'utils.py',
    ],
)

py_library(
    name='wavefunctions',
    srcs=[
        'wavefunctions.py',
    ],
    deps = [
        ":layers",
    ],
)

py_library(
    name='graph_builders',
    srcs=[
        'graph_builders.py',
    ],
    deps = [
        ":layers",
        ":wavefunctions"
    ],
)

py_library(
    name='operators',
    srcs=[
        'operators.py',
    ],
    deps = [
        ":layers",
        ":wavefunctions",
    ],
)

py_library(
    name='normalizer',
    srcs=[
        'normalizer.py',
    ],
    deps = [
        ":wavefunctions",
    ],
)

py_library(
    name='training',
    srcs=[
        'training.py',
    ],
    deps = [
        ":layers",
        ":wavefunctions",
        ":graph_builders",
    ],
)

py_library(
    name='evaluation',
    srcs=[
        'evaluation.py',
    ],
    deps = [
        ":wavefunctions",
        ":graph_builders",
    ],
)

############################################################################
# Binaries
############################################################################

py_binary(
    name = "run_training",
    srcs = ["run_training.py"],
    deps = [
        ":wavefunctions",
        ":training",
        ":utils",
        ":operators",
    ],
)

py_binary(
    name = "run_supervised_training",
    srcs = ["run_supervised_training.py"],
    deps = [
    	":evaluation",
        ":wavefunctions",
        ":training",
        ":utils",
    ],
)

py_binary(
    name = "run_energy_evaluation",
    srcs = ["run_energy_evaluation.py"],
    deps = [
        ":wavefunctions",
        ":operators",
    ],
)
