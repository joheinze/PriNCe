PriNCe documentation 
====================

:Release: |release|
:Date: |today|

PriNCe is derived from **Pr**\opagation **i**\ncluding **N**\uclear **C**\ascade **e**\quations.
This code is to be a numerical solver of the transport equations for Ultra-High Energy Cosmic
Rays through the radiation fields of the intergalactic space. Its specific features are:

- **Time dependent UHECR transport equation solver**; efficient enough to compute a single spectrum within seconds,
- **Fast and easy variation of input parameters**; such as cross section models and extragalactic photon backgrounds,
- **Accessibility and modularity**; ability to easily modify and extend specific parts of the code through interfaces.

To achieve these goals, PriNCe is written in pure Python using vectorized expressions for the performance
intensive parts. It accelerates those using libraries like Numpy and Scipy.

## Data and other requirements

The code requires data files, which are bundled with this package in binary form. On first import around ~150MB of data will be automatically downloaded and placed in the data directory.

These data are constructed from a variety of sources. The code is publicly available in a different repository called `PriNCe-data-utils <https://github.com/joheinze/PriNCe-data-utils>`_.

System requirements
...................

- ~16GB RAM
- several GB disk space

The package is portable and will run on any flavor of Linux, Mac OS X and Windows that satisfy the dependencies below. Due to memory requirements 32-bit architectures are not recommended but may work under certain circumstances.

The code is pure Python relies on vectorization via numpy/scipy, and optionally Intel's MKL and nVidia's CUDA (via `cupy <https://cupy.chainer.org/>`_).

Dependencies:

- Python 3 (2.7 is not supported.)
- numpy
- scipy
- matplotlib
- tqdm
- jupyter (optional, but needed for examples)

Optional:

- mkl (Intel's MKL runtime from pip/anaconda works)
- cupy (For GPU acceleration with CUDA. Support is experimental and memory requirements are not yet understood. Tested on RTX 2080TI/11GB, but typical memory requirements should not exceed 3-4 GB. )

Installation
............

All you need is a python distribution including `pip`. For scientific computing [Anaconda/Miniconda](https://www.anaconda.com/products/individual/) may be good choice, but not necessary.

The installation via PyPi is the simplest method::

    pip install prince-cr

To install from source::

    git clone https://github.com/joheinze/PriNCe
    cd PriNCe
    pip install -e .

with the option `-e` (or `--editable`) this will link the local foulder to your pip installation, such that any local code edits will take effect. Use this for testing and development.

To test an installation:

    pip install pytest
    python -m pytest --pyargs prince_cr

Examples
........

Follow the :ref:`tutorial` and/or download and run the notebooks from 
`github <https://github.com/joheinze/PriNCe-examples>`_.

Citations
.........

If you use PriNCe in your scientific publications, please cite the code **AND** the physical models.
Have a look at the :ref:`citations` for instructions how to cite.

Main documentation
..................

.. toctree::
   :maxdepth: 2

   tutorial
   citations
   advanced

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

