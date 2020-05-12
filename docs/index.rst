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

Development status
..................

Although the code is numerically accurate for its main purpose (UHECR propagation), one should consider it
as *early alpha*, simply because so far only us (the devs) have been using it and the interfaces aren't
sufficiently polished to be used *error-free* for tasks that we didn't foresee. Please file issues for
anything strange, unclear, wrong, etc.. It will help us to debug the code and simplify the user interface.

Installation
............

The installation via PyPi is the simplest method::

    pip install prince-cr

To install from source::

    git clone https://github.com/joheinze/PriNCe
    cd PriNCe
    pip install -e .

The '-e' flag will symlink the source folder to pip and behave as an ordinary package from PyPi.

Supported architectures
.......................

Due to memory requirements 32-bit architectures are not recommended.
The code will run on Python 3 on

- Linux
- Mac OS X
- Windows

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

