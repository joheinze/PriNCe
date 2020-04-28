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

    pip install prince

Supported architectures
.......................

Due to memory requirements 32-bit architectures are not recommended.
It should run on Python 2.7 and above on

- Linux
- Mac OS X
- Windows

Installation from source
.........................

To modify the code and contribute, the code needs to be installed from the github source::

    git clone https://github.com/joheinze/PriNCe.git
    cd PriNCe
    pip install -e .

With the last command pip will symlink this installation to your site-packages folder. There is no need to modify the PYTHONPATH.

Quick start
...........

Open an new python file or jupyter notebook/lab::

    from prince import bla
    # matplotlib used plotting. Not required to run the code.
    import matplotlib.pyplot as plt


    # Initalize the Prince user interface
    prince = PrinceRun(
        ...
    )

    # ...
    # ...

    # Solve the equation system
    prince.solve()

    # Plot results
    plt.loglog(prince.e_grid, proton_flux, label='muons')
    plt.loglog(prince.e_grid, numu_flux, label='muon neutrinos')
    plt.loglog(prince.e_grid, nue_flux, label='electron neutrinos')

    plt.xlim(1., 1e9)
    plt.xlabel('Kinetic energy (GeV)')
    plt.ylim(1e-6, 1.)
    plt.ylabel(r'$(E/\text{GeV})^3\,\Phi$ (GeV cm$^{-2}$\,$s$^{-1}\,$sr$^{-1}$) (GeV)')
    plt.legend()
    plt.show()

Examples
........

Follow the :ref:`tutorial` and/or download and run the notebooks from 
`github <https://github.com/joheinze/prince/tree/master/examples>`_.

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

