PriNCe documentation 
====================

:Release: |release|
:Date: |today|

Purpose of the code (... compied from MCEq):

    This program is a toolkit to compute the evolution of particle densities
    that evolve as a cascade in the Earth's atmosphere or other target media.
    Particles are represented by average densities on discrete energy bins.
    The results are differential energy spectra or total particle numbers.
    Various models/parameterizations for particle interactions and atmospheric
    density profiles are packaged with the code.  

Installation
............

The installation via PyPi is the simplest method::

    pip install prince

Supported architectures:

- Linux 32- and 64-bit
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

If you use MCEq in your scientific publication, please cite the code **AND** the physical models.

The current citation for the MCEq is:

    | *A new view on Auger data and cosmogenic neutrinos in light of different nuclear disintegration and air-shower models*  
    | J. Heinze, A. Fedynitch, D. Boncioli and W. Winter  
    | Astrophys.J. 873 (2019) no.1, 88  
    | https://doi.org/10.3847/1538-4357/ab05ce

Find the :ref:`citations` for the physical models.

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

