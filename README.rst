PriNCe - PRopagation Including Nuclear Cascade equations
========================================================

This code is supposed to solve the transport equation for ultra-high enregy cosmic rays on cosmological scales. The development
is part of the `NEUCOS project <https://astro.desy.de/theory/neucos/index_eng.html>`_

Status
------

early development

Documentation
-------------

Uses `sphinx`-docs. 

	.. code-block:: bash

	   $ cd docs
	   $ make html

System requirements
-------------------

- some kind of modern CPU
- X GB RAM
- ~1GB of disk space

Software requirements
---------------------

The majority of the code consists of pure Python modules. 

Dependencies:

* python-2.7 (Python 3 not compatible yet)
* numpy
* scipy
* matplotlib
* jupyter notebook (optional, but needed for examples)


Installation
------------
The installation simplest method relies on the Python package manager `Anaconda/Miniconda <https://store.continuum.io/cshop/anaconda/>`_ by `Continuum Analytics <http://www.continuum.io>`_. It doesn't just improve your life, but also provides most of the scientific computing packages by default. It will not spoil your system Python paths and will install itself into a specified directory. The only action which is needed for activation, is to add this directory to your system `$PATH` variable. To uninstall just delete this directory.

#. Download one of the installers for your system architecure from here:

	* `Anaconda <http://continuum.io/downloads>`_ - larger download, already containing most of the scientific packages and the package manager `conda` itself
	* `Miniconda <http://conda.pydata.org/miniconda.html>`_ - minimal download, which contains the minimum requirements for the package manager `conda`.

#. Run the installer and follow the instructions:

	.. code-block:: bash

	   $ bash your-chosen-conda-distribution.sh

	Open a new terminal window to reload your new `$PATH` variable.


#. `Cd` to you desired working directory. And clone this project including submodules:

	.. code-block:: bash

	   $ git clone --recursive https://github.com/afedynitch/MCEq.git

	It will clone this github repository into a folder called `MCEq` and download all files.
	Enter this directory. 

#. To install all dependencies into you new conda environment

	.. code-block:: bash

	   $ conda install --file conda_req.txt

	This will ask conda to download and install all needed packages into its default environment. 

#. (**Optional**) If you know what a `virtualenv` is, the corresponding commands to download and install all packages in a newly created environment `mceq_env` are

	.. code-block:: bash

	   $ conda create -n mceq_env --file conda_req.txt
	   $ source activate mceq_env

	To quit this environment just

	.. code-block:: bash

	   $ deactivate

#. (**Optional**) Acceleration of the integration routines can be achieved using `Intel Math Kernel Library <https://software.intel.com/en-us/intel-mkl>`_ (MKL). Anaconda offers MKL-linked numpy binaries free for academic use. It is necessary to register using your *.edu* mail adress to receive a license. The demo period is 30 days. If you want to give it a try

	.. code-block:: bash

		   $ conda install mkl

	Change in `mceq_config.py` the `kernel` entry to 'MKL'.

#. Run some example

	.. code-block:: bash

	   $ ipython notebook

	click on the examples directory and select `basic_flux.ipynb`. Click through the blocks and see what happens.

Troubleshoting
--------------
You might run into `problems with Anaconda <https://github.com/conda/conda/issues/394>`_  if you have previous 
Python installations. A workaround is to set the environement variable
	.. code-block:: bash

	   $ export PYTHONNOUSERSITE=1
	   
Thanks to F.C. Penha for pointing this out.

Citation
--------


Contributers
------------

*Anatoli Fedynitch*

Copyright and license
---------------------
Code and documentation copyright 2017 Anatoli Fedynitch. Private code. All rights reserved.
