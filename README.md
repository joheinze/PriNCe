# PriNCe - **Pr**opagation **i**ncluding **N**uclear **C**ascade **e**quations

This code is written to solve the transport equation for ultra-high energy cosmic rays on cosmological scales.  

## Status

Code stable and tested for UHECR propagation. Used in [Heinze et al., Astrophys.J. 873 (2019)](https://doi.org/10.3847/1538-4357/ab05ce)

## Data and other requirements

The required data files are currently **not** contained in this repository:

If you have access to the old `SVN` repository, copy the `PriNCe/data` subfolder, otherwise ask the authors

The old `SVN` repository also contains utility and test notebooks that have not yet been migrated to `git`

## Documentation

Uses `sphinx`-docs (incomplete documentation).

```bash
cd docs
make html
```

## System requirements

- lots of RAM
- several GB disk space

## Software requirements

The majority of the code consists of pure Python modules.

Dependencies (list might be incomplete):

- python-3.7 or later (The legacy python 2.7 version is retained as a branches/Py2_legacy)
- numpy
- scipy
- matplotlib
- tqdm
- jupyter notebook or jupyter lab (optional, but needed for examples)

## Installation

The installation simplest method relies on the Python package manager [Anaconda/Miniconda](https://store.continuum.io/cshop/anaconda/) by [Continuum Analytics](http://www.continuum.io). It doesn't just improve your life, but also provides most of the scientific computing packages by default. It will not spoil your system Python paths and will install itself into a specified directory. The only action which is needed for activation, is to add this directory to your system `$PATH` variable. To uninstall just delete this directory.

1. Download one of the installers for your system architecure from here:

   - [Anaconda](http://continuum.io/downloads) - larger download, already containing most of the scientific packages and the package manager `conda` itself
   - [Miniconda](http://conda.pydata.org/miniconda.html) - minimal download, which contains the minimum requirements for the package manager `conda`.

2. Run the installer and follow the instructions:

    ```bash
    bash your-chosen-conda-distribution.sh
    ```

    Open a new terminal window to reload your new `$PATH` variable.

3. To install all dependencies into you new conda environment

    ```bash
    conda install [package name]
    ```

    This will ask conda to download and install all needed packages into its default environment.

4. Adjust your `PYTHONPATH` to make the package available in any folder:

    ```bash
    export PYTHONPATH=$PYTHONPATH:<main dir>
    ```

    (We recommend adding this export to .zshrc or .bashrc)

5. (**Optional**) Acceleration of the integration routines can be achieved using [Intel Math Kernel Library](https://software.intel.com/en-us/intel-mkl) (MKL).  
Anaconda offers MKL-linked numpy binaries free for academic use. It is necessary to register using your *.edu* mail address to receive a license. The demo period is 30 days. If you want to give it a try

    ```bash
    conda install mkl
    ```

6. Run some example

    ```bash
    jupyter lab
    ```

    click on the examples directory and start with `create_kernel.ipynb`. Click through the blocks and see what happens.

## Examples

## Citation

If you are using this code in your work, please cite:

*A new view on Auger data and cosmogenic neutrinos in light of different nuclear disintegration and air-shower models*  
J. Heinze, A. Fedynitch, D. Boncioli and W. Winter  
Astrophys.J. 873 (2019) no.1, 88  
doi: [10.3847/1538-4357/ab05ce](https://doi.org/10.3847/1538-4357/ab05ce)

## Contributors

- *Anatoli Fedynitch*
- *Jonas Heinze*

## Copyright and license

Code released under [the BSD 3-clause license (see LICENSE)](LICENSE).

## Acknowledgements

This code has been initially developed as part of the [NEUCOS project](https://astro.desy.de/theory/neucos/index_eng.html) and has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([Grant No. 646623](https://cordis.europa.eu/project/id/646623)).
