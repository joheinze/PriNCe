
# PriNCe - **Pr**opagation **i**ncluding **N**uclear **C**ascade **e**quations

[![Documentation Status](https://readthedocs.org/projects/prince/badge/?version=latest)](https://prince.readthedocs.io/en/latest/?badge=latest)

This code is written to solve the transport equation for ultra-high energy cosmic rays on cosmological scales.  

## Development status

Although the code is numerically accurate for its main purpose (UHECR propagation), one should consider it
as *early alpha*, simply because so far only us (the devs) have been using it and the interfaces aren't
sufficiently polished to be used *error-free* for tasks that we didn't foresee. Please file issues for
anything strange, unclear, wrong, etc.. It will help us to debug the code and simplify the user interface.

## Data and other requirements

The code requires data files which are **not** contained in the repository. You need to manually download the tarball from the [latest release](https://github.com/joheinze/PriNCe/releases) and unpack it to `./data`.

We are planning to make this download automatic in a future release.

## [Documentation](https://prince.readthedocs.io/en/latest/)

Uses `sphinx`-docs, which can be read at [readthedocs](https://prince.readthedocs.io/en/latest/)

The documentation is still *incomplete*. Feedback/comments are welcome via issues.

## System requirements

- ~16GB RAM
- several GB disk space

## Software requirements

The code is pure Python heavily using vectorization via numpy/scipy.

Dependencies (list might be incomplete):

- python-3.7 or later
- numpy
- scipy
- matplotlib
- tqdm
- jupyter (optional, but needed for examples)

Optional:

- mkl (Intel's MKL runtime from pip/anaconda works)
- cupy (For GPU acceleration with CUDA. Support is experimental and memory requirements are not yet understood.)

## Installation

Since this code is written in pure Python 3, it requries no installation. All you need is a python distribution including `pip`. For scientific computing [Anaconda/Miniconda](https://www.anaconda.com/products/individual/) is a good choice.

Simply `git clone` (or manually download) this repository and link it using `pip`:

```bash
git clone https://github.com/joheinze/PriNCe
pip install -e PriNCe
```

with the option `-e` (or `--editable`) this will link the local foulder to your pip installation, such that any local code edits will take effect. Use this for testing and development.

To test an installation:

```bash
pip install pytest
python -m pytest --pyargs prince_cr
```

This will automatically install the dependencies. If you do not want to use `pip` you can also just add this folder to your `PYTHONPATH`.

## [Examples](https://github.com/joheinze/PriNCe-examples)

Examples are separately maintained: [Examples repository](https://github.com/joheinze/PriNCe-examples)

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
