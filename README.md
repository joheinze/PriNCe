
[![PyPI](https://img.shields.io/pypi/v/prince-cr)](https://pypi.org/project/prince-cr/)
[![Documentation Status](https://readthedocs.org/projects/prince/badge/?version=latest)](https://prince.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://dev.azure.com/jonasheinze/PriNCe/_apis/build/status/joheinze.PriNCe?branchName=master)](https://dev.azure.com/jonasheinze/PriNCe/_build/latest?definitionId=1&branchName=master)
![Deployment Status](https://vsrm.dev.azure.com/jonasheinze/_apis/public/Release/badge/23c377f2-2078-4a05-9329-f3888f8b6c6d/1/1)

# PriNCe - **Pr**opagation **i**ncluding **N**uclear **C**ascade **e**quations

This code solves the transport equation for ultra-high energy cosmic rays on cosmological scales.

## About this version

The package is numerically accurate for its current purpose (UHECR propagation). Since this the first public version, one should be cautious when deviating from the examples and check if the result makes sense, simply because so far only us (the devs) have been using it and the interfaces aren't sufficiently polished to be used *error-free* for tasks that we didn't foresee. Please file issues for anything strange, unclear, wrong, etc.. It will help us to debug the code and simplify the user interface.

## Installation

is as simple as 

```bash
pip install prince-cr
```

Check [the docs](https://prince.readthedocs.io/en/latest/) for more details on the requirements. 

## [Documentation](https://prince.readthedocs.io/en/latest/)

The docs are hosted at [readthedocs](https://prince.readthedocs.io/en/latest/). They are still *incomplete* and we are working on improvements. Feedback/comments are welcome.

## [Examples](https://github.com/joheinze/PriNCe-examples)

To get started check out the [examples repository](https://github.com/joheinze/PriNCe-examples) and/or follow [the tutorial](https://prince.readthedocs.io/en/latest/tutorial.html). 

## Citation

If you are using this code in your work, please cite:

*A new view on Auger data and cosmogenic neutrinos in light of different nuclear disintegration and air-shower models*  
J. Heinze, A. Fedynitch, D. Boncioli and W. Winter  
Astrophys.J. 873 (2019) no.1, 88  
doi: [10.3847/1538-4357/ab05ce](https://doi.org/10.3847/1538-4357/ab05ce)

## Authors

- *Anatoli Fedynitch*
- *Jonas Heinze*

## Copyright and license

Code released under [the BSD 3-clause license (see LICENSE)](LICENSE).

## Acknowledgements

This code has been initially developed as part of the [NEUCOS project](https://astro.desy.de/theory/neucos/index_eng.html) and has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([Grant No. 646623](https://cordis.europa.eu/project/id/646623)).
