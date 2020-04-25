
Advanced documentation
----------------------

The "advanced documentation" is the almost complete documentation of all modules. 

.. contents::
   :local:
   :depth: 2

:mod:`prince_config` -- default configuration options
.....................................................

The module contains all static options of PriNCe that are not meant to be changed in run time.

.. automodule:: prince_config
   :members:

:mod:`prince.core` -- Core module
.................................

This module contains the main program features. Instantiating :class:`prince.core.PriNCeRun`
will initialize the data structures and particle tables, create and fill the
interaction and decay matrix and check if all information for the calculation
of inclusive fluxes in the atmosphere is available.

.. automodule:: prince.core
   :members: 

:mod:`prince.data` -- (particle) Species management
...................................................

The :class:`prince.data.SpeciesManager` handles the bookkeeping of :class:`prince.data.PrinceSpecies`s.

.. automodule:: prince.data
   :members:

:mod:`prince.solvers` -- PDE solver implementations
...................................................

Here are various solvers ... blabla.... 

The steps performed by the solver are:

.. math::

  \Phi_{i + 1} = \Delta X_i\boldsymbol{M}_{int} \cdot \Phi_i  + \frac{\Delta X_i}{\rho(X_i)}\cdot\boldsymbol{M}_{dec} \cdot \Phi_i)

with

.. math::
  \boldsymbol{M}_{int} = (-\boldsymbol{1} + \boldsymbol{C}){\boldsymbol{\Lambda}}_{int}
  :label: int_matrix

and

.. math::
  \boldsymbol{M}_{dec} = (-\boldsymbol{1} + \boldsymbol{D}){\boldsymbol{\Lambda}}_{dec}.
  :label: dec_matrix


.. automodule:: prince.solvers
   :members:

Miscellaneous
.............

Different helper functions.

.. automodule:: MCEq.misc
   :members: