
Advanced documentation
----------------------

The "advanced documentation" is the almost complete documentation of all modules. 

.. contents::
   :local:
   :depth: 2

:mod:`prince_cr.config` -- default configuration options
.....................................................

The module contains all static options of PriNCe that are not meant to be changed in run time.

.. automodule:: prince_cr.config
   :members:

:mod:`prince_cr.core` -- Core module
.................................

This module contains the main program features. Instantiating :class:`prince_cr.core.PriNCeRun`
will initialize the data structures and particle tables, create and fill the
interaction and decay matrix and check if all information for the calculation
of inclusive fluxes in the atmosphere is available.

.. automodule:: prince_cr.core
   :members: 

:mod:`prince_cr.data` -- (particle) Species management
...................................................

The :class:`prince_cr.data.SpeciesManager` handles the bookkeeping of :class:`prince_cr.data.PrinceSpecies`s.

.. automodule:: prince_cr.data
   :members:

:mod:`prince_cr.solvers` -- PDE solver implementations
...................................................

Contains solvers to solve the coupled differential equation system

The steps performed by the solver are:

.. math::

  \Phi_{i + 1} = \Delta X_i\boldsymbol{M}_{int} \cdot \Phi_i  + \frac{\Delta X_i}{\rho(X_i)}\cdot\boldsymbol{M}_{dec} \cdot \Phi_i)


.. automodule:: prince_cr.solvers.propagation
   :members:

.. automodule:: prince_cr.solvers.partial_diff
   :members:

:mod:`prince_cr.cross_sections` -- Cross section data management
................................................................

Contains function to load and combined cross section models

.. automodule:: prince_cr.cross_sections
   :members:

:mod:`prince_cr.interaction_rates` -- Computation of Interaction Matrices
.........................................................................

Contains function to precompute the sparse Interaction matrices efficiently

.. automodule:: prince_cr.interaction_rates
   :members:

:mod:`prince_cr.photonfields` -- EBL and CMB photon fields
..........................................................

Contains different EBL models as a function of redshift

.. automodule:: prince_cr.photonfields
   :members:

:mod:`prince_cr.cr_sources` -- UHECR source class definitions
.............................................................

Defines the (simple) source injection for the extragalactic propagation

.. automodule:: prince_cr.cr_sources
   :members: