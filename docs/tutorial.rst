.. _tutorial:

Tutorial
--------

The main interface to the cde is provided by the class :class:`prince_cr.core.PriNCeRun`.
To create an instance type::

    from prince_cr.core import PriNCeRun
    prince_run = PriNCeRun()

This assume will assume the Talys disintegration model at low energies 
and a superposition model based on Sophia at for photo-meson production at high energies.
The photon field will be the CMB and the Gilmore et. al. CIB model.

`prince_run` will take a few minutes to instanciate. This oject can be saved using `pickle` to skip this step.

Changing model assumption
.........................

Either of these assumptions can be changed by manually creating objects::

    from prince_cr import photonfields
    pf = photonfields.CombinedPhotonField(
                    [photonfields.CMBPhotonSpectrum, 
                    photonfields.CIBDominguez2D])

    from prince_cr import cross_sections
    cs = cross_sections.CompositeCrossSection([(0., cross_sections.TabulatedCrossSection, ('peanut_IAS',)),
                                            (0.14, cross_sections.SophiaSuperposition, ())])

The numbers in the arguments to :class:`prince_cr.cross_sections.CompositeCrossSection` 
define the energies (in GeV) above which the models are joined. 
In this case Peanut will be used at low energies (greater then 0) and Sophia for energies above 0.14 GeV

These objects then need to be passed to :class:`prince_cr.core.PriNCeRun`::

    from prince_cr.core import PriNCeRun
    prince_run_talys = core.PriNCeRun(max_mass = 56, photon_field=pf, cross_sections=cs)

Can be used to set a maximal nucleus mass. All heavier nuclei will be ignored, which can lead to a very significant speedup.
Setting this option is equivalent to setting the same option in the config::

    from prince_cr.config import
    config.max_mass = 14

The config is a simple dictionary, see `prince/config.py` for all available options.
Most settings need to be set before creating other objects, such as `PriNCeRun`

Solving the Transport equation
..............................

To solve the transport equation, create an instance of :class:`UHECRPropagationSolverBDF`::

    from prince_cr.solvers import UHECRPropagationSolverBDF
    solver = UHECRPropagationSolverBDF(initial_z=1., final_z = 0.,prince_run=prince_run,
                                    enable_pairprod_losses = True, enable_adiabatic_losses = True)

This will use Backward Differentiation to solve the transport equation from redshift 1 to 0.
The Switches can be used to enable or disable interactions. There are no UHECRs injected into the system yet.
Add a source class by::

    rmax = 1e11
    gamma = 1.2

    from prince_cr.cr_sources import AugerFitSource,
    solver.add_source_class(
        AugerFitSource(prince_run, norm = 1e-50, params={1407: (gamma, rmax, 1.)}))

This will inject pure nitrogen with maximal rigidity 1e11 GV and a spectral index of E^-1.2.
You can add multiple sources by repeated calls to `add_source_class`. However note that this will get slow for a large number of sources.

Finally solve the system by calling::

    solver.solve(dz=1e-3,verbose=False,progressbar=True)

The result is available as `solver.res`, which is an instance of :class:`UHECRPropagationResults`
and contains several convienience functions for accessing the propagated spectra (see full documentation or examples).

Custom UHECR sources
....................

In the above examples we injected a pure nitrogen source. To inject a mix of elements,
provide more keys to the `params` argument::

    rmax = 1e11
    gamma = 1.2

    f_hydrogen = 0.
    f_helium = 67.3
    f_nitrogen = 28.1
    f_silicon = 4.6
    f_iron = 0.

    from prince_cr.cr_sources import AugerFitSource,
    solver.add_source_class(prince_run, norm = total_norm,
                   params={101: (gamma, rmax, f_hydrogen),
                           402: (gamma, rmax, f_helium),
                           1407: (gamma, rmax, f_nitrogen),
                           2814: (gamma, rmax, f_silicon),
                           5626: (gamma, rmax, f_iron)}))

`gamma` and `rmax` can also be defined separately for each element.

The spectral shape is defined by the source class (in this case :class:`prince_cr.cr_sources.AugerFitSource`)
:mod:`prince_cr.cr_sources` contains several other predefined classes. 
You can also define your own source class by subclassing :class:`prince_cr.cr_sources.CosmicRaySource`
and implementing :func:`CosmicRaySource.injection_spectrum`.