.. _citations:

References
----------

The PriNCe code
...............

Whenever publishing results, we would like ask for a citation of the original paper that contains the description of the code in the appendix.

    | *A new view on Auger data and cosmogenic neutrinos in light of different nuclear disintegration and air-shower models*  
    | J. Heinze, A. Fedynitch, D. Boncioli and W. Winter  
    | Astrophys.J. 873 (2019) no.1, 88  
    | https://doi.org/10.3847/1538-4357/ab05ce

In case the code mentioned on your slides or internal notes,
please advertise the GitHub page https://github.com/joheinze/PriNCe.


Physical models
...............

The models for nuclear interactions and decays are parameterizations
of the models that we did not make. The default settings will use
a combination of the TALYS code for nuclear interactions and the SOPHIA event generator to run.

* Nuclear photo-disintegration models:
    -   | *TALYS: Comprehensive Nuclear Reaction Modeling*
        | A.J. Koning, S. Hilaire, M.C. Duijvestijn
        | AIP Conf.Proc. 769 (2005) 1, 1154
        | International Conference on Nuclear Data for Science and Technology (ND2004), 1154 
        | DOI: 10.1063/1.1945212

    -   | *PEANUT* is part of FLUKA:
        | *FLUKA: A multi-particle transport code (Program version 2005)*
        | Alfredo Ferrari, Paola R. Sala, Alberto Fasso, Johannes Ranft
        | Report number: CERN-2005-010, SLAC-R-773, INFN-TC-05-11, CERN-2005-10
        | DOI: 10.2172/877507
        
    -   | *PSB: Photonuclear Interactions of Ultrahigh-Energy Cosmic Rays and their Astrophysical Consequences*
        | Puget, J. L. and Stecker, F. W. and Bredekamp, J. H.
        | Astrophys. J. 205 (1976)

* Photo-hadronic interactions and photo-pion production:

    -   | *SOPHIA: Monte Carlo simulations of photohadronic processes in astrophysics*
        | A. Mucke, Ralph Engel, J.P. Rachen, R.J. Protheroe, Todor Stanev
        | Comput.Phys.Commun. 124 (2000) 290-314
        | `astro-ph/9903478 [astro-ph] <https://arxiv.org/abs/astro-ph/9903478>`_
        | DOI: 10.1016/S0010-4655(99)00446-4

    -   | *Improved photomeson model for interactions of cosmic ray nuclei*
        | Leonel Morejon, Anatoli Fedynitch, Denise Boncioli, Daniel Biehl, Walter Winter
        | JCAP 11 (2019) 007
        | `arXiv:1904.07999 <http://arxiv.org/abs/1904.07999>`_
        | DOI: 10.1088/1475-7516/2019/11/007 (publication)
        | `AstroPhoMes code <https://github.com/mohller/AstroPhoMes>`_


* Air-shower models (if used in comparisons with Xmax):
    | *The hadronic interaction model Sibyll 2.3c and extensive air showers*
    | Felix Riehn, Ralph Engel, Anatoli Fedynitch, Thomas K. Gaisser, Todor Stanev
    | `arXiv:1912.03300 <http://arxiv.org/abs/1912.03300>`_

    | *Monte Carlo treatment of hadronic interactions in enhanced Pomeron scheme: I. QGSJET-II model*
    | Sergey Ostapchenko
    | Phys.Rev. D83 (2011) 014018, `arXiv:1010.1869 <http://arxiv.org/abs/1010.1869>`_
    
    