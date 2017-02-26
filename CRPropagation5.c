/* ***************************************************************
   * NEUCOSMA -- NEutrinos from Cosmic Accelerators              *
   * (c) 2010-2015 NEUCOSMA team  		                 *
   *************************************************************** */

//================================================================================================
// CRPropagation.c										//
//												//
// Routines the calculate the propagation of ultra-high-energy cosmic rays (UHECRs) from source	//
// to Earth, taking account adiabatic energy losses due to the cosmological expansion, energy 	//
// losses due to interaction with the photon backgrounds (through pair production and photopion //
// processes), and CR injection from sources at different redshifts (e.g., GRB, AGN, etc.). CRs //
// are assumed to be composed only of protons. The propagation is performed by numerically	//
// solving the Boltzmann transport equation for protons (see the ncoCRProp and ncoCRPropStep 	//
// subroutines).										//
//												//
// Ref.:											//
//												//
//	M. Ahlers, L.A. Anchordoqui, and S. Sarkar, Phys. Rev. D 79, 083009 (2009) [0902.3993]	//
//												//
//	M. Ahlers, M.C. Gonzalez-Garcia, and F. Halzen, Astropart.Phys. 35, 87 (2011) 		//
//		[1103.3421]									//
//												//
// Created: 11/06/2012										//
// Last modified: 22/04/2015									//
//================================================================================================

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>    // WW ADDED
#include <getopt.h> 

#include "nco_boost.h"
#include "myio.h"
#include "nco_utils.h"
#include "nco_photo.h"
#include "nco_decays.h"
#include "nco_synchr.h"
#include "nco_steady.h"

#include <gsl/gsl_integration.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//// 											      ////
//// 	VARIABLES AND PARAMETERS FOR CR PROPAGATION AND COSMOGENIC NEUTRINOS		      ////
//// 											      ////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

// GLOBAL VARIABLES (later into nco_utils.c or so)

// Note that CMB is treated as built-in function, an additional target photon spectrum can be specified as "CIB"
// This two-component split makes the code much more efficient! For neutrino production, both target photon spectra added

const int NCO_CRPROP_TRACKSPECTRUM = 0; 	// If 1, print the CR spectrum (either on screen or to an external file) every few redshift steps; default: 1
const int NCO_CRPROP_WRITESPECTRUM = 0;		// [Only if NCO_CRPROP_TRACKSPECTRUM = 1] If 0, write the CR spectrum on screen; if 1, write to external file; default: 1
const int NCO_CRPROP_PAIRPROD_CMB_LOSSES = 1;	// If 1, include pair production losses on CMB in the propagation of protons
const int NCO_CRPROP_PAIRPROD_CIB_LOSSES = 1;	// If 1, include pair production losses on CIB and other photons in the propagation of protons; the effect is small at high E
const int NCO_CRPROP_PHOTOPION_CMB_LOSSES = 1;	// If 1, include photopion losses on CMB photons in the propagation of protons; has to be one if CMB neutrinos to be computed
const int NCO_CRPROP_PHOTOPION_CIB_LOSSES = 1;	// If 1, include photopion losses on CIB and other photons in the propagation of protons; the effect is visible, but small
const int NCO_CRPROP_COSMNEUTRINOS = 2;		// If 0, no cosmogenic neutrinos; 1: compute cosmogenic neutrino injection from CMB interactions as well; 2: CMB+CIB interactions; default: 2
const int NCO_CRPROP_CMB_SCALING = 1;		// If 1, compute interactions with CMB at z=0 only and scale to find values at other redshifts; if 0, compute interactions for any z 
						// If scaling not used, about factor 1/DELTAZ slower right now: O(50s) -> O(40mins) on wtpp177 (full photohadronics); 
						// typically use scaling, since CMB scaling exact, i.e., default 1; CIB is calculated extra!
const int NCO_CRPROP_CIB_MODEL = 1;		// If 1, use the CIB1 model (by Franceschini et al.); if 2: use the CIB2 model (by Stecker et al.)
const int NCO_CRPROP_EMCASC_E_DENSITY = 0;	// If 1, calculate the energy density of electromagnetic cascades

int NCO_PHOTO_IT;	                 	// Interaction types taken into account for neutrino production; default: -1 (all pion); set below in main program!

// THESE NUMBERS SHOULD NOT BE TOUCHED ANYMORE, UNLESS YOU REALLY KNOW WHAT YOU DO

static int NX = 80;//60;		// Number of points to calculate in the x-direction (x = log(E/GeV)); standard now: 60
					// ONE NEEDS TO INCREASE NX AND DECREASE DELTAZ SIMULTANEOUSLY! (e.g. 100 points here, DELTAZ=0.00001 for better precision)
static double DELTAXDERIV = 1.e-10;	// Step size used by ncoSymmDeriv to calculate the derivative w.r.t. x. Def.: 1e-10
static double DELTAZ = 0.00005;           // Step size in the redshift direction (used in CRProp); standard now: pair prod. only: 0.005; better: 0.001, photohadronics: 5.0^e-5
static double DELTAZDIV = 500;		// DELTAZ will be devided by this factor below Z=2*ZHOM -> here absolute DELTAZ ~ 0.0001 required
static double XMIN = 5.0;		// log10 of minimum proton energy [GeV]; need at least minimum neutrino energy * 20
static double XMAX = 14.0;//12.0;	// log10 of maximum proton energy [GeV]; about one order of magnitude above z=0 plot range XM0 (XM0*(1+z) ~ XMAX)
					// WW: energy range should be slighly larger in CRPropagation5 to reproduce same results (different boundary conditions!)
					// subtle point for neutrino injection: multi-pion processes access rather large Ep, and are therefore sensitive to threshold effects (upper threshold). 
					// Therefore larger E-range
static double ZHOM = 0.00;//0.001;      // Threshold where universe becomes inhomogeneous if non-trivial source distribution used (injection cutoff below that threshold); typically 0.02???
static double ZSWITCH = 0.00;//0.001;	// Threshold below which the algorithm is switched to Euler method: about 2*ZHOM
static int NEUTRINOSTEPS = 1000;	// Trick: neutrino fluxes only computed every NEUTRINOSTEPS redshift step, i.e., for a redshift spacing NEUTRINOSTEPS*DELTAZ; 1000 (Delta z = 0.05) is
					// sufficient; adjust if DELTAZ changed to approx. 0.05/DELTAZ
static int CIBSTEPS = 500;      	// Trick: interaction rate for CIB calculated only every CIBSTEPS redshift step, i.e., for a redshift spacing CIBSTEPS*DELTAZ; 500 (Delta z = 0.025) is 
					// sufficient; adjust if DELTAZ changed to approx. 0.025/DELTAZ
static int CASCADESTEPS = 500;		// Trick: contribution to e.m. cascade density calculated only every CASCADESTEPS redshift steps (not much improvement in execution time; set to 1 by default)
static double XNMIN = 3.0;		// Minimal neutrino energy in spectra (max. energy: XMAX)
static int NXN = 50;			// Number of energy (x) steps for neutrino flux computation; good number: at least 50

// MODULE-RELATED DEFINITIONS (FOR THE MODULE, GLOBAL)

static double H0 = 2.28475e-18;		// Hubble's constant [s^-1]; value of 2.28475e-18 s^-1 corresponds to 70.5 km s^-1 Mpc^-1.
static double OmegaM = 0.27; 		// cold matter density 0.27
static double OmegaL = 0.73; 		// cosmological constant density 0.73
static double LH = 3.89e3;		// Hubble length [Mpc]
static double T0 = 2.725;		// present-day CMB temperature [K]
static double kB = 8.617343e-14;	// Boltzmann constant [GeV^-1 K^-1]
static double ECMB = 2.34823e-13;	// = kB*T0 [GeV]
static double h = 6.58211915e-25;	// = Planck's constant [GeV s]
static double hh = 43.3243e-50;		// = h*h [GeV^2 s^2]
static double me = 0.510998918e-3;	// electron mass [GeV]
static double mp = 938.272029e-3;	// proton mass [GeV]
static double Y2F = 8.12e-65;		// Conversion factor comoving density Mpc^-3 GeV^-1 -> Flux cm^-2 GeV^-1 s^-1 sr^-1 (*c/(4 pi))
static double MpcToCm = 3.0856775807e24;
static double ErgToGeV = 624.15;

// FLAVOUR MIXING PARAMETERS

static double theta12 = 0.587; // Oscillation angles for the three neutrino flavors. From 1205.5254v2 
static double theta13 = 0.157; // ----- MB: these need to be updated to the values in 1205.5254v3
static double theta23 = 0.683;
static double deltacp = 0.89*M_PI;

// GLOBAL VARIABLES (FOR THIS MODULE ONLY)

static nco_ip bpair_ip;
static nco_ip bpion_ip;
static nco_ip cib_ip;
static nco_ip cib_scale_ip;
static nco_ip logEc_ip;
static int CIBCOUNTER;
static int CASCADECOUNTER;
static double arrPhotoPionCIBLoss[1000];     // WW: Bug: these global arrays are not used anymore. CIB only effectively added every CIBSteps step (bug introduced after CRPropagation3.c)
static double arrPairProductionCIBLoss[1000];

// filenamebase, filenamebody and filenameext are used to dynamically generate output filenames in ncoCRPropStep
static char* filenamebase1 = "spectrum.adiabatic.only.z.";
static char* filenamebase2 = "spectrum.adiabatic.+.pair.prod.z.";
static char* filenamebase3 = "spectrum.adiabatic.+.pair.prod.+.photopion.z.";
static char* filenameext = ".dat";
char filenamebody[2];


//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//// 											      ////
//// 	INPUT-RELATED VARIABLES (INPUT IS NOT PART OF CR PROPAGATION CODE)		      ////
//// 											      ////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

static double INJEPMAX = 1e12;//3.16228e11;		// Maximal proton energy [GeV] at source; 3.16228*10^11 GeV = 10^20.5 eV
const int NCO_CRPROP_SOURCEDIST_CR = 3;		// CR source evolution: if 0, no source evolution; if 1, the sources follow the star formation rate; if 2, the GRB rate; if 3, SFR*(1+z)^NCO_CRPROP_COSMEVOL_CR
static double NCO_CRPROP_COSMEVOL_CR = 1.8;	// CR source evolution: exponent of the correction ~(1+z)^NCO_CRPROP_COSMEVOL_CR of the GRB comoving rate
static double NCO_CRPROP_ALPHA = 2.5;		// CR injection index
const int NCO_CRPROP_SOURCEDIST_CIB = 1;	// CIB source evolution: if 0, no source evolution; if 1, the sources follow the star formation rate; if 2, GRB rate; if 3, SFR*(1+z)^NCO_CRPROP_COSMEVOL_CIB
					  	// if 4, use the tabulated CIB spectrum density from Franceschini et al.; if 5, use the CIB spectrum calculated with the two-Gaussian model of the CIB 
					  	// injection; if 6, use the CIB by Inoue et al. (baseline model); if 7, use Inoue with lower Pop-III limit; if 8, use Inoue with upper Pop-III limit
static double NCO_CRPROP_COSMEVOL_CIB = 1.8; 	// CIB source evolution: exponent of the correction ~(1+z)^NCO_CRPROP_COSMEVOL_CIB of the GRB comoving rate

// Type definition for CR injection spectrum as a function of (log of) energy [GeV] in the comoving frame and redshift
// Input: log(E/GeV), z
// Units: GeV^-1 Mpc^-3 s^-1
typedef double (*nco_crinjspectrum)(double,double);  

// Type definition for target photon spectrum as a function of (log of) energy [GeV] in the comoving frame and redshift; 
// Input: log(E/GeV), z
// Units: GeV^-1 cm^-3
typedef double (*nco_targetphotons)(double,double);  

// Arrays used by CRInjectionSpectrumTwoComponentInterpolation to ouptut the interpolated CR
// injection spectrum calculated within the two-component model
static int NNZZ = 61;//239;
static int NNXX = 151;
double arrzcrspectrumtwocomp[61];//[239];
double arrxcrspectrumtwocomp[151];
double arrlogcrspectrumtwocomp[61][151];//[239][151];
static nco_ip logCRSpectrumTwoCompAtFixedZ_ip[61];//[239];

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//// 											      ////
//// 	CRITICAL ENERGY Ec ABOVE WHICH NEUTRONS INTERACT BEFORE DECAYING		      ////
//// 											      ////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

const static int nEc=61;
static double arrneutronZ[61]={0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 
1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 
2.8, 2.9, 3., 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4., 4.1, 
4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5., 5.1, 5.2, 5.3, 5.4, 5.5,
5.6, 5.7, 5.8, 5.9, 6.};
static double arrneutronlogEc[61]={11.6656, 11.5604, 11.4697, 11.3892, 11.3173, 11.2528, 11.1944, 
11.1408, 11.091, 11.0444, 11.0024, 10.9621, 10.9243, 10.8894, 
10.8553, 10.8242, 10.7935, 10.7652, 10.7378, 10.7111, 10.6867, 
10.662, 10.6391, 10.6171, 10.5952, 10.5746, 10.5549, 10.5354, 
10.5162, 10.4985, 10.4812, 10.4635, 10.4467, 10.4313, 10.4157, 
10.3998, 10.3847, 10.3709, 10.3568, 10.3427, 10.3288, 10.3158, 
10.3033, 10.2907, 10.2783, 10.2659, 10.2538, 10.2426, 10.2316, 
10.2205, 10.2094, 10.1981, 10.1874, 10.1777, 10.1678, 10.1579, 
10.1479, 10.1378, 10.128, 10.1191, 10.1103};


//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//// 											      ////
//// 	LOCAL (z = 0) CIB PARAMETRISATION DATA FOR INTERPOLATION			      ////
//// 											      ////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

// Raw data for CIB1 model at z=0
// Ref.: A. Franceschini et al., Astron. Astrphys. 487, 837 (2008) [arXiv:0805.1841]
// Units: GeV^-1 cm^-3
const static int ncib1=33;
static double cibrawx1[33]={-15., -11.835, -11.604, -11.45, -11.303, -11.136, -11.052, -10.947, 
-10.86, -10.78, -10.751, -10.684, -10.508, -10.383, -10.303, -10.206, 
-10.082, -9.9846, -9.8598, -9.8085, -9.7314, -9.6688, -9.5597, 
-9.4713, -9.3792, -9.1359, -9.01936, -8.90553, -8.8085, -8.6459, 
-8.5075, -8.2065, 1.};
static double cibrawy1[33]={6, 10.8983, 11.2507, 11.3374, 11.2726, 
  11.0669, 10.8765, 10.5735, 10.2867, 10.0076, 9.8984, 9.7006, 9.09, 
  8.749, 8.51, 8.323, 8.037, 7.7156, 7.4308, 7.3555, 7.3084, 7.2318, 
  7.1517, 7.1093, 7.0432, 6.7459, 6.52736, 6.26753, 5.9855, 5.5049, 
  4.9675, 4.0405, -29};

// Raw data for CIB2 model at z=0
// Ref.: F.W. Stecker et al., Astrophys. J. 648, 774 (2006) [astro-ph/0510449]
// Units: GeV^-1 cm^-3
const static int ncib2=86;
static double cibrawx2[86]={-11.4806, -11.4355, -11.3967, -11.3645, -11.3257, -11.287, -11.2418,
-11.2031, -11.1579, -11.1256, -11.0934, -11.0546, -11.0224, -11.003,
-10.9772, -10.9449, -10.8997, -10.8417, -10.7965, -10.7513, -10.6803,
-10.6287, -10.5512, -10.5125, -10.4608, -10.4157, -10.3447, -10.2801,
-10.1833, -10.0865, -10.0284, -9.97031, -9.90577, -9.84768, -9.77668,
-9.71859, -9.68632, -9.64759, -9.60886, -9.55077, -9.49269, -9.46041,
-9.42169, -9.38941, -9.35069, -9.31842, -9.28614, -9.26033, -9.23451,
-9.20869, -9.16997, -9.13769, -9.09897, -9.0796, -9.04088, -9.00215,
-8.95697, -8.93115, -8.89888, -8.87306, -8.84725, -8.81497, -8.7827,
-8.73107, -8.67298, -8.64071, -8.60198, -8.56971, -8.54389, -8.51162,
-8.47935, -8.44707, -8.42126, -8.3309, -8.29217, -8.25344, -8.20826,
-8.15017, -8.07917, -7.98236, -7.93718, -7.91136, -7.892, -7.87909,
-7.85327, -7.84036};
static double cibrawy2[86]={11.2694, 11.2458, 11.2286, 11.1964, 11.1576, 11.1081, 11.0414,
10.9703, 10.8821, 10.8067, 10.7313, 10.6387, 10.5525, 10.4901,
10.4104, 10.3135, 10.1713, 9.99467, 9.84173, 9.69957, 9.46693,
9.32909, 9.11155, 9.02972, 8.91343, 8.82515, 8.70027, 8.59262,
8.42037, 8.24813, 8.15771, 8.05652, 7.94887, 7.85845, 7.75513,
7.67549, 7.63244, 7.58294, 7.53343, 7.45379, 7.37415, 7.3311,
7.27082, 7.22778, 7.1675, 7.11368, 7.05985, 7.04481, 7.01899,
6.99317, 6.94367, 6.90062, 6.85112, 6.82098, 6.76071, 6.70043,
6.62292, 6.57555, 6.52173, 6.47436, 6.42699, 6.36239, 6.28701,
6.17072, 6.04798, 5.9726, 5.88, 5.79385, 5.7357, 5.63877, 5.55262,
5.47725, 5.38677, 5.124, 4.99907, 4.87413, 4.73197, 4.54457, 4.2796,
3.91339, 3.72813, 3.62688, 3.56441, 3.5084, 3.40715, 3.35114};


//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//// 											      ////
//// 	CIB PHOTON DENSITY PARAMETRISATION BY FRANCESCHINI ET AL. UP TO Z = 2		      ////
//// 											      ////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

// Raw data for CIB model by Franceschini et al., extracted from Tables 1 and 2 of arXiv:0805.1841
// These data is used for interpolation
// Units: [energy (x)] = log10(GeV)
//        [dn/denergy (y)] = log10(GeV^-1 cm^-3)

static int ncibfrancz = 11;
static double cibfrancz[11] = {0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0};

static nco_ip cibfrancz00_ip;
static nco_ip cibfrancz02_ip;
static nco_ip cibfrancz04_ip;
static nco_ip cibfrancz06_ip;
static nco_ip cibfrancz08_ip;
static nco_ip cibfrancz10_ip;
static nco_ip cibfrancz12_ip;
static nco_ip cibfrancz14_ip;
static nco_ip cibfrancz16_ip;
static nco_ip cibfrancz18_ip;
static nco_ip cibfrancz20_ip;

static int ncibfranc = 31;
// z = 0.0
static double cibfrancz00x[31] = {-11.835, -11.604, -11.45, -11.303, -11.136, -11.052, -10.947, 
-10.86, -10.78, -10.751, -10.684, -10.508, -10.383, -10.303, -10.206, 
-10.082, -9.9846, -9.8598, -9.8085, -9.7314, -9.6688, -9.5597, 
-9.4713, -9.3792, -9.1359, -9.01936, -8.90553, -8.8085, -8.6459,
-8.5075, -8.2065};
static double cibfrancz00y[31] = {10.8983, 11.2507, 11.3374, 11.2727, 11.0669, 10.8765, 10.5735, 
10.2867, 10.0076, 9.8984, 9.7006, 9.09, 8.749, 8.51, 8.323, 8.037, 
7.7156, 7.4308, 7.3555, 7.3084, 7.2318, 7.1517, 7.1093, 7.0432, 
6.7459, 6.52736, 6.26753, 5.9855, 5.5049, 4.9675, 4.0405};
// z= 0.2
static double cibfrancz02x[31] = {-11.756, -11.525, -11.37, -11.224, -11.057, -10.972, -10.868, 
-10.78, -10.701, -10.671, -10.604, -10.428, -10.303, -10.224, 
-10.127, -10.002, -9.9055, -9.7804, -9.7293, -9.6523, -9.5897,
-9.4795, -9.3921, -9.3, -9.0567, -8.94805, -8.8262, -8.7293, -8.5666, 
-8.4283, -8.1273};
static double cibfrancz02y[31] = {11.0137, 11.3608, 11.4485, 11.3963, 11.1873, 10.992, 10.6791, 
10.3763, 10.0903, 9.9776, 9.7748, 9.163, 8.834, 8.631, 8.439, 8.144, 
7.8425, 7.5544, 7.4783, 7.4283, 7.3557, 7.2755, 7.2231, 7.157, 
6.8067, 6.59305, 6.2912, 6.0113, 5.4616, 4.9883, 4.0893};
// z = 0.4
static double cibfrancz04x[31] = {-11.689, -11.458, -11.303, -11.157, -10.99, -10.906, -10.801, 
-10.714, -10.634, -10.604, -10.537, -10.361, -10.236, -10.157, 
-10.06, -9.9355, -9.8386, -9.7135, -9.6623, -9.5854, -9.5227, 
-9.4136, -9.3251, -9.2331, -8.99, -8.8732, -8.7595, -8.6623, -8.4996, 
-8.3614, -8.0604};
static double cibfrancz04y[31] = {11.1579, 11.5029, 11.5872, 11.5271, 11.3097, 11.1016, 10.7693, 
10.4569, 10.1623, 10.0499, 9.8473, 9.246, 8.93, 8.727, 8.543, 8.2455, 
7.9236, 7.6365, 7.5593, 7.5104, 7.4397, 7.3576, 7.3031, 7.2211, 
6.728, 6.5782, 6.2915, 5.9903, 5.4066, 5.0094, 4.0894};
// z = 0.6
static double cibfrancz06x[31] = {-11.631, -11.4, -11.245, -11.099, -10.932, -10.847, -10.743, 
-10.656, -10.576, -10.546, -10.48, -10.303, -10.178, -10.099, 
-10.002, -9.8775, -9.7804, -9.6556, -9.6045, -9.5274, -9.4647,
-9.3556, -9.2672, -9.1751, -8.93181, -8.8153, -8.7014, -8.6045,
-8.4417, -8.3034, -8.0024};
static double cibfrancz06y[31] = {11.2638, 11.6084, 11.6862, 11.6137, 11.381, 11.1534, 10.8029, 
10.4822, 10.1875, 10.0757, 9.8752, 9.292, 8.98, 8.782, 8.592, 8.3105, 
7.9464, 7.6646, 7.5915, 7.5474, 7.4807, 7.4026, 7.3432, 7.2381, 
6.81081, 6.5533, 6.2604, 5.9295, 5.3917, 5.0674, 4.0634};
// z = 0.8
static double cibfrancz08x[31] = {-11.58, -11.349, -11.194, -11.048, -10.881, -10.796, -10.692, 
-10.604, -10.525, -10.495, -10.428, -10.252, -10.127, -10.048, 
-9.9512, -9.8262, -9.7293, -9.6045, -9.5533, -9.4763, -9.4136, 
-9.3044, -9.216, -9.124, -8.8807, -8.7642, -8.6501, -8.5533, -8.3905, 
-8.2523, -7.951};
static double cibfrancz08y[31] = {11.3555, 11.6896, 11.7486, 11.6526, 11.3921, 11.1388, 10.7737, 
10.4472, 10.1553, 10.0447, 9.845, 9.2874, 8.981, 8.79, 8.6032, 
8.2852, 7.8573, 7.6005, 7.5433, 7.5283, 7.4706, 7.4114, 7.336, 7.21, 
6.7657, 6.5012, 6.1991, 5.8293, 5.3795, 5.1043, 4.03};
// z = 1.0
static double cibfrancz10x[31] = {-11.534, -11.303, -11.148, -11.002, -10.835, -10.751, -10.646,
-10.559, -10.48, -10.45, -10.383, -10.206, -10.082, -10.002, -9.9055,
-9.7804, -9.6836, -9.5586, -9.5075, -9.4305, -9.3678, -9.2587,
-9.1702, -9.07825, -8.8348, -8.7183, -8.6045, -8.5075, -8.3448,
-8.2065, -7.906};
static double cibfrancz10y[31] = {11.4278, 11.7481, 11.7837, 11.6656, 11.3741, 11.1018, 10.7226, 
10.3985, 10.1089, 9.9991, 9.8, 9.264, 8.966, 8.761, 8.6105, 8.2044, 
7.7486, 7.5206, 7.4815, 7.4915, 7.4388, 7.3927, 7.2982, 7.15125, 
6.6968, 6.4353, 6.1015, 5.7085, 5.3818, 5.1275, 4.027};
// z = 1.2
static double cibfrancz12x[31] = {-11.492, -11.262, -11.107, -10.961, -10.793, -10.709, -10.604, 
-10.517, -10.438, -10.408, -10.341, -10.165, -10.04, -9.961, -9.8642, 
-9.7392, -9.6423, -9.5173, -9.4661, -9.3891, -9.3265, -9.2173, 
-9.1289, -9.04015, -8.7934, -8.666, -8.567, -8.4661, -8.3034, 
-8.1651, -7.864};
static double cibfrancz12y[31] = {11.4825, 11.7844, 11.7944, 11.6578, 11.3333, 11.0492, 10.6601, 
10.34, 10.0519, 9.9434, 9.7458, 9.2246, 8.939, 8.726, 8.6112, 8.0662, 
7.6693, 7.4593, 7.4301, 7.4541, 7.4195, 7.3493, 7.2329, 7.08815, 
6.6334, 6.345, 6.017, 5.6391, 5.4874, 5.1421, 4.087};
// z = 1.4
static double cibfrancz14x[31] = {-11.455, -11.224, -11.069, -10.923, -10.756, -10.671, -10.567, 
-10.48, -10.4, -10.37, -10.303, -10.127, -10.002, -9.9234, -9.8262, 
-9.7014, -9.6045, -9.4795, -9.4283, -9.3513, -9.2887, -9.1795, 
-9.08155, -9.0024, -8.7557, -8.6281, -8.5292, -8.4283, -8.2656,
-8.1273, -7.826};
static double cibfrancz14y[31] = {11.5206, 11.7931, 11.772, 11.6161, 11.2573, 10.9504, 10.5635, 
10.2499, 9.9606, 9.8573, 9.6666, 9.1556, 8.87, 8.7174, 8.5552, 
7.8934, 7.5915, 7.3955, 7.3733, 7.4073, 7.3817, 7.2885, 7.14755, 
7.01739, 6.5807, 6.2791, 5.9082, 5.6213, 5.4256, 5.1423, 4.165};
// z = 1.6
static double cibfrancz16x[31] = {-11.42, -11.189, -11.035, -10.888, -10.721, -10.637, -10.532, 
-10.445, -10.366, -10.336, -10.269, -10.093, -9.9678, -9.8884, 
-9.7916, -9.6666, -9.5698, -9.4448, -9.3936, -9.3166, -9.2539, 
-9.1448, -9.05637, -8.96428, -8.728, -8.6123, -8.4905, -8.3936,
-8.2308, -8.0925, -7.792};
static double cibfrancz16y[31] = {11.5356, 11.7784, 11.734, 11.5562, 11.1706, 10.8438, 10.4642, 
10.1558, 9.8661, 9.7688, 9.5891, 9.088, 8.7788, 8.7154, 8.4526, 
7.7596, 7.5638, 7.3668, 7.3456, 7.2686, 7.3429, 7.2238, 7.08837, 
6.94928, 6.54, 6.2673, 5.8195, 5.5936, 5.4378, 5.1265, 4.219};
// z = 1.8
static double cibfrancz18x[31] = {-11.388, -11.157, -11.002, -10.856, -10.689, -10.604, -10.5,
-10.413, -10.333, -10.303, -10.236, -10.06, -9.9355, -9.8564,
-9.7595, -9.6343, -9.5375, -9.4125, -9.3614, -9.2844, -9.2217,
-9.1115, -9.02462, -8.93124, -8.6958, -8.5612, -8.4622, -8.3614,
-8.1987, -8.0604, -7.759};
static double cibfrancz18y[31] = {11.5397, 11.7547, 11.6894, 11.4908, 11.0773, 10.7388, 10.3714, 
10.0612, 9.7739, 9.6819, 9.5099, 9.033, 8.7115, 8.6784, 8.3135, 
7.6533, 7.5435, 7.3455, 7.3224, 7.3544, 7.2997, 7.1555, 7.02362, 
6.87824, 6.4798, 6.1272, 5.7792, 5.5684, 5.4387, 5.0864, 4.229};
// z = 2.0
static double cibfrancz20x[31] = {-11.358, -11.127, -10.972, -10.826, -10.659, -10.574, -10.47,
-10.383, -10.303, -10.273, -10.206, -10.03, -9.9055, -9.8262,
-9.7293, -9.6045, -9.5075, -9.3826, -9.3314, -9.2545, -9.1918,
-9.08265, -8.98464, -8.90217, -8.6658, -8.55, -8.4283, -8.3314,
-8.1687, -8.0304, -7.729};
static double cibfrancz20y[31] = {11.5283, 11.7172, 11.6349, 11.4144, 10.9674, 10.632, 10.2754, 
9.9551, 9.6801, 9.5896, 9.4224, 8.962, 8.6785, 8.5872, 8.1663, 
7.5835, 7.5085, 7.3046, 7.2784, 7.3065, 7.2408, 7.07765, 6.85464, 
6.79917, 6.4118, 6.084, 5.7183, 5.5404, 5.4247, 5.0124, 4.228};


//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//// 											      ////
//// 	CIB PHOTON DENSITY PARAMETRISATION BY INOUE ET AL. UP TO Z = 10			      ////
//// 											      ////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

// Ref.: Y. Inoue et al., arXiv:1212.1683
// Proper photon density data retrieved from http://www.slac.stanford.edu/~yinoue/Download.html
// Original data was epsilon*flux [nW m^-2 sr^-1] for z = 0 and  epsilon*dn/depsilon [GeV cm^-3] for z > 0
// It was converted to dn/depsilon [GeV^-1 cm^-3] and stored as tables in the following external files:
// - EBL_inoue_baseline.dat: baseline model
// - EBL_inoue_low_pop3.dat: model with lower Pop-III limit
// - EBL_inoue_up_pop3.dat: model with upper Pop-III limit
// The data in these files is read by the routine PrecomputeCIBInoue and used for interpolation by CIBPhotonSpectrumInoue.
// See the routine PrecomputeCIBInoue for details on the internal formatting of these files.
// Units: [energy (x)] = log10(GeV)
//        [dn/denergy (y)] = log10(GeV^-1 cm^-3)

static int CIBInoueNumberZValues = 110; // number of redshift values at which the CIB spectrum by Inoue et al. has been computed
static int CIBInoueNumberXValues = 100; // number of x=log10(E/GeV) values at which the CIB spectrum by Inoue et al. has been computed
// CIBInoueZValues: redshift values at which the CIB spectrum is known
static double CIBInoueZValues[110] = {0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3,
0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1, 
3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4., 4.1, 4.2, 4.3, 4.4, 4.5,
4.6, 4.7, 4.8, 4.9, 5., 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9,
6., 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7., 7.1, 7.2, 7.3, 
7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8., 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7,
8.8, 8.9, 9., 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.};
// CIBInoueXValues: x (=log(E/GeV)) values at which the CIB spectrum is known
static double CIBInoueXValues[100] = {-12., -11.9508, -11.8996, -11.8508, -11.8013, -11.7496, -11.699,
-11.6498, -11.6003, -11.5498, -11.5003, -11.4498, -11.4001, -11.3497,
-11.3002, -11.2503, -11.2, -11.15, -11.1002, -11.0501, -11.,
-10.9508, -10.8996, -10.8508, -10.8013, -10.7496, -10.699, -10.6498,
-10.6003, -10.5498, -10.5003, -10.4498, -10.4001, -10.3497, -10.3002,
-10.2503, -10.2, -10.15, -10.1002, -10.0501, -10., -9.95078,
-9.89963, -9.85078, -9.80134, -9.74958, -9.69897, -9.64975, -9.60033,
-9.54975, -9.50031, -9.44977, -9.40012, -9.34969, -9.30016, -9.25026,
-9.19997, -9.14997, -9.10018, -9.05012, -9., -8.95078, -8.89963,
-8.85078, -8.80134, -8.74958, -8.69897, -8.64975, -8.60033, -8.54975,
-8.50031, -8.44977, -8.40012, -8.34969, -8.30016, -8.25026, -8.19997,
-8.14997, -8.10018, -8.05012, -8., -7.95078, -7.89963, -7.85078,
-7.80134, -7.74958, -7.69897, -7.64975, -7.60033, -7.54975, -7.50031,
-7.44977, -7.40012, -7.34969, -7.30016, -7.25026, -7.19997, -7.14997,
-7.10018, -7.05012};
// The i-th element of the array CIBInoue_ip will hold the interpolating function of the Inoue CIB
// (log10[dn/denergy/(GeV^-1 cm^-3)] for the value of redshift specified by CIBInoueZValues[i].
static nco_ip CIBInoue_ip[110];
// Filenames where the CIB data for the Inoue et al. models are stored.
char* filenameCIBInoueBaseline = "EBL_inoue_baseline.dat";
char* filenameCIBInoueLower = "EBL_inoue_low_pop3.dat";
char* filenameCIBInoueUpper = "EBL_inoue_up_pop3.dat";

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//// 											      ////
//// 	EXTRA ROUTINES (UTILITIES, AUXILIARY, ETC.)					      ////
//// 											      ////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

//================================================================================================
// ncoIPAllocLinear										//
//												//
// Creates and returns a linear interpolating function object ipdata based on the points 	//
// ((x[0],y[0]),(x[1],y[1]), ..., (x[size-1],y[size-1])).					//
//												//
// Input:											//
//												//
//	ipdata: the interpolating object							//
//	x[0..size-1]: array of abscissae							//
//	y[0..size-1]: array of function values							//
//	size: number of data points								//
//												//
//												//
// Output:											//
//												//
//	Interpolating function ipdata				 				//
//	units: units of y									//
//												//
// Created: 19/07/2012										//
// Last modified: 13/08/2012									//
//================================================================================================
/* now in library
void ncoIPAllocLinear(nco_ip* ipdata, double x[],double y[],int size)
{
	ipdata->acc = gsl_interp_accel_alloc();
	ipdata->spline = gsl_spline_alloc (gsl_interp_linear, size);
	int res = gsl_spline_init(ipdata->spline, x, y, size);
	if (res!=0) printf("Spline init error %i occured in IPAlloc\n",res);
	ipdata->min=x[0];
	ipdata->max=x[size-1];
}
*/

//================================================================================================
// ncoSymmDeriv											//
//												//
// Returns the value of the derivative of y = 10^func evaluated at x. The numerical derivative 	//
// is calculated in a symmetrical way.								//
//												//
// Input:											//
//												//
//	func: log10 of the function y to be derived						//
//	x: the only argument of the function; if y is a function of E, then x = log10(E/GeV)	//
//												//
// Output:											//
//												//
//	Derivative dy/dx, evaluated at the specified value of x					//
//	units: units of dy/dx									//
//												//
// Created: 18/06/2012										//
// Last modified: 20/08/2012									//
//================================================================================================

double ncoSymmDeriv(nco_ip* func, double x)
{ 
	double fh, fl, xh, xl;
	double deriv, dx;

	if ((x<func->min) || (x>func->max))
	{
		printf("ncoSymmDeriv range error!\n");
		exit(-1);
	}

        // WW modified; reason: use simple (accurate) case in most evaluations now
	dx=0.0;
	if (x+DELTAXDERIV <= func->max)
	{
		dx+=DELTAXDERIV;
		xh=x+DELTAXDERIV;
	}
        else
	{
		dx+=(func->max - x);
		xh=func->max;
	}

	if (x-DELTAXDERIV >= func->min)
	{
		dx+=DELTAXDERIV; 
		xl=x-DELTAXDERIV;
	}
	else
	{
		dx+=(x - func->min);
		xl=func->min;
	}
 
	fh = pow(10.0,ncoIP(func,xh));
	fl = pow(10.0,ncoIP(func,xl));
	
        // This is the derivative dF/dx	
	deriv = (fh-fl)/dx/log(10.0)/pow(10.0,(xh+xl)/2.0); // WW modified; actual definition of derivative of log. function from dF/dx (more precise)

	return deriv;
}


//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//// 											      ////
//// 	ROUTINES RELATING TO COSMOLOGY							      ////
//// 											      ////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

//================================================================================================
// HubbleParameter										//
//												//
// Returns the value of the Hubble parameter at redshift z.					//
//												//
// Input:											//
//												//
//	z: redshift										//
//												//
// Output:											//
//												//
//	H(z) for the cosmology defined by (globals) OmegaM, OmegaL, H0 				//
//	units: s^-1										//
//												//
// Created: 13/05/2012										//
// Last modified: 23/05/2012									//
//================================================================================================

double HubbleParameter(double z)
{
	return H0*sqrt(OmegaM*pow(1.0+z,3.0)+OmegaL);
}

//================================================================================================
// CMBPhotonSpectrum										//
//												//
// Returns the redshift-scaled number density of CMB photons, in the CMB frame (equivalent to 	//
// the observer's frame). Normalisation from Planck's spectrum. The scaling goes as		//
// n(E,z) = (1+z)^3 n(E/(1+z),0). The CMB spectrum is a blackbody spectrum with the present-day	//
// temperature T0 = 2.725 K.									//
//												//
// Input:											//
//												//
//	E: photon energy [GeV]									//
//	z: redshift										//
//												//
// Output:											//
//												//
//	CMBPhotonSpectrum(E,z)									//
//	units: GeV^-1 cm^-3									//
//												//
// Ref.:											//
//												//
//	M. Ahlers, L.A. Anchordoqui, and S. Sarkar, Phys. Rev. D 79, 083009 (2009) [0902.3993]	//
//												//
// Created: 23/05/2012										//
// Last modified: 13/08/2012									//
//================================================================================================

double CMBPhotonSpectrum(double E, double z)
{
	static double pref = 1.31868e40; // = 1/pi^2/(hbar*c)^3 [GeV^-3 cm^-3]
	double zp1, Ered, nlocal,res;
	
	zp1 = 1.0+z;
	Ered = E/zp1;
	nlocal = pref*Ered*Ered / (exp(Ered/ECMB)-1.0); // density at z=0, for energy E/(1+z); ECMB = kB*T0
	
 	res = pow(zp1,2.0)*nlocal;
//	res = pow(zp1,3.0)*nlocal;
	return res;
}

// Some routines moved from below (used earlier)

//================================================================================================
// StarFormationRate										//
//												//
// Returns the star formation rate, per comoving volume, evaluated at the specified redshift.	//
//												//
// Input:											//
//												//
//	z: redshift										//
//												//
// Output:											//
//												//
//	StarFormationRate(z)									//
//	units: Mpc^-3										//
//												//
// Ref.:											//
//												//
//	A.M. Hopkins and J.F. Beacom, Astrophys. J. 651, 142 (2006) [astro-ph/0601463]		//
//												//
//	P. Baerwald, S. Huemmer, and W. Winter, Astropart. Phys. 35, 508 (2012) [1107.5583]	//
//												//
// Created: 23/07/2012										//
// Last modified: 13/08/2012									//
//================================================================================================

double StarFormationRate(double z)
{
	if (z<ZHOM) return 0.0;
	else if (z <= 0.97) return pow(1.0+z,3.44);
	else if (z > 0.97 && z <= 4.48) return pow(10.0,1.09)*pow(1.0+z,-0.26);
	else if (z > 4.48) return pow(10.0,6.66)*pow(1.0+z,-7.8);
}

//================================================================================================
// CRSourceDistribution										//
//												//
// Returns the density of cosmic-ray sources at the specified redshift. The model of redshift	//
// evolution is selected by setting the NCO_CRPROP_SOURCEDIST_CR option: 0 -- no evolution;	//
// 1 -- follows the star formation rate; 2 -- follows the GRB rate; 3 -- follows arbitrary	// 
// evolution.											//
//												//
// Input:											//
//												//
//	z: redshift										//
//												//
// Output:											//
//												//
//	CRSourceDistribution(z)									//
//	units: Mpc^-3										//
//												//
// Ref.:											//
//												//
//	A.M. Hopkins and J.F. Beacom, Astrophys. J. 651, 142 (2006) [astro-ph/0601463]		//
//												//
//	M.D. Kistler et al., Astrophys. J. 705, L104 (2009) [0906.0590]				//
//												//
//	P. Baerwald, S. Huemmer, and W. Winter, Astropart. Phys. 35, 508 (2012) [1107.5583]	//
//												//
// Created: 23/07/2012										//
// Last modified: 28/09/2012									//
//================================================================================================

double CRSourceDistribution(double z)
{ 
	switch (NCO_CRPROP_SOURCEDIST_CR)
	{
	  case 0: // No source evolution
		return 1.0;
		break;
	  case 1: // Source redshift evolution follows the star formation rate
		return StarFormationRate(z);
		break;
	  case 2: // Source redshift evolution follows the GRB rate
		return pow(1.0+z,1.2)*StarFormationRate(z);
		break;
	  case 3: // Source redshift evolution follows arbitrary evolution
		return pow(1.0+z,NCO_CRPROP_COSMEVOL_CR)*StarFormationRate(z);
		break;
	}
}

//================================================================================================
// CIBSourceDistribution									//
//												//
// Returns the density of IR/optical photon sources at the specified redshift. The model of 	//
// redshift evolution is selected by setting the NCO_CRPROP_SOURCEDIST_CR option: 0 -- no 	//
// evolution; 1 -- follows the star formation rate; 2 -- follows the GRB rate; 3 -- follows	//
// arbitrary evolution.										//
//												//
// Input:											//
//												//
//	z: redshift										//
//												//
// Output:											//
//												//
//	CIBSourceDistribution(z)								//
//	units: Mpc^-3										//
//												//
// Ref.:											//
//												//
//	A.M. Hopkins and J.F. Beacom, Astrophys. J. 651, 142 (2006) [astro-ph/0601463]		//
//												//
//	M.D. Kistler et al., Astrophys. J. 705, L104 (2009) [0906.0590]				//
//												//
//	P. Baerwald, S. Huemmer, and W. Winter, Astropart. Phys. 35, 508 (2012) [1107.5583]	//
//												//
// Created: 28/09/2012										//
// Last modified: 28/09/2012									//
//================================================================================================

double CIBSourceDistribution(double z)
{  
	switch (NCO_CRPROP_SOURCEDIST_CIB)
	{
	  case 0: // No source evolution
		return 1.0;
		break;
	  case 1: // Source redshift evolution follows the star formation rate
		return StarFormationRate(z);
		break;
	  case 2: // Source redshift evolution follows the GRB rate
		return pow(1.0+z,1.2)*StarFormationRate(z);
		break;
	  case 3: // Source redshift evolution follows arbitrary evolution
		return pow(1.0+z,NCO_CRPROP_COSMEVOL_CIB)*StarFormationRate(z);
		break;
	}
}

//================================================================================================
// PrecomputeCIBInoue										//
//												//
// Build the basic interpolating functions for the CIB spectrum by Inoue et al., i.e., 		//
// log10[dn/denergy/(GeV)] vs. x, for 110 different values of redshift between 0 and 10. For	//
// the i-th value of z, an interpolating function is built and stored in CIBInoue_ip[i] for use	//
// by the CIBPhotonSpectrumInoue routine. The global flag NCO_CRPROP_SOURCEDIST_CIB selects	//
// which model to use: NCO_CRPROP_SOURCEDIST_CIB = 6 (baseline), 7 (lower Pop-III limit), or	//
// 8 (upper Pop-III limit).									//
//												//
// Input:											//
//												//
//	None											//
//												//
// Output:											//
//												//
//	None											//
//	log10[dn/denergy/(GeV)] vs. x for different redshifts are stored in the array of 	//
//	interpolating functions CIBInoue_ip							//
//												//
// Ref.:											//
//												//
//	Y. Inoue et al. [arXiv:1212.1683]							//
//												//
// Created: 08/04/2013										//
// Last modified: 08/04/2013									//
//================================================================================================

void PrecomputeCIBInoue()
{
	int i, j;
	double temp,CIBSpectrumFixedZ[CIBInoueNumberXValues];
	FILE * fh;

	switch (NCO_CRPROP_SOURCEDIST_CIB)
	{
	  case 6: // Select the baseline model by Inoue et al.
		fh = fopen(filenameCIBInoueBaseline,"r");
		break;
	  case 7: // Select the model with the lower Pop-III limit
		fh = fopen(filenameCIBInoueLower,"r");
		break;
	  case 8: // Select the model with the upper Pop-III limit
		fh = fopen(filenameCIBInoueUpper,"r");
		break;
	}
	
	// Read in the external data: each file contains CIBInoueNumberZValues lines, one for each redshift
	// value in the array CIBInoueZValues. Each line contains CIBInoueNumberXValues different entries, one for
	// each value of x = log(E/GeV) in CIBInoueXValues.
	for (i=0; i<CIBInoueNumberZValues; i++)
	{
		// The array CIBSpectrumFixedZ contains the CIB energy spectrum for a fixed value of z
		for (j=0; j<CIBInoueNumberXValues; j++)
		{
			fscanf(fh,"%lf",&temp);
			CIBSpectrumFixedZ[j] = temp;
		}
		// Build an interpolating function for these data and store it as an element of the array of
		// interpolating functions CIBInoue_ip
		ncoIPAllocLinear(&CIBInoue_ip[i],CIBInoueXValues,CIBSpectrumFixedZ,CIBInoueNumberXValues);
	}

	fclose(fh);
}

//================================================================================================
// PrecomputeCIBCoeff										//
//												//
// Precompute N_CIB(z)/N_CIB(0) function from Ahlers et al.					//
//												//
// Input:											//
//												//
//	zmin, zmax: minimum and maximum redshifts at which N_CIB(z)/N_CIB(0) is computed	//
//	z: redshift										//
//												//
// Output:											//
//												//
//	None											//
//	N_CIB(z)/N_CIB(0) is stored in the interpolation function cib_scale_ip			//
//												//
// Ref.:											//
//												//
//	M. Ahlers, L.A. Anchordoqui, and S. Sarkar, Phys. Rev. D 79, 083009 (2009) [0902.3993]	//
//												//
// Created: 13/08/2012										//
// Last modified: 13/08/2012									//
//================================================================================================

void PrecomputeCIBCoeff(double zmin,double zmax)
{
	double integr(double z)
	{ 
		double val=CIBSourceDistribution(z)/((1.0+z)*HubbleParameter(z));
		
		return val;
	}
  
	double arrx[100];
	double arry[100];
	double thez;
	int N=0;
	double denom,num;

	denom=ncoIntegrate(&integr,0.0,10.0,21); // maximum redshift assumed to be z = 10

	for (thez=zmin;thez<zmax+0.0001;thez+=(zmax-zmin)/(100-1))
	{
		arrx[N]=thez;
		num=ncoIntegrate(&integr,thez,10.0,21); // maximum redshift assumed to be z = 10
		arry[N]=pow(1.0+thez,3.0)*num/denom;
		// printf("N: %i z: %g num: %g denom: %g v: %g\n",N,arrx[N],num,denom,arry[N]);
		N++;  
	}
  
	ncoIPAllocLinear(&cib_scale_ip,arrx,arry,N);
}



//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//// 											      ////
//// 			ROUTINES FOR ADIABATIC LOSSES					      ////
//// 											      ////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

//================================================================================================
// fAdiabaticLossPreDeriv									//
//												//
// Returns the value of the term for adiabatic energy losses in the Boltzmann equation, before	//
// derivating w.r.t. x, i.e. H(z)*E*Y(x,z).							//
//												//
// Input:											//
//												//
//	x: log10(E0/GeV), with E0 the observed energy at z = 0					//
//	z: redshift										//
//	y: comoving CR density evaluated at (x,z) [GeV^-1 Mpc^-3]				//
//												//
// Output:											//
//												//
//	Adiabatic energy-loss term before derivating w.r.t. x, H(z)*E*Y(x,z)			//
//	units: s^-1 Mpc^-3									//
//												//
// Created: 18/06/2012										//
// Last modified: 17/07/2012									//
//================================================================================================

double fAdiabaticLossPreDeriv(double x, double z, double y)
{
	return HubbleParameter(z)*pow(10.0,x)*y;
}


//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//// 											      ////
//// 			ROUTINES FOR PAIR PRODUCTION LOSSES				      ////
//// 											      ////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

//================================================================================================
// phi 												//
//												//
// Returns the value of the phi function used in the computation of the energy loss rate	//
// through e+ e- pair production on a photon background. An approximate expression is used,	//
// with relative error of the order of 10^-3, compared to the exact calculation.		//
//												//
// Input:											//
//												//
//	xi: photon energy in units of me*c2							//
//												//
// Output:											//
//												//
//	phi(xi)											//
//	units: adimensional									//
//												//
// Ref.:											//
//												//
//	M.J. Chodorowski, A.A. Zdziarski, and M. Sikora, Astrop. J. 400, 181 (1992)		//
//												//
// Created: 28/06/2012										//
// Last modified: 28/06/2012									//
//================================================================================================

double phi(double xi)
{
	static double c[4] = {0.8048, 0.1459, 1.137e-3, -3.879e-6};
	static double d[4] = {-86.07, 50.96, -14.45, 2.67};
	static double f[3] = {2.910, 78.35, 1837.0};

	int i;
	double xim2, lnxi, s1, s2, result;
	
	if (xi < 25)
	{
		xim2 = xi-2.0;
		s1 = 0.0;
		for (i=0; i<4; i++)
		{
			s1 += c[i]*pow(xim2,i+1.0);
		}
		result = 0.262*pow(xim2,4.0)/(1.0+s1); // pi/12 = 0.262
	}
	else
	{
		lnxi = log(xi);
		s1 = 0.0;
		s2 = 0.0;
		for (i=0; i<3; i++)
		{
			s1 += d[i]*pow(lnxi,i);
			s2 += f[i]/pow(xi,i+1.0);
		}
		s1 += d[3]*pow(lnxi,3.0);
		result = xi*s1/(1.0-s2);
	}
	
	return result;
}

//================================================================================================
// bpair 											//
//												//
// Returns the energy loss rate b = dE/dt due to e+e- pair creation on a given isotropic	//
// photon background, at the specified proton energy and redshift.				//
//												//
// Input:											//
//												//
//	E: proton energy in the comoving frame [GeV]						//
//	z: redshift										//
//	nphoton: target isotropic background photon spectrum, as a function of (log of) photon	//
//		 energy [GeV] and redshift [GeV^-1 cm^-3]					//
//												//
// Output:											//
//												//
//	Energy loss rate b = dE/dt								//
//	units: GeV s^-1										//
//												//
// Ref.:											//
//												//
//	G. Blumenthal, Phys. Rev. D 1, 1596 (1970).						//
//												//
//	M.J. Chodorowski, A.A. Zdziarski, and M. Sikora, Astrop. J. 400, 181 (1992)		//
//												//
// Created: 28/06/2012										//
// Last modified: 13/08/2012									//
//================================================================================================

double bpair(double E, double z, nco_targetphotons nphoton)
{

   if (((NCO_CRPROP_CMB_SCALING == 0) && (nphoton==CMBPhotonSpectrum)) || (nphoton!=CMBPhotonSpectrum) || (z==0.0))
   {
	static double LOG10XIMIN = 0.30103; // = log10(XIMIN) = log10(2.0)
	static double LOG10XIMAX = 4.0;     // = log10(XIMAX)
  
	double Ein = me*mp/2.0/E; // [GeV]
	double pref = -4.53617e-24; // = -alpha * r0^2 * (me*c^2)^2 * c [GeV^2 cm^3 s^-1]
	double xi, integral;
	
	// Since nphoton depends on both photon energy and redshift, we need to define an auxiliary
	// function that depends only on energy, which is the integration variable
	double integrand(double log10xi)
	{
		// xi is the photon energy in units of me*c^2
		// We will integrate in log10(xi) instead of directly in xi
		xi = pow(10.0,log10xi);
		return nphoton(xi*Ein,z)*phi(xi)/xi;
	}
	
	integral = ncoIntegrate(&integrand,LOG10XIMIN,LOG10XIMAX,10);

	return pref*log(10.0)*integral;
    }
    else if ((NCO_CRPROP_CMB_SCALING == 1) && (nphoton==CMBPhotonSpectrum)) 
    {
	double mye = log10((1.0+z)*E);
	double rr;

	if((mye>=bpair_ip.min) && (mye<=bpair_ip.max))  // always ask for range, otherwise problems if max E cutoff
		rr=-pow(10.0,ncoIP(&bpair_ip,mye));
	else
		rr=0.0;
	return pow(1.0+z,2.0)*rr;
    }
}

//================================================================================================
// fPairProductionLossPreDeriv									//
//												//
// Returns the value of the term for pair production losses in the Boltzmann equation, before	//
// derivating w.r.t. x, i.e., bpair(x,z)*Y(x,z).						//
//												//
// Input:											//
//												//
//	x: log10(E/GeV), with E the energy in the comoving frame				//
//	z: redshift										//
//	y: comoving CR density evaluated at (x,z) [GeV^-1 Mpc^-3]				//
//	nphoton: target isotropic background photon spectrum, as a function of (log of) photon 	//
//		 energy [GeV] and redshift [GeV^-1 cm^-3]					//
//												//
// Output:											//
//												//
//	e^+ e^- pair production energy-loss term before derivating w.r.t. x, bpair(x,z)*Y(x,z)	//
//	units: s^-1 Mpc^-3									//
//												//
// Created: 02/07/2012										//
// Last modified: 02/07/2012									//
//================================================================================================

double fPairProductionLossPreDeriv(double x, double z, double y, nco_targetphotons nphoton)
{
	return bpair(pow(10.0,x),z,nphoton)*y;  
}


//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//// 											      ////
//// 			ROUTINES FOR PHOTOHADRONIC LOSSES				      ////
//// 											      ////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

//================================================================================================
// bpion 											//
//												//
// Returns the energy loss rate b = dE/dt due to photopion production on a given isotropic	//
// photon background, at the specified proton energy and redshift.				//
//												//
// Input:											//
//												//
//	E: proton energy in the comoving frame [GeV]						//
//	z: redshift										//
//	nphoton: target isotropic background photon spectrum, as a function of (log of) photon  //
//		 energy [GeV] and redshift [GeV^-1 cm^-3]					//
//												//
// Output:											//
//												//
//	Energy loss rate b = dE/dt								//
//	units: [GeV s^-1]									//
//												//
// Ref.:											//
//												//
//	Hümmer et al, Astrophys.J. 721 (2010) 630						//
//												//
// Created: 08/08/2012										//
// Last modified: 13/08/2012									//
//================================================================================================

double bpion(double E, double z, nco_targetphotons nphoton)
{
	if (((NCO_CRPROP_CMB_SCALING == 0) && (nphoton==CMBPhotonSpectrum)) || (nphoton!=CMBPhotonSpectrum) || (z==0.0))
	{
		double cool,esc;

		double target(double energy)
		{
			return nphoton(energy,z);
		}
   
		// ncoComputeCoolEscRate returns the photohadronic interaction rate in s^-1
// 		ncoComputeCoolEscRate(NCO_ALL_PION, NCO_PROTON_NEUTRON, target, E, &cool, &esc);  // here use all pion processes (full x-sec)
		ncoComputeCoolEscRate(NCO_ALL_PION, NCO_PROTON_NEUTRON, target, E, &cool, &esc);  // here use all pion processes (full x-sec)
// 		return -ncoComputeInteractionRate(NCO_PHOTO_IT, target, E)*E; // use this to produce the data to find the critical energy Ec
    
		// printf("z: %g E: %g, cool: %g\n",z,E,cool);

		return -cool*E;  // re-scale in terms of b (different definition here)
	}
	else if ((NCO_CRPROP_CMB_SCALING == 1) && (nphoton==CMBPhotonSpectrum))
	{
		double mye = log10((1.0+z)*E);
		double rr;

		if((mye>=bpair_ip.min) && (mye<=bpair_ip.max))
			rr=-pow(10.0,ncoIP(&bpion_ip,mye));
		else
			rr=0.0;
		return pow(1.0+z,2.0)*rr;
	}
}


//================================================================================================
// fPhotoPionLossPreDeriv									//
//												//
// Returns the value of the term for photopion losses in the Boltzmann equation, before		//
// derivating w.r.t. x, i.e., bpion(x,z)*Y(x,z).						//
//												//
// Input:											//
//												//
//	x: log10(E0/GeV), with E the energy in the comoving frame				//
//	z: redshift										//
//	y: comoving CR density evaluated at (x,z) [GeV^-1 Mpc^-3]				//
//	nphoton: target isotropic background photon spectrum, as a function of photon energy	//
//		 (in GeV) and redshift [GeV^-1 cm^-3]						//
//												//
// Output:											//
//												//
//	photo-pion energy-loss term before derivating w.r.t. x, bpion(x,z)*Y(x,z)	        //
//	units: s^-1 Mpc^-3									//
//												//
// Created: 07/08/2012										//
// Last modified: 07/08/2012									//
//================================================================================================

double fPhotoPionLossPreDeriv(double x, double z, double y, nco_targetphotons nphoton)
{
	double bp = bpion(pow(10.0,x),z,nphoton);
	
	return bp*y; 
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//// 											      ////
//// 	NEUTRINO INJECTION								      ////
//// 											      ////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

//================================================================================================
// ncoComputeCosmoNeutrinoInjection								//
//												//
// Computes the neutrino flux contribution at a redshift z over a stepize deltaz		//
//												//
// Input:											//
//												//
//	z:  redshift										//
//	zmin: min. redshift (comoving density automatically scaled to this zmin)		//
//	deltaz: redshift stepsize								//
//	y: comoving number density of CR at z [GeV^-1 Mpc^-3]					//
//	nphoton: the target photon spectrum is function of type nco_targetphotons		//
//		(of x and z); see defintion [GeV^-1 cm^-3]					//
//												//
// Output:											//
//												//
//	Neutrino fluxes including flavor mixing at zmin						//
//	units: GeV^-1 Mpc^-3	(double log interpolated)	 				//
//      Note that output function may have shifted energy range!				//
//												//
// Ref.:											//
//												//
//	Hümmer et al, Astrophys.J. 721 (2010) 630						//
//												//
// Created: 08/08/2012										//
// Last modified: 08/08/2012									//
//================================================================================================

// Interpolating functions for the injection spectra (from the secondary spectrum)
static nco_ip s_piplus_in;
static nco_ip s_piminus_in;
static nco_ip s_kplus_in;
static nco_ip s_neutron_in;

static 	nco_ip s_muplusr_in;
static 	nco_ip s_muplusl_in;
static 	nco_ip s_muminusr_in;
static 	nco_ip s_muminusl_in;

static double thez;
static nco_ip* they;
static nco_targetphotons thephoton;

double grb_photons(double energy)
{
	double phd = thephoton(energy,thez); // CMB spectrum at right energy
	return phd;
}

double grb_protons(double energy)
{
	double pd;
	double loge = log10(energy);
		
	// subtle point: nco_photo.c accesses also very high proton energies (E/chi) for multi-pion production, which may not exist in the interpolating function
	if ((loge>=they->min) && (loge<=they->max)) 
		pd = pow(10.0,ncoIP(they,loge))*pow(1.0+thez,3.0);
		// conversion co-moving density GeV^-1 Mpc^-3 in real photon density GeV^-1 cm^⁻3: 1/pow(30.857e23,3.0);
		// however, to avoid too small numbers, I did not put this factor, and I did not put it at the end (where I convert back again)	    
	else
		pd=0.0;

	return pd;
}

double w_piplus_steady(double energy)
{
	if (log10(energy) < s_piplus_in.min || log10(energy) > s_piplus_in.max) {return 0;}
	else {return pow(10,ncoIP(&s_piplus_in,log10(energy)))/ncoComputeDecayEscapeRate(NCO_PI_PLUS,energy);}
}

double w_piminus_steady(double energy)
{
	if (log10(energy) < s_piminus_in.min || log10(energy) > s_piminus_in.max) {return 0;}
	else {return pow(10,ncoIP(&s_piminus_in,log10(energy)))/ncoComputeDecayEscapeRate(NCO_PI_MINUS,energy);}
}

double w_kplus_steady(double energy)
{
	if (log10(energy) < s_kplus_in.min || log10(energy) > s_kplus_in.max) {return 0;}
	else {return pow(10,ncoIP(&s_kplus_in,log10(energy)))/ncoComputeDecayEscapeRate(NCO_K_PLUS,energy);}
}

double w_neutron_steady(double energy)
{
	if (log10(energy) < s_neutron_in.min || log10(energy) > s_neutron_in.max) {return 0;}
	else 
	{
// 			double af = exp(-pow(energy/4.0e11,2.0));
		double af = exp(-pow(energy/pow(10.0,ncoIP(&logEc_ip,thez)),2.0));
		// introduce artificial cutoff in comoving frame
		// Neutrons with energies above this threshold will interact with CMB photons before they decay; neutrino fluxes from neutron decay otherwise too high above this energy
		return pow(10,ncoIP(&s_neutron_in,log10(energy)))*af/ncoComputeDecayEscapeRate(NCO_NEUTRON,energy);
	}
}

double w_muplusr_steady(double energy)
{
	if (log10(energy) < s_muplusr_in.min || log10(energy) > s_muplusr_in.max) {return 0;}
	else {return pow(10,ncoIP(&s_muplusr_in,log10(energy)))/ncoComputeDecayEscapeRate(NCO_MU_PLUS_R,energy);}
}

double w_muplusl_steady(double energy)
{
	if (log10(energy) < s_muplusl_in.min || log10(energy) > s_muplusl_in.max) {return 0;}
	else {return pow(10,ncoIP(&s_muplusl_in,log10(energy)))/ncoComputeDecayEscapeRate(NCO_MU_PLUS_L,energy);}
}

double w_muminusr_steady(double energy)
{
	if (log10(energy) < s_muminusr_in.min || log10(energy) > s_muminusr_in.max) {return 0;}
	else {return pow(10,ncoIP(&s_muminusr_in,log10(energy)))/ncoComputeDecayEscapeRate(NCO_MU_MINUS_R,energy);}
}

double w_muminusl_steady(double energy)
{
	if (log10(energy) < s_muminusl_in.min || log10(energy) > s_muminusl_in.max) {return 0;}
	else {return pow(10,ncoIP(&s_muminusl_in,log10(energy)))/ncoComputeDecayEscapeRate(NCO_MU_MINUS_L,energy);}
}

void ncoComputeCosmoNeutrinoInjection(double z,double zmin,double deltaz,nco_ip* y,nco_targetphotons nphoton, nco_ip* rese,nco_ip* resmu,nco_ip* restau)
{

        thez=z;
	they=y;
	thephoton=nphoton;
	
	const int PIONIT = NCO_PHOTO_IT; // Type of interaction - here: all pion interactions.

	// Computation of the secondary particles from photohadronics and direct escape.

	double xa[NXN];
	double ya[NXN];
	double yb[NXN];
	double yc[NXN];
	double yd[NXN];
	double ypp[NXN];
//	mioInitOutput(""); // File name has to be set!
	double x,ex,respp,respm,resk,resn,resp;
	int n=0;
	
	// Neutron spectrum extends to higher E
	// This computes the secondary spectrum from photons and protons. pi+, pi-, K+ and neutrons are considered as resulting particles
	// The input spectra are in [GeV^-1 cm^-3], while the resulting spectra of secondary particles are in [GeV^-1 cm^-3 s^-1]
	for(x=XNMIN;x<XMAX+0.00001;x+=(XMAX-XNMIN)/(NXN-1)) 
	{
		ex=pow(10.0,x);
		respp=ncoComputeSecondarySpectrum(PIONIT,NCO_PROTON,NCO_PI_PLUS,grb_protons,grb_photons,ex); // pi+ production
		respm=ncoComputeSecondarySpectrum(PIONIT,NCO_PROTON,NCO_PI_MINUS,grb_protons,grb_photons,ex); // pi- production
		resk=ncoComputeSecondarySpectrum(NCO_K_PLUS_PROD,NCO_PROTON,NCO_K_PLUS,grb_protons,grb_photons,ex); // K+ and K-production
		resn=ncoComputeSecondarySpectrum(PIONIT,NCO_PROTON,NCO_NEUTRON,grb_protons,grb_photons,ex); // neutron production
  		xa[n]=x;
		if(respp>1e-250) ya[n]=log10(respp); else ya[n]=-250.0;
		if(respm>1e-250) yb[n]=log10(respm); else yb[n]=-250.0;
		if(resk>1e-250) yc[n]=log10(resk); else yc[n]=-250.0;
		if(resn>1e-250) yd[n]=log10(resn); else yd[n]=-250.0;
		n++;
//		mioAddToOutput5(x,respp*ex*ex,respm*ex*ex,resk*ex*ex,resn*ex*ex); // The spectra multiplied with the energy^2 are in [GeV cm^-3 s^-1].
	}
//	 mioCloseOutput();
//	 exit(-1);

	ncoIPAlloc(&s_piplus_in,xa,ya,n); // PI_PLUS

	ncoIPAlloc(&s_piminus_in,xa,yb,n); // PI_MINUS

	ncoIPAlloc(&s_kplus_in,xa,yc,n); // K_PLUS

	ncoIPAlloc(&s_neutron_in,xa,yd,n); // NEUTRON
	 
	// Decay spectra of neutrons, pions and kaons. The input spectra for the particle decay function need to be in [GeV^-1 cm^-3] while the resulting spectra are in [GeV^-1 cm^-3 s^-1].
	double xb[NXN];
	double ye[NXN];
	double yf[NXN];
	double yg[NXN];
	double yh[NXN];
	double nbe1[NXN];
	double nmu1[NXN];
	double nbmu2[NXN];
	double nmu3[NXN];
	// mioInitOutput("kaondata.dat");
	double resnne,respmr,respml,respn,resmpmr,resmpml,resmpn,reskm;
	int w=0;
	
	for(x=XNMIN;x<XMAX+0.00001;x+=(XMAX-XNMIN)/(NXN-1.0))
	{
		ex=pow(10,x);
		resnne=ncoComputeParticleDecaySpectrum(w_neutron_steady,NCO_NEUTRON,NCO_NU_BAR_E,ex); // First set of (anti-)electron neutrinos (from neutron decay) -> nbe1.
		respmr=ncoComputeParticleDecaySpectrum(w_piplus_steady,NCO_PI_PLUS,NCO_MU_PLUS_R,ex);
		respml=ncoComputeParticleDecaySpectrum(w_piplus_steady,NCO_PI_PLUS,NCO_MU_PLUS_L,ex);
		respn=ncoComputeParticleDecaySpectrum(w_piplus_steady,NCO_PI_PLUS,NCO_NU_MU,ex); // First set of muon neutrinos (from piâº decay) -> nmu1.
		resmpmr=ncoComputeParticleDecaySpectrum(w_piminus_steady,NCO_PI_MINUS,NCO_MU_MINUS_R,ex);
		resmpml=ncoComputeParticleDecaySpectrum(w_piminus_steady,NCO_PI_MINUS,NCO_MU_MINUS_L,ex);
		resmpn=ncoComputeParticleDecaySpectrum(w_piminus_steady,NCO_PI_MINUS,NCO_NU_BAR_MU,ex); // Second set of (anti-)muon neutrinos (from pi- decay ) -> nbmu2.
		reskm=ncoComputeParticleDecaySpectrum(w_kplus_steady,NCO_K_PLUS,NCO_NU_MU,ex); // Third set of muon neutrinos (from K+ decay) -> nmu3.
		// mioAddToOutput9(x,resnne*ex*ex,respmr*ex*ex,respml*ex*ex,respn*ex*ex,resmpmr*ex*ex,resmpml*ex*ex,resmpn*ex*ex,reskm*ex*ex); // The spectra multiplied with the energy^2 are in [GeV cm^-3 s^-1].
		xb[w]=x;
		if(respmr>1e-250) ye[w]=log10(respmr); else ye[w]=-250.0;
		if(respml>1e-250) yf[w]=log10(respml); else yf[w]=-250.0;
		if(resmpmr>1e-250) yg[w]=log10(resmpmr); else yg[w]=-250.0;
		if(resmpml>1e-250) yh[w]=log10(resmpml); else yh[w]=-250.0;
		if(resnne>1e-250) nbe1[w]=log10(resnne); else nbe1[w]=-250.0;
		if(respn>1e-250) nmu1[w]=log10(respn); else nmu1[w]=-250.0;
		if(resmpn>1e-250) nbmu2[w]=log10(resmpn); else nbmu2[w]=-250.0;
		if(reskm>1e-250) nmu3[w]=log10(reskm); else nmu3[w]=-250.0;
		w++;
	}
	// mioCloseOutput();

	// Interpolating functions for the injection spectra of the muons.
	ncoIPAlloc(&s_muplusr_in,xb,ye,w); // MU_PLUS_R

	ncoIPAlloc(&s_muplusl_in,xb,yf,w); // MU_PLUS_L

	ncoIPAlloc(&s_muminusr_in,xb,yg,w); // MU_MINUS_R

	ncoIPAlloc(&s_muminusl_in,xb,yh,w); // M_MINUS_L

	// Decay of muons - input in [GeV^-1 cm^-3], output in [GeV^-1 cm^-3 s^-1].
	double xc[NXN];
	double nbmu4[NXN];
	double ne2[NXN];
	double nbmu5[NXN];
	double ne3[NXN];
	double nmu6[NXN];
	double nbe4[NXN];
	double nmu7[NXN];
	double nbe5[NXN];
	// mioInitOutput("muondata.dat");
	double resmlbm,resmle,resmrbm,resmre,resmlm,resmlbe,resmrm,resmrbe;
	int v=0;
	
	for(x=XNMIN;x<XMAX+0.00001;x+=(XMAX-XNMIN)/(NXN-1.0))
	{
		ex=pow(10,x);
		resmlbm=ncoComputeParticleDecaySpectrum(w_muplusl_steady,NCO_MU_PLUS_L,NCO_NU_BAR_MU,ex); // Fourth set of (anti-)muon neutrinos -> nbmu4.
		resmle=ncoComputeParticleDecaySpectrum(w_muplusl_steady,NCO_MU_PLUS_L,NCO_NU_E,ex); // Second set of electron neutrinos -> ne2.
		resmrbm=ncoComputeParticleDecaySpectrum(w_muplusr_steady,NCO_MU_PLUS_R,NCO_NU_BAR_MU,ex); // Fifth set of (anti-)muon neutrinos -> nbmu5.
		resmre=ncoComputeParticleDecaySpectrum(w_muplusr_steady,NCO_MU_PLUS_R,NCO_NU_E,ex); // Third set of electron neutrinos -> ne3.
		resmlm=ncoComputeParticleDecaySpectrum(w_muminusl_steady,NCO_MU_MINUS_L,NCO_NU_MU,ex); // Sixth set of muon neutrinos -> nmu6.
		resmlbe=ncoComputeParticleDecaySpectrum(w_muminusl_steady,NCO_MU_MINUS_L,NCO_NU_BAR_E,ex); // Fourth set of (anti-)electron neutrinos -> nbe4.
		resmrm=ncoComputeParticleDecaySpectrum(w_muminusr_steady,NCO_MU_MINUS_R,NCO_NU_MU,ex); // Seventh set of muon neutrinos -> nmu7.
		resmrbe=ncoComputeParticleDecaySpectrum(w_muminusr_steady,NCO_MU_MINUS_R,NCO_NU_BAR_E,ex); // Fifth set of (anti-)electron neutrinos -> nbe5.
//		steadymuminusl=muminusr(ex);
		// mioAddToOutput10(x,resmlbm*ex*ex,resmle*ex*ex,resmrbm*ex*ex,resmre*ex*ex,resmlm*ex*ex,resmlbe*ex*ex,resmrm*ex*ex,resmrbe*ex*ex,steadymuminusl*ex*ex); // The spectra multiplied by the energy^2 are in [GeV cm^-3 s^-1].
		xc[v]=x;
		if(resmlbm>1e-250) nbmu4[v]=log10(resmlbm); else nbmu4[v]=-250.0;
		if(resmle>1e-250) ne2[v]=log10(resmle); else ne2[v]=-250.0;
		if(resmrbm>1e-250) nbmu5[v]=log10(resmrbm); else nbmu5[v]=-250.0;
		if(resmre>1e-250) ne3[v]=log10(resmre); else ne3[v]=-250.0;
		if(resmlm>1e-250) nmu6[v]=log10(resmlm); else nmu6[v]=-250.0;
		if(resmlbe>1e-250) nbe4[v]=log10(resmlbe); else nbe4[v]=-250.0;
		if(resmrm>1e-250) nmu7[v]=log10(resmrm); else nmu7[v]=-250.0;
		if(resmrbe>1e-250) nbe5[v]=log10(resmrbe); else nbe5[v]=-250.0;
		v++;
	}
	// mioCloseOutput();
	 
	// conversion injection spectrum to comoving density (GeV^-1 Mpc^-3) from real photon density (GeV^-1 cm^⁻3): 1/pow(30.857e23,3.0)
	// however, to avoid too small numbers, I did not put this factor, and I did not put it at the end (where I convert back again)
	double ToCMD(double n)
	{
		return n*deltaz/pow(1.0+z,4.0)/HubbleParameter(z);
		// dt/dz = 1/(1+z) *H^(-1) [s]; also divide by (1+z)^3 for n -> Y; corresponds to summand in integral int dt = int dz dt/dz = sum deltaz*dt/dz	  
	}
 
	double netot[NXN]; // Summing up the different contributions to a neutrino flavor. All functions in [GeV^-1 cm^-3 s^-1].
	double nmutot[NXN];
	double ntautot[NXN];
	double xd[NXN];
	// mioInitOutput("prodneutrinos.dat");
	int u=0;
	for(x=XNMIN;x<XMAX+0.00001;x+=(XMAX-XNMIN)/(NXN-1.0))
	{
		ex = pow(10.0,x);
		
		double nbe = pow(10.0,nbe1[u])+pow(10.0,nbe4[u])+pow(10,nbe5[u]);
		double ne = pow(10.0,ne2[u])+pow(10.0,ne3[u]);
		double nbmu = pow(10.0,nbmu2[u])+pow(10.0,nbmu4[u])+pow(10.0,nbmu5[u]);
		double nmu = pow(10.0,nmu1[u])+pow(10.0,nmu3[u])+pow(10.0,nmu6[u])+pow(10.0,nmu7[u]);
		
		// printf("x: %g nbe: %g ne: %g nbmu: %g nmu: %g\n",x,nbe,ne,nbmu,nmu);
		
		double nemix = ncoComputeFlavorMixing(NCO_NU_BAR_E,NCO_NU_BAR_E,theta12,theta13,theta23,deltacp)*nbe + 
		               ncoComputeFlavorMixing(NCO_NU_E,NCO_NU_E,theta12,theta13,theta23,deltacp)*ne + 
		               ncoComputeFlavorMixing(NCO_NU_BAR_MU,NCO_NU_BAR_E,theta12,theta13,theta23,deltacp)*nbmu + 
		               ncoComputeFlavorMixing(NCO_NU_MU,NCO_NU_E,theta12,theta13,theta23,deltacp)*nmu ;
			       
		double nmumix = ncoComputeFlavorMixing(NCO_NU_BAR_E,NCO_NU_BAR_MU,theta12,theta13,theta23,deltacp)*nbe + 
		                ncoComputeFlavorMixing(NCO_NU_E,NCO_NU_MU,theta12,theta13,theta23,deltacp)*ne + 
		                ncoComputeFlavorMixing(NCO_NU_BAR_MU,NCO_NU_BAR_MU,theta12,theta13,theta23,deltacp)*nbmu + 
		                ncoComputeFlavorMixing(NCO_NU_MU,NCO_NU_MU,theta12,theta13,theta23,deltacp)*nmu ;
			       
		double ntaumix = ncoComputeFlavorMixing(NCO_NU_BAR_E,NCO_NU_BAR_TAU,theta12,theta13,theta23,deltacp)*nbe + 
		                ncoComputeFlavorMixing(NCO_NU_E,NCO_NU_TAU,theta12,theta13,theta23,deltacp)*ne + 
		                ncoComputeFlavorMixing(NCO_NU_BAR_MU,NCO_NU_BAR_TAU,theta12,theta13,theta23,deltacp)*nbmu + 
		                ncoComputeFlavorMixing(NCO_NU_MU,NCO_NU_TAU,theta12,theta13,theta23,deltacp)*nmu ;

		// printf("x: %g ne: %g nm: %g nt: %g\n",x,nemix,nmumix,ntaumix);

		// Conversion from injection spectrum into comoving density contribution
		nemix=ToCMD(nemix);
		nmumix=ToCMD(nmumix);
		ntaumix=ToCMD(ntaumix);

		// Rescaling of co-moving density from z to zmin according to Eq. (31) of Mauricio's note
		xd[u]=x+log10((1.0+zmin)/(1.0+z)); 
		nemix=nemix*(1.0+z)/(1.0+zmin);
		nmumix=nmumix*(1.0+z)/(1.0+zmin);
		ntaumix=ntaumix*(1.0+z)/(1.0+zmin);

		if(nemix>1e-250) netot[u]=log10(nemix); else netot[u]=-250.0;
		if(nmumix>1e-250) nmutot[u]=log10(nmumix); else nmutot[u]=-250.0;
		if(ntaumix>1e-250) ntautot[u]=log10(ntaumix); else ntautot[u]=-250.0;

		// mioAddToOutput5(x,pow(10,nbetot[u])*pow(10.0,2.0*x),pow(10,netot[u])*pow(10.0,2.0*x),pow(10,nbmutot[u])*pow(10.0,2.0*x),pow(10,nmutot[u])*pow(10.0,2.0*x));
		u++;
	}
	// mioCloseOutput();

	ncoIPAlloc(rese,xd,netot,u);
	ncoIPAlloc(resmu,xd,nmutot,u);
	ncoIPAlloc(restau,xd,ntautot,u);

	ncoIPFree(&s_neutron_in);
	ncoIPFree(&s_piplus_in);
	ncoIPFree(&s_piminus_in);
	ncoIPFree(&s_kplus_in);
	ncoIPFree(&s_muplusr_in);
	ncoIPFree(&s_muplusl_in);
	ncoIPFree(&s_muminusr_in);
	ncoIPFree(&s_muminusl_in);
}


//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//// 											      ////
//// 			SOLUTION OF THE BOLTZMANN EQUATION AND EVOLUTION		      ////
//// 											      ////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
   
//================================================================================================
// ncoCRPropStep V4										//
//												//
// Propagates the comoving number density of CR spec_from_z at redshift from_z (input) to 	//
// spec_to_z at redshift to_z (output), by numerically solving a Boltzmann transport equation.	//
//												//
// Input:											//
//												//
//	from_z: initial redshift								//
//	to_z: final redshift									//
//	spec_from_z: (log of) comoving number density of CR at z = from_z [GeV^-1 Mpc^-3]	//
//	spec_to_z: (log of) comoving CR spectrum at z = to_z [GeV^-1 Mpc^-3]			//
//	crinj: the CR injection spectrum is a function of type nco_crinjspectrum		//
//	       (of x=log10(E) and redshift); see definition [GeV^-1 s^-1 Mpc^-3]		//
//	cib_target: the CIB target photon spectrum is function of type nco_targetphotons	//
//		    (of x and z); see defintion [GeV^-1 cm^-3]					//
//	wcasstep: e.m. cascade energy density for this redshift step [GeV^-1 cm^-3]		//
//												//
// Output:											//
//												//
//	The (log of) comoving CR spectrum spec_to_z calculated at z = to_z			//
//	units: (log of) GeV^-1 Mpc^-3								//
//												//
// Ref.:											//
//												//
//	M. Ahlers, L.A. Anchordoqui, and S. Sarkar, Phys. Rev. D 79, 083009 (2009) [0902.3993]	//
//												//
//	M. Ahlers, M.C. Gonzalez-Garcia, and F. Halzen, Astropart.Phys. 35, 87 (2011) 		//
//		[1103.3421]									//
//												//
// Created: 18/06/2012										//
// Last modified: 26/11/2012									//
//================================================================================================

void ncoCRPropStep4(double from_z,double to_z,nco_ip* spec_from_z,nco_ip* spec_to_z,nco_crinjspectrum crinj,nco_targetphotons cib_target, double* wcasstep)
{
	// total target photon spectrum
	double target(double E,double z)
	{
		double res=0.0;
		
		if (NCO_CRPROP_PHOTOPION_CMB_LOSSES==1) res+=CMBPhotonSpectrum(E,z);
		if (NCO_CRPROP_PHOTOPION_CIB_LOSSES==1) res+=cib_target(E,z);

		return res;
	}

	int i;
	double x, deltax;

	double pre, corr, tloss, tgain, YFromZ;
	double tPairProductionLossCMBPreDeriv;
	double tPairProductionLossCIBPreDeriv;
	double tPhotopionLossCMBPreDeriv;
	double tPhotopionLossCIBPreDeriv;
	double arrx[NX], arrE[NX], y[NX];
	double arrCRInjectionRate[NX];
	double arrPairProductionLossPreDeriv[NX];
	double arrPhotopionLossPreDeriv[NX];
	double arrTotalLossPreDeriv[NX];
	double arrTotalLoss[NX];
	
	double bcas;
	double arrEMCascEnergyDensity[NX];
	
	char* filenamebase;
	
	// Interpolation functions
	nco_ip TotalLossPreDeriv_ip;
	nco_ip EMCascEnergyDensity_ip;

	deltax = (XMAX-XMIN)/((double)NX-1.0);  // Step size in the x-direction
	
	CIBCOUNTER++;
	CASCADECOUNTER++;

	// STEP 1: calculate the functions to be derived at NX different values of x = log10(E), with E the energy in the comoving frame

	pre = 1.0/(1.0+from_z)/HubbleParameter(from_z); // Prefactor 
	
	x = XMIN;
	for (i=0; i<NX; i++)
	{
		arrx[i] = x; 

		tPairProductionLossCMBPreDeriv = 0.0;
		//tPairProductionLossCIBPreDeriv = 0.0;
		tPhotopionLossCMBPreDeriv = 0.0;
		//tPhotopionLossCIBPreDeriv = 0.0;
		
		// Proton density at redshift = from_z
		YFromZ = pow(10.0,ncoIP(spec_from_z,x));
		
		// Add adiabatic loss term (prior to energy differentiation)
		arrTotalLossPreDeriv[i] = fAdiabaticLossPreDeriv(x,from_z,YFromZ);
  
		if (NCO_CRPROP_PAIRPROD_CMB_LOSSES == 1) // Pair production on CMB
                {  
			tPairProductionLossCMBPreDeriv = -fPairProductionLossPreDeriv(x,from_z,YFromZ,&CMBPhotonSpectrum); 
                }
          
		if ((NCO_CRPROP_PAIRPROD_CIB_LOSSES == 1) && (CIBCOUNTER == CIBSTEPS)) // Pair production on CIB only updated every CIBSTEPS and stored in global variable
                {
			arrPairProductionCIBLoss[i] = -fPairProductionLossPreDeriv(x,from_z,YFromZ,cib_target);
                }
                else if (NCO_CRPROP_PAIRPROD_CIB_LOSSES != 1) arrPairProductionCIBLoss[i] = 0.0;

		if (NCO_CRPROP_PHOTOPION_CMB_LOSSES == 1) // Photohadronic interactions on CMB
                {
			tPhotopionLossCMBPreDeriv = -fPhotoPionLossPreDeriv(x,from_z,YFromZ,&CMBPhotonSpectrum);
                }

		if ((NCO_CRPROP_PHOTOPION_CIB_LOSSES == 1) && (CIBCOUNTER == CIBSTEPS)) // Photohadronic interactions on CIB only updated every CIBSTEPS and stored in global variable
                {
			arrPhotoPionCIBLoss[i] = -fPhotoPionLossPreDeriv(x,from_z,YFromZ,cib_target);
                }
                else if (NCO_CRPROP_PHOTOPION_CIB_LOSSES != 1) arrPhotoPionCIBLoss[i] = 0.0;

		// Add all the energy-loss contributions and convert to log
		arrTotalLossPreDeriv[i] += tPairProductionLossCMBPreDeriv + arrPairProductionCIBLoss[i] + tPhotopionLossCMBPreDeriv + arrPhotoPionCIBLoss[i];
		if (arrTotalLossPreDeriv[i] > 1.e-250) arrTotalLossPreDeriv[i] = log10(arrTotalLossPreDeriv[i]);
		else arrTotalLossPreDeriv[i] = -250.0;
		
		// CR injection term
		arrCRInjectionRate[i] = crinj(x,from_z);
	
		// Calculate the contribution to the total e.m. cascade from this redshift
		if ((NCO_CRPROP_EMCASC_E_DENSITY == 1) && (CASCADECOUNTER == CASCADESTEPS))
		{
			// WW: WHY ARE THERE 10^loss RATES USED? I MEAN tPairProductionLossCMBPreDeriv ETC, ARE NOT LOG FUNCTIONS!?
			// Watch here also that global variables are used for CIB losses
			// Also: Logic wrong: add in very step, but only compute (evaluate) every CIBSTEPS! -> fixes below
			bcas = 0.0; // bcas is the total energy loss rate into electrons, positrons, and photons (i.e., pair production + pions)
			if (NCO_CRPROP_PAIRPROD_CMB_LOSSES == 1) bcas += pow(10.0,tPairProductionLossCMBPreDeriv);
			//if ((NCO_CRPROP_PAIRPROD_CIB_LOSSES == 1) && (CIBCOUNTER == CIBSTEPS)) bcas += pow(10.0,tPairProductionLossCIBPreDeriv);
			if (NCO_CRPROP_PAIRPROD_CIB_LOSSES == 1) bcas += pow(10.0,arrPairProductionCIBLoss[i]);
			// The factor 5/8 is because five out of the eight decay products inject energy into the e.m. cascades (the other three are neutrinos)
			if (NCO_CRPROP_PHOTOPION_CMB_LOSSES == 1) bcas += pow(10.0,tPhotopionLossCMBPreDeriv)*5.0/8.0;
			//if ((NCO_CRPROP_PHOTOPION_CIB_LOSSES == 1)  && (CIBCOUNTER==CIBSTEPS)) bcas += pow(10.0,tPhotopionLossCIBPreDeriv)*5.0/8.0;
			if (NCO_CRPROP_PHOTOPION_CIB_LOSSES == 1) bcas += pow(10.0,arrPhotoPionCIBLoss[i])*5.0/8.0;
			// arrEMCascEnergyDensity is the integrand to calculate the energy density at redshift from_z
			arrEMCascEnergyDensity[i] = bcas*pre/(1.0+from_z)*YFromZ;
			if (arrEMCascEnergyDensity[i] > 1.e-250) arrEMCascEnergyDensity[i] = log10(arrEMCascEnergyDensity[i]); // double log
			else arrEMCascEnergyDensity[i] = -250.0;
		}
		
		x += deltax;
	}
	
	// STEP 2: build interpolation functions for the functions to be derived (adiabatic losses, pair production losses, and photopion losses)
	
	ncoIPAllocLinear(&TotalLossPreDeriv_ip,arrx,arrTotalLossPreDeriv,NX);
	
	if ((NCO_CRPROP_EMCASC_E_DENSITY == 1) && (CASCADECOUNTER == CASCADESTEPS))
	{
		ncoIPAllocLinear(&EMCascEnergyDensity_ip,arrx,arrEMCascEnergyDensity,NX);
		double fEMCascEnergyDensityIntegrand(double xx)
		{
			return pow(10.0,xx)*pow(10.0,ncoIP(&EMCascEnergyDensity_ip,xx)); // The 10^x = E is added because integration is in x = log10(E), not in E
		}
	
		// Calculate all of the energy dumped into electrons and photons at the UHECR energies (E >~ 10^16 GeV), which eventually cascade down to 
		// the Fermi-LAT range (GeV - 100 GeV)
		*wcasstep = log(10.0)*ncoIntegrate(&fEMCascEnergyDensityIntegrand,XMIN,XMAX,21);
		ncoIPFree(&EMCascEnergyDensity_ip);
	}
	
	for (i=0; i<NX; i++)
	{ 
		// STEP 3: calculate the derivative w.r.t. x for the adiabatic losses, pair production losses, and photopion losses terms at NX different values of x
		tloss = ncoSymmDeriv(&TotalLossPreDeriv_ip,arrx[i]);
		tgain = arrCRInjectionRate[i];
		corr = DELTAZ*pre*(tloss+tgain);

		// STEP 4: compute the spectrum Y(to_z) at NX different values of x
		// WW edited, for stability + precision: when the next-step correction corr of the comoving density y is positive, just add it to the current value of y;
		// otherwise, define a new, positive denom = 1-corr/y, and divide the comoving density by it. This prevents y from ever being negative.
		double val,tv;
		YFromZ = pow(10.0,ncoIP(spec_from_z,arrx[i]));
                tv = YFromZ+corr;
		if (tv >= 0.0) val = tv;
		else
                { 
			double denom = 1.0-corr/YFromZ;

			if (fabs(denom) > 0.0) 
				val = YFromZ / denom;
			else
				val = 0.0;
		}
		// Calculate the y-values at the next redshift value to_z = from_z-DELTAZ
		if (val > 1e-250) y[i] = log10(val); // important: use same cutoff (10^-250) everywhere, otherwise numerical instabilities possible!
		else y[i] = -250.0;
	}
	
	// STEP 5: build an interpolation function for the resulting spectrum at z = to_z
	ncoIPAllocLinear(spec_to_z,arrx,y,NX); 
	
	// Print the resulting spectrum at z = to_z if NCO_CRPROP_TRACKSPECTRUM == 1
	if (NCO_CRPROP_TRACKSPECTRUM == 1)
	{
// 		if ( fabs(to_z) <= DELTAZ )
		if ( (to_z <= 6.5 && fabs(to_z-round(to_z)) <= DELTAZ) || from_z == 6.0)
// 		if (fabs(to_z-5.95) <= 1.001*DELTAZ)
		{
// 			printf("from_z = %f   to_z = %f \n", from_z,to_z);
			if (NCO_CRPROP_WRITESPECTRUM == 0) // Print to screen
			{
				ncoIPPrint(spec_to_z,"",NX); 
			}
			else if (NCO_CRPROP_WRITESPECTRUM == 1) // Print to external file
			{
				// Generate a filename dynamically for this value of z
				if (from_z == 6.0)
				{
					sprintf(filenamebody,"%d",(int)(from_z+0.5));
				}
				else
				{
	 				sprintf(filenamebody,"%d",(int)(to_z+0.5));
// 					sprintf(filenamebody,"%d",(int)(to_z*100+0.5));
				}
				if (NCO_CRPROP_PAIRPROD_CMB_LOSSES == 0 && NCO_CRPROP_PHOTOPION_CMB_LOSSES == 0)
				{
					filenamebase = filenamebase1;
				}
				else if (NCO_CRPROP_PAIRPROD_CMB_LOSSES == 1 && NCO_CRPROP_PHOTOPION_CMB_LOSSES == 0)
				{
					filenamebase = filenamebase2;
				}
				else if (NCO_CRPROP_PAIRPROD_CMB_LOSSES == 1 && NCO_CRPROP_PHOTOPION_CMB_LOSSES == 1)
				{
					filenamebase = filenamebase3;
				}
				char* s = malloc(snprintf(NULL, 0, "%s%s%s", filenamebase, filenamebody, filenameext));
				sprintf(s, "%s%s%s", filenamebase, filenamebody, filenameext);
// 				sprintf(s, "spectrum.adiabatic.+.pair.prod.z.5.95.dat");
				if (from_z == 6.0)
				{
					ncoIPPrint(spec_from_z,s,NX-1);
				}
				else
				{
					ncoIPPrint(spec_to_z,s,NX-1);
// 					ncoIPPrint(spec_to_z,"spectrum.adiabatic.+.pair.prod.z.5.95.dat",NX-1);
				}
			}
		}
	}
	
	// Destroy unneeded interpolation functions to free memory space
	ncoIPFree(&TotalLossPreDeriv_ip);

	if (CIBCOUNTER == CIBSTEPS) CIBCOUNTER=0;
	if (CASCADECOUNTER == CASCADESTEPS) CASCADECOUNTER=0;

	return;
}


//================================================================================================
// ncoCRPropStep										//
//												//
// Propagates the comoving number density of CR spec_from_z at redshift from_z (input) to 	//
// spec_to_z at redshift to_z (output), by numerically solving a Boltzmann transport equation.	//
//												//
// Input:											//
//												//
//	from_z: initial redshift								//
//	to_z: final redshift									//
//	spec_from_z: (log of) comoving number density of CR at z = from_z [GeV^-1 Mpc^-3]	//
//	spec_to_z: (log of) comoving CR spectrum at z = to_z [GeV^-1 Mpc^-3]			//
//	crinj: the CR injection spectrum is a function of type nco_crinjspectrum		//
//	       (of x=log10(E) and redshift); see definition [GeV^-1 s^-1 Mpc^-3]		//
//	cib_target: the CIB target photon spectrum is function of type nco_targetphotons	//
//		    (of x and z); see defintion [GeV^-1 cm^-3]					//
//	wcasstep: e.m. cascade energy density for this redshift step [GeV^-1 cm^-3]		//
//												//
// Output:											//
//												//
//	The (log of) comoving CR spectrum spec_to_z calculated at z = to_z			//
//	units: (log of) GeV^-1 Mpc^-3								//
//												//
// Ref.:											//
//												//
//	M. Ahlers, L.A. Anchordoqui, and S. Sarkar, Phys. Rev. D 79, 083009 (2009) [0902.3993]	//
//												//
//	M. Ahlers, M.C. Gonzalez-Garcia, and F. Halzen, Astropart.Phys. 35, 87 (2011) 		//
//		[1103.3421]									//
//												//
// Created: 18/06/2012										//
// Last modified: 26/11/2012									//
//================================================================================================

double specgrid[1000]; // store F = E^3 Y  externally

void ncoCRPropStep(double from_z,double to_z,nco_ip* spec_from_z,nco_ip* spec_to_z,nco_crinjspectrum crinj,nco_targetphotons cib_target, double* wcasstep)
{
	// total target photon spectrum
	double target(double E,double z)
	{
		double res=0.0;
		
		if (NCO_CRPROP_PHOTOPION_CMB_LOSSES==1) res+=CMBPhotonSpectrum(E,z);
		if (NCO_CRPROP_PHOTOPION_CIB_LOSSES==1) res+=cib_target(E,z);

		return res;
	}

	int i;
	double x, deltax;

	double pre, pre1, corr, tloss, tgain, YFromZ;
	double tPairProductionLossCMBPreDeriv;
	double tPairProductionLossCIBPreDeriv;
	double tPhotopionLossCMBPreDeriv;
	double tPhotopionLossCIBPreDeriv;
	double tPairProductionLossCMBPreDeriv1;
	double tPairProductionLossCIBPreDeriv1;
	double tPhotopionLossCMBPreDeriv1;
	double tPhotopionLossCIBPreDeriv1;
	double arrx[NX], arrE[NX], y[NX];
	double arrCRInjectionRate[NX];
	double arrCRInjectionRate1[NX];
	double arrPairProductionLossPreDeriv[NX];
	double arrPhotopionLossPreDeriv[NX];
	double arrTotalLossPreDeriv[NX];
	double arrTotalLossPreDeriv1[NX];
	double arrTotalLoss[NX];
	
	double bcas;
	double arrEMCascEnergyDensity[NX];
	
	char* filenamebase;
	
	// Interpolation functions
	// nco_ip TotalLossPreDeriv_ip;
	nco_ip EMCascEnergyDensity_ip;

	deltax = (XMAX-XMIN)/((double)NX-1.0);  // Step size in the x-direction
	
	CIBCOUNTER++;
	CASCADECOUNTER++;

	// STEP 1: calculate the functions to be derived at NX different values of x = log10(E), with E the energy in the comoving frame
	
	// Quantities with ...1 are for timestep to_z (Crank Nicolson solver)

	pre = 1.0/(1.0+from_z)/HubbleParameter(from_z); // Prefactor 
	pre1 = 1.0/(1.0+to_z)/HubbleParameter(to_z); // Prefactor 
	
	x = XMIN;
	for (i=0; i<NX; i++)
	{
		arrx[i] = x; 

		tPairProductionLossCMBPreDeriv = 0.0;
		tPairProductionLossCIBPreDeriv = 0.0;
		tPhotopionLossCMBPreDeriv = 0.0;
		tPhotopionLossCIBPreDeriv = 0.0;
		tPairProductionLossCMBPreDeriv1 = 0.0;
		tPairProductionLossCIBPreDeriv1 = 0.0;
		tPhotopionLossCMBPreDeriv1 = 0.0;
		tPhotopionLossCIBPreDeriv1 = 0.0;
		
		// Proton density at redshift = from_z
		YFromZ = specgrid[i]*pow(10.0,-3.0*x);   // use spectral grid directly to increase stability
		
		// Add adiabatic loss term (prior to energy differentiation)
		arrTotalLossPreDeriv[i] = fAdiabaticLossPreDeriv(x,from_z,YFromZ);
		arrTotalLossPreDeriv1[i] = fAdiabaticLossPreDeriv(x,to_z,1.0);
		
		if (NCO_CRPROP_PAIRPROD_CMB_LOSSES == 1) // Pair production on CMB
                {  
			tPairProductionLossCMBPreDeriv = -fPairProductionLossPreDeriv(x,from_z,YFromZ,&CMBPhotonSpectrum); 
			tPairProductionLossCMBPreDeriv1 = -fPairProductionLossPreDeriv(x,to_z,1.0,&CMBPhotonSpectrum); 
		}

		if (NCO_CRPROP_PAIRPROD_CIB_LOSSES == 1) // Pair production on CIB only updated every CIBSTEPS and stored in global variable
                {
		   if(CIBCOUNTER == CIBSTEPS)  // update
		   {
  			if(CIBSTEPS>1) 
			{
       			  tPairProductionLossCIBPreDeriv = arrPairProductionCIBLoss[i]*YFromZ;  
  			  arrPairProductionCIBLoss[i] = -fPairProductionLossPreDeriv(x,to_z,1.0,cib_target);
			  tPairProductionLossCIBPreDeriv1 = arrPairProductionCIBLoss[i];
			}
			else
			{
			  tPairProductionLossCIBPreDeriv = arrPairProductionCIBLoss[i]*YFromZ;  
			  tPairProductionLossCIBPreDeriv1 = -fPairProductionLossPreDeriv(x,to_z,1.0,cib_target);
			  arrPairProductionCIBLoss[i] = tPairProductionLossCIBPreDeriv1;
			}
			
		   }
		   else  // do not uodate
		   {
			  tPairProductionLossCIBPreDeriv = arrPairProductionCIBLoss[i]*YFromZ;  
			  tPairProductionLossCIBPreDeriv1 = arrPairProductionCIBLoss[i];		     
		   }
                }
                else
		{
		  tPairProductionLossCIBPreDeriv = 0.0;
		  tPairProductionLossCIBPreDeriv1 = 0.0;
		}


		if ((NCO_CRPROP_PAIRPROD_CIB_LOSSES == 1) && (CIBCOUNTER == CIBSTEPS)) // Pair production on CIB
                {
			tPairProductionLossCIBPreDeriv = -fPairProductionLossPreDeriv(x,from_z,YFromZ,cib_target);
			tPairProductionLossCIBPreDeriv1 = -fPairProductionLossPreDeriv(x,to_z,1.0,cib_target);
                }

		if (NCO_CRPROP_PHOTOPION_CMB_LOSSES == 1) // Photohadronic interactions on CMB
                {
			tPhotopionLossCMBPreDeriv = -fPhotoPionLossPreDeriv(x,from_z,YFromZ,&CMBPhotonSpectrum);
			tPhotopionLossCMBPreDeriv1 = -fPhotoPionLossPreDeriv(x,to_z,1.0,&CMBPhotonSpectrum);
		}

		if ((NCO_CRPROP_PHOTOPION_CIB_LOSSES == 1) && (CIBCOUNTER == CIBSTEPS)) // Photohadronic interactions on CIB only updated every CIBSTEPS and stored in global variable
                {
		    if(CIBCOUNTER == CIBSTEPS)
		    {
  			if(CIBSTEPS>1) 
			{
      			   tPhotopionLossCIBPreDeriv = arrPhotoPionCIBLoss[i]*YFromZ;
    		           arrPhotoPionCIBLoss[i] = -fPhotoPionLossPreDeriv(x,to_z,1.0,cib_target);
			   tPhotopionLossCIBPreDeriv1 = arrPhotoPionCIBLoss[i];
			}
			else
			{
			   tPhotopionLossCIBPreDeriv = arrPhotoPionCIBLoss[i]*YFromZ;
			   tPhotopionLossCIBPreDeriv1 = -fPhotoPionLossPreDeriv(x,to_z,1.0,cib_target);
			   arrPhotoPionCIBLoss[i] = tPhotopionLossCIBPreDeriv1;
			}
		    }
		    else
		    {
  			   tPhotopionLossCIBPreDeriv = arrPhotoPionCIBLoss[i]*YFromZ;
			   tPhotopionLossCIBPreDeriv1 = arrPhotoPionCIBLoss[i];		      
		    }
                }
                else 
		{
  			   tPhotopionLossCIBPreDeriv = 0.0;
			   tPhotopionLossCIBPreDeriv1 = 0.0;		      
		}

		// Add all the energy-loss contributions and convert to log
		arrTotalLossPreDeriv[i] += tPairProductionLossCMBPreDeriv + tPairProductionLossCIBPreDeriv + tPhotopionLossCMBPreDeriv + tPhotopionLossCIBPreDeriv;
		arrTotalLossPreDeriv1[i] += tPairProductionLossCMBPreDeriv1 + tPairProductionLossCIBPreDeriv1 + tPhotopionLossCMBPreDeriv1 + tPhotopionLossCIBPreDeriv1;

		// CR injection term
		arrCRInjectionRate[i] = crinj(x,from_z);
		arrCRInjectionRate1[i] = crinj(x,to_z);
		
		// Calculate the contribution to the total e.m. cascade from this redshift
		if ((NCO_CRPROP_EMCASC_E_DENSITY == 1) && (CASCADECOUNTER == CASCADESTEPS))
		{
		        // WW: WHY ARE THERE 10^loss RATES USED? I MEAN tPairProductionLossCMBPreDeriv ETC, ARE NOT LOG FUNCTIONS!? -> ALSO WRONG IN CRPropagation4.c?
			bcas = 0.0; // bcas is the total energy loss rate into electrons, positrons, and photons (i.e., pair production + pions)
			if (NCO_CRPROP_PAIRPROD_CMB_LOSSES == 1) bcas += pow(10.0,tPairProductionLossCMBPreDeriv);
			if ((NCO_CRPROP_PAIRPROD_CIB_LOSSES == 1) && (CIBCOUNTER == CIBSTEPS)) bcas += pow(10.0,tPairProductionLossCIBPreDeriv);
			// The factor 5/8 is because five out of the eight decay products inject energy into the e.m. cascades (the other three are neutrinos)
			if (NCO_CRPROP_PHOTOPION_CMB_LOSSES == 1) bcas += pow(10.0,tPhotopionLossCMBPreDeriv)*5.0/8.0;
			if ((NCO_CRPROP_PHOTOPION_CIB_LOSSES == 1)  && (CIBCOUNTER==CIBSTEPS)) bcas += pow(10.0,tPhotopionLossCIBPreDeriv)*5.0/8.0;
			// arrEMCascEnergyDensity is the integrand to calculate the energy density at redshift from_z
			arrEMCascEnergyDensity[i] = bcas*pre/(1.0+from_z)*YFromZ;
			if (arrEMCascEnergyDensity[i] > 1.e-250) arrEMCascEnergyDensity[i] = log10(arrEMCascEnergyDensity[i]); // double log
			else arrEMCascEnergyDensity[i] = -250.0;
		}
		
		x += deltax;
	}
	
	// STEP 2: build interpolation functions for the functions to be derived (adiabatic losses, pair production losses, and photopion losses)
	
	// ncoIPAllocLinear(&TotalLossPreDeriv_ip,arrx,arrTotalLossPreDeriv,NX);
	
	if ((NCO_CRPROP_EMCASC_E_DENSITY == 1) && (CASCADECOUNTER == CASCADESTEPS))
	{
		ncoIPAllocLinear(&EMCascEnergyDensity_ip,arrx,arrEMCascEnergyDensity,NX);
		double fEMCascEnergyDensityIntegrand(double xx)
		{
			return pow(10.0,xx)*pow(10.0,ncoIP(&EMCascEnergyDensity_ip,xx)); // The 10^x = E is added because integration is in x = log10(E), not in E
		}
	
		// Calculate all of the energy dumped into electrons and photons at the UHECR energies (E >~ 10^16 GeV), which eventually cascade down to 
		// the Fermi-LAT range (GeV - 100 GeV)
		*wcasstep = log(10.0)*ncoIntegrate(&fEMCascEnergyDensityIntegrand,XMIN,XMAX,21);
		ncoIPFree(&EMCascEnergyDensity_ip);
	}
	
	double coeffa[NX];
        double coeffb[NX];
        double coeffc[NX];
        double coeffr[NX];

        // GSL stuff
        gsl_vector* d = gsl_vector_alloc(NX);
        gsl_vector* e = gsl_vector_alloc(NX-1);
        gsl_vector* f = gsl_vector_alloc(NX-1);
        gsl_vector* vx = gsl_vector_alloc(NX);
        gsl_vector* b = gsl_vector_alloc(NX);
	
	for (i=0; i<NX; i++)
	{ 
	        double ex = pow(10.0,arrx[i]);
		
		// Here arrTotalLossPreDeriv does not contain Y (just coefficient)

		if(i>0) coeffa[i]=+0.25/(deltax*log(10.0))*pre1*DELTAZ*arrTotalLossPreDeriv1[i-1]/pow(10.0,arrx[i-1]); 
		if(i<NX-1) coeffc[i]=-0.25/(deltax*log(10.0))*pre1*DELTAZ*arrTotalLossPreDeriv1[i+1]/pow(10.0,arrx[i+1]);

		/* old version before bugfix on May 19, 2015: 
		coeffa[i]=+0.25/(deltax*log(10.0))*pre1*DELTAZ/ex*arrTotalLossPreDeriv1[i];  // Sign depends on how solver set up -> try
		coeffc[i]=-coeffa[i];
		*/
		
                coeffb[i]=1.0+1.0*DELTAZ*pre1*arrTotalLossPreDeriv1[i]/ex;
		
		
                if((i>0) && (i<NX-1)) 
         	 coeffr[i]=specgrid[i]  //  + - + +
         	   +0.25*DELTAZ*pre/(deltax*log(10.0))*(arrTotalLossPreDeriv[i+1]*pow(10.0,2.0*arrx[i+1])-arrTotalLossPreDeriv[i-1]*pow(10.0,2.0*arrx[i-1])) // contains Y in terms, need F= E^3 Y here. One power of E cancels
         	   -1.0*pre*DELTAZ*arrTotalLossPreDeriv[i]*pow(10.0,2.0*arrx[i])
         	   +0.5*pre*arrCRInjectionRate[i]*DELTAZ*pow(10.0,3.0*arrx[i])+0.5*pre1*arrCRInjectionRate1[i]*DELTAZ*pow(10.0,3.0*arrx[i]); 
                else 
   	         coeffr[i]=specgrid[i];
   	        
   	        // printf("i: %i a: %g b: %g c: %g r: %g pre: %g\n",i,coeffa[i],coeffb[i]-1.0,coeffc[i],coeffr[i],pre);


	}
	
  
        // Gsl solver initialize
        for(i=0;i<NX;i++)
        { 
          gsl_vector_set(d, i, coeffb[i]);
          if(i<NX-1) gsl_vector_set(e, i, coeffc[i]);
          if(i>0) gsl_vector_set(f, i-1, coeffa[i]);
          gsl_vector_set(b, i, coeffr[i]);
        }
	
      
        // solve tridiagonal system -> result in y again
        int res=gsl_linalg_solve_tridiag(d, e, f, b, vx);
	if(res>0) printf("Error tridiag system: %i",res);
	
        for(i=0;i<NX;i++)
	{
	  y[i]=gsl_vector_get(vx, i);
	  specgrid[i]=y[i];
	  if (y[i]/pow(10.0,3.0*arrx[i]) > 1e-250) y[i] = log10(y[i]/pow(10.0,3.0*arrx[i])); // only for external interface; calc uses specgrid (F=E^3 Y)
         	else y[i] = -250.0;
	}	  
	
	ncoIPAllocLinear(spec_to_z,arrx,y,NX); 
	
        // GSL stuff
        gsl_vector_free(d);
        gsl_vector_free(e);
        gsl_vector_free(f);
        gsl_vector_free(vx);
        gsl_vector_free(b);
	
	// Print the resulting spectrum at z = to_z if NCO_CRPROP_TRACKSPECTRUM == 1
	if (NCO_CRPROP_TRACKSPECTRUM == 1)
	{
// 		if ( fabs(to_z) <= DELTAZ )
		if ( (to_z <= 6.5 && fabs(to_z-round(to_z)) <= DELTAZ) || from_z == 6.0)
// 		if (fabs(to_z-5.95) <= 1.001*DELTAZ)
		{
// 			printf("from_z = %f   to_z = %f \n", from_z,to_z);
			if (NCO_CRPROP_WRITESPECTRUM == 0) // Print to screen
			{
				ncoIPPrint(spec_to_z,"",NX); 
			}
			else if (NCO_CRPROP_WRITESPECTRUM == 1) // Print to external file
			{
				// Generate a filename dynamically for this value of z
				if (from_z == 6.0)
				{
					sprintf(filenamebody,"%d",(int)(from_z+0.5));
				}
				else
				{
	 				sprintf(filenamebody,"%d",(int)(to_z+0.5));
// 					sprintf(filenamebody,"%d",(int)(to_z*100+0.5));
				}
				if (NCO_CRPROP_PAIRPROD_CMB_LOSSES == 0 && NCO_CRPROP_PHOTOPION_CMB_LOSSES == 0)
				{
					filenamebase = filenamebase1;
				}
				else if (NCO_CRPROP_PAIRPROD_CMB_LOSSES == 1 && NCO_CRPROP_PHOTOPION_CMB_LOSSES == 0)
				{
					filenamebase = filenamebase2;
				}
				else if (NCO_CRPROP_PAIRPROD_CMB_LOSSES == 1 && NCO_CRPROP_PHOTOPION_CMB_LOSSES == 1)
				{
					filenamebase = filenamebase3;
				}
				char* s = malloc(snprintf(NULL, 0, "%s%s%s", filenamebase, filenamebody, filenameext));
				sprintf(s, "%s%s%s", filenamebase, filenamebody, filenameext);
// 				sprintf(s, "spectrum.adiabatic.+.pair.prod.z.5.95.dat");
				if (from_z == 6.0)
				{
					ncoIPPrint(spec_from_z,s,NX-1);
				}
				else
				{
					ncoIPPrint(spec_to_z,s,NX-1);
// 					ncoIPPrint(spec_to_z,"spectrum.adiabatic.+.pair.prod.z.5.95.dat",NX-1);
				}
			}
		}
	}
	
	// Destroy unneeded interpolation functions to free memory space
//	ncoIPFree(&TotalLossPreDeriv_ip);

	if (CIBCOUNTER == CIBSTEPS) CIBCOUNTER=0;
	if (CASCADECOUNTER == CASCADESTEPS) CASCADECOUNTER=0;

	return;
}

//================================================================================================
// ncoCRProp											//
//												//
// Propagates comoving number density of CR from zmax to zmin, taking into account adiabatic	//
// losses, interaction losses, and CR injection by sources. Calls ncoCRPropStep starting from 	//
// zmax, propagating the spectrum one redshift step further every step, down to zmin.		//
//												//
// Input:											//
//												//
//	zmin: minimum redshift									//
//	zmax: maximum redshift									//
//	crinj: the CR injection spectrum is a function of type nco_crinjspectrum		//
//	       (of x=log10(E) and redshift); see definition [GeV^-1 s^-1 Mpc^-3]		//
//	cib_target: the CIB target photon spectrum is function of type nco_targetphotons	//
//		    (of x and z); see defintion [GeV^-1 cm^-3]					//
//	wcas: (log10 of) e.m. cascade energy density [GeV^-1 cm^-3]				//
//												//
// Output:											//
//												//
//	The CR flux at z = zmin is returned as "result"						//
//	units: GeV^-1 cm^-2 s^-1 sr^-1								//
//	The total neutrino flux at z=zmin is also returned: nue, numu, nutau			//
//      units: GeV^-1 cm^-2 s^-1 sr^-1								//
//	(these pointers are empty if the cosmogenic neutrino flux is not computed!		//
//												//
// 	The memory space for the result functions has to be returned with ncoIPFree by the user //
//												//
// Created: 18/06/2012										//
// Last modified: 19/10/2012									//
//================================================================================================

void ncoCRProp(double zmin,double zmax,nco_crinjspectrum crinj,nco_targetphotons cib_target,nco_ip* result, nco_ip* nue, nco_ip* numu, nco_ip* nutau, double* wcas)
{
  
 	// total target photon spectrum for neutrino production
	double target(double E,double z)
	{
		double res=0.0;
		
		if ((NCO_CRPROP_PHOTOPION_CMB_LOSSES == 1) && (NCO_CRPROP_COSMNEUTRINOS > 0))
			res+=CMBPhotonSpectrum(E,z);
		if (NCO_CRPROP_COSMNEUTRINOS>1)
			res+=cib_target(E,z);

		return res;
	}

	int i;
      
	double x, deltax, from_z, to_z;
	double arrSpecFromZ[NX];
	double arre[NXN];
	double arrmu[NXN];
	double arrtau[NXN];
	double arrx[NX];	
	double arrFlux[NX];
	double arrnx[NXN];
	double logY2F = log10(Y2F);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	double wcasstep; // e.m. cascade energy density (GeV^-1 cm^-3)
	double wcastot = 0.0;
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	nco_ip spec_from_z;
	nco_ip spec_to_z;
	
	nco_ip resulte;
	nco_ip resultmu;
	nco_ip resulttau;
	
	CIBCOUNTER = CIBSTEPS-1;
	CASCADECOUNTER = CASCADESTEPS-1; 
	
	// Initialize the spec_from_z interpolating function
	// The boundary condition at z = zmax is that the spectrum be zero for all values of x 
	
	deltax = (XMAX-XMIN)/((double)NX-1.0);
	x = XMIN;
	for (i=0; i<NX; i++)
	{
		arrx[i] = x;
		if(crinj(x,zmax)>1e-250) arrSpecFromZ[i] = log10(crinj(x,zmax));
		else arrSpecFromZ[i] = -250.0;
		specgrid[i]=0.0; //crinj(x,zmax)*pow(pow(10.0,x),3.0);
		x += deltax;
		
          // global arrays
	  arrPairProductionCIBLoss[i] = 0.0;
	  arrPhotoPionCIBLoss[i] = 0.0;
	}

	ncoIPAllocLinear(&spec_from_z,arrx,arrSpecFromZ,NX);

	if (NCO_CRPROP_COSMNEUTRINOS > 0) 
	{
		deltax = (XMAX-XNMIN)/((double)NXN-1.0);
		x = XNMIN;
		for (i=0; i<NXN; i++)
		{
			arrnx[i] = x;
			arre[i] = 0.0; 
			arrmu[i] = 0.0;
			arrtau[i] = 0.0;
			x += deltax;
		}
	}

	// Start at z = zmax and evolve down to z = zmin
	from_z = zmax;
	to_z = from_z-DELTAZ;
	int STEPCOUNT = 0;
	int mode = 0;
	int CIBSTEPSBEF = CIBSTEPS;
	

	while (to_z >= zmin)
	{
		if (STEPCOUNT % 1000 == 0) printf("from_z = %f\n",from_z);
		STEPCOUNT ++;

		if(to_z>=ZSWITCH) ncoCRPropStep(from_z,to_z,&spec_from_z,&spec_to_z,crinj,cib_target,&wcasstep);
		else 
		{
		 if(mode==0){ DELTAZ=DELTAZ/DELTAZDIV; mode=1; 	CIBSTEPS=CIBSTEPS*DELTAZDIV; CIBCOUNTER= CIBSTEPS-1; }  // Reset CIB Counter to force re-computation of array
		 ncoCRPropStep4(from_z,to_z,&spec_from_z,&spec_to_z,crinj,cib_target,&wcasstep);
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		wcastot += wcasstep;
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		// Calculate the cosmogenic neutrino flux
		if ((NCO_CRPROP_COSMNEUTRINOS > 0) && (STEPCOUNT % NEUTRINOSTEPS == 0) && (to_z>=ZHOM))  // only every NEUTRINOSTEPS step
		{
			//printf("from_z = %f\n",from_z);	  
			ncoComputeCosmoNeutrinoInjection(to_z,zmin,DELTAZ*NEUTRINOSTEPS,&spec_to_z,target,&resulte,&resultmu,&resulttau);

			deltax = (XMAX-XNMIN)/((double)NXN-1.0);
			x = XNMIN;
			for (i=0; i<NXN; i++)
			{
				if ((x>=resulte.min) && (x<=resulte.max)) arre[i]+=pow(10.0,ncoIP(&resulte,x)); // Here the energy range may be shifted compared to standard!
				if ((x>=resultmu.min) && (x<=resultmu.max)) arrmu[i]+=pow(10.0,ncoIP(&resultmu,x));
				if ((x>=resulttau.min) && (x<=resulttau.max)) arrtau[i]+=pow(10.0,ncoIP(&resulttau,x));
				x += deltax;
			}

			ncoIPFree(&resulte);
			ncoIPFree(&resultmu);
			ncoIPFree(&resulttau);
// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // 		  
// 			}
// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // 		  
		}
		
		ncoIPFree(&spec_from_z);
		spec_from_z = spec_to_z;
		from_z = to_z;
		to_z = from_z-DELTAZ; // breaks at z>0 closest to zmin
	}

	if(mode == 1) //JH: This if was missing before, which causes an error when the mode is actually not switched
	{
		DELTAZ=DELTAZ*DELTAZDIV;
		CIBSTEPS=CIBSTEPSBEF;
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// (Log of) Total e.m. cascade energy density at z = zmin
	// Note that this still needs to be multiplied by the normalisation N (~ 10^{39}-10^{45} if fitting to HiRes), which has not been
	// specified within the code; also needs conversion from GeV Mpc^-3 to GeV cm^-3 (multiply by factor ~(3.4*10^74))
	*wcas = log10(wcastot*DELTAZ*(3.4e-74));
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Result for cosmogenic neutrinos
	if (NCO_CRPROP_COSMNEUTRINOS > 0) 
	{
		// log conversion
		for (i=0; i<NXN; i++)
		{
			if (arre[i]>1e-250) arre[i]=log10(arre[i])+logY2F; else arre[i]=-250.0+logY2F; // here also conversion in flux!
			if (arrmu[i]>1e-250) arrmu[i]=log10(arrmu[i])+logY2F; else arrmu[i]=-250.0+logY2F;
			if (arrtau[i]>1e-250) arrtau[i]=log10(arrtau[i])+logY2F; else arrtau[i]=-250.0+logY2F;
		}
	  
		ncoIPAlloc(nue,arrnx,arre,NXN);
		ncoIPAlloc(numu,arrnx,arrmu,NXN);
		ncoIPAlloc(nutau,arrnx,arrtau,NXN);
	}

	// Resulting CR flux at to_z = zmin; conversion in flux
	deltax = (XMAX-XMIN)/((double)NX-1.0);
	x = XMIN;
	
	for (i=0; i<NX; i++)
	{
		arrx[i] = x;
		arrFlux[i] = ncoIP(&spec_to_z,x)+logY2F;
		x += deltax;
	}

	ncoIPAllocLinear(result,arrx,arrFlux,NX);

	ncoIPFree(&spec_to_z);
}


//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//// 											      ////
//// 			INITIALISATION AND MEMORY HANDLING ROUTINES			      ////
//// 											      ////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

//================================================================================================
// ncoCRPropInit										//
//												//
// Initialises cosmic ray propagation code, including the CMB and CIB local spectra if scaling	//
// is used.											//
//												//
// Input:											//
//												//
//	zmin, zmax: lower and upper limits of the redshift range over which the propagation is 	//
//		    performed									//
//												//
// Output:											//
//												//
//	None											//
//												//
// Created: 13/08/2012										//
// Last modified: 10/04/2013									//
//================================================================================================

void ncoCRPropInit(double zmin, double zmax)
{  
	if (NCO_CRPROP_CMB_SCALING == 1)
	{
		int i;
		double x, deltax;
		double XEXTRA = 2.0;

		deltax = (XMAX+XEXTRA-XMIN)/((double)(NX*4)-1.0);  // Step size in the x-direction: here more steps and larger range! (later (1+z)*E accessed!)

		double arrPP[NX*4];
		double arrPhotoPion[NX*4];
		double arrx[NX*4];

		// Calculate the pair-production and photohadronic energy loss rates at z = 0 only once
		x = XMIN;
		for (i=0; i<NX*4; i++)
		{
			arrPP[i] = -bpair(pow(10.0,x),0.0,CMBPhotonSpectrum); // negative, therefore need to store negative values
			arrPhotoPion[i] = -bpion(pow(10.0,x),0.0,CMBPhotonSpectrum); // negative, therefore need to store negative values
    
			if (arrPhotoPion[i]<0) { printf("Photopion negative\n"); exit(-1); }

			// WW: Double log conversion
			if (arrPP[i]>1e-250) arrPP[i]=log10(arrPP[i]);
			else arrPP[i]=-250.0;

			if (arrPhotoPion[i]>1e-250) arrPhotoPion[i]=log10(arrPhotoPion[i]);
			else arrPhotoPion[i]=-250.0;

			arrx[i] = x;
			x += deltax;
		}

		ncoIPAllocLinear(&bpair_ip,arrx,arrPP,NX*4);
		ncoIPAllocLinear(&bpion_ip,arrx,arrPhotoPion,NX*4);
	}

	// Data needed for CIB spectrum; NCO_CRPROP_CIB_MODEL selects what model to use for the local CIB
	if (NCO_CRPROP_CIB_MODEL == 1) ncoIPAllocLinear(&cib_ip,cibrawx1,cibrawy1,ncib1);
	else if (NCO_CRPROP_CIB_MODEL == 2) ncoIPAllocLinear(&cib_ip,cibrawx2,cibrawy2,ncib2);

	if (NCO_CRPROP_SOURCEDIST_CIB < 4) PrecomputeCIBCoeff(zmin,zmax);  // Pre-compute CIB evolution (part of input functions); for this purpose: zmax=10

	// NCO_CRPROP_SOURCEDIST_CIB == 4 selects the tabulated redshift scaling of the CIB by Franceschini et al. (only up to z = 2)
	if (NCO_CRPROP_SOURCEDIST_CIB == 4)
	{
		ncoIPAllocLinear(&cibfrancz00_ip,cibfrancz00x,cibfrancz00y,ncibfranc);
		ncoIPAllocLinear(&cibfrancz02_ip,cibfrancz02x,cibfrancz02y,ncibfranc);
		ncoIPAllocLinear(&cibfrancz04_ip,cibfrancz04x,cibfrancz04y,ncibfranc);
		ncoIPAllocLinear(&cibfrancz06_ip,cibfrancz06x,cibfrancz06y,ncibfranc);
		ncoIPAllocLinear(&cibfrancz08_ip,cibfrancz08x,cibfrancz08y,ncibfranc);
		ncoIPAllocLinear(&cibfrancz10_ip,cibfrancz10x,cibfrancz10y,ncibfranc);
		ncoIPAllocLinear(&cibfrancz12_ip,cibfrancz12x,cibfrancz12y,ncibfranc);
		ncoIPAllocLinear(&cibfrancz14_ip,cibfrancz14x,cibfrancz14y,ncibfranc);
		ncoIPAllocLinear(&cibfrancz16_ip,cibfrancz16x,cibfrancz16y,ncibfranc);
		ncoIPAllocLinear(&cibfrancz18_ip,cibfrancz18x,cibfrancz18y,ncibfranc);
		ncoIPAllocLinear(&cibfrancz20_ip,cibfrancz20x,cibfrancz20y,ncibfranc);
	}



	// NCO_CRPROP_SOURCEDIST_CIB == 6,7,8 selects the CIB spectrum as computed by Inoue et al. (up to z = 10)
	if ((NCO_CRPROP_SOURCEDIST_CIB >= 6) && (NCO_CRPROP_SOURCEDIST_CIB <= 8))
	{
		PrecomputeCIBInoue();
	}

 	// Interpolating function of the neutron critical energy Ec
	ncoIPAllocLinear(&logEc_ip,arrneutronZ,arrneutronlogEc,nEc);

	/* WW: Does not work, file missing in CVS [WW]

	// Read in the data on the two-component CR injection spectrum data from external file which will be used
	// for interpolation
	double crspectrum, z;
	int i, j;
	FILE * fh;
	// The first column contains x; the remaining columns contain the value of the (unnormalised)
	// CR injection spectrum [GeV^-1 Mpc^-3 s^-1]
	// 239 lines (i.e., 239 different values of z)
	// 152 columns per line (column 0 is z value; columns 1-151 are log of spectrum at different values of x)
 	fh=fopen("IC40CRtheoreticalspreadnoheader.proton.injection.alphap.2.0.eta.1.0.n.only.dat","r");
// 	fh=fopen("IC40CRtheoreticalspreadnoheader_quicktest_2.3_acc1_Lisofixed2.dat","r");
	for (i=0;i<NNZZ;i++)
	{
		for(j=0;j<NNXX+1;j++) // Each of the NNXX entries corresponds to a different x=log(E) value, from x = 0 to x = 12 in steps of 0.1
		{
			if (j == 0)
			{
				fscanf(fh,"%lf",&z);
				arrzcrspectrumtwocomp[i] = z;
			}
			else
			{
				fscanf(fh,"%lf",&crspectrum);
				if (crspectrum == 0.0) arrlogcrspectrumtwocomp[i][j-1] = -250.0;
				else
				{
//	 				crspectrum = log10(pow(MpcToCm,3.0)*crspectrum); // convert from GeV^-1 cm^-3 s^-1 to GeV^-1 Mpc^-3 s^-1
					crspectrum = log10(crspectrum);
					if (crspectrum < -250.0) arrlogcrspectrumtwocomp[i][j-1] = -250.0;
					else arrlogcrspectrumtwocomp[i][j-1] = crspectrum;
				}
			}
		}
	}
	fclose(fh);
	for (i=0;i<NNXX;i++)
	{
		arrxcrspectrumtwocomp[i] = 0.1*i;
	}
	// Build the interpolating functions of the (logarithm of) the proton spectrum in the two-component model at different z; this is done only once, here
	for (i=0;i<NNZZ-1;i++)
	{
		ncoIPAllocLinear(&logCRSpectrumTwoCompAtFixedZ_ip[i],arrxcrspectrumtwocomp,arrlogcrspectrumtwocomp[i],NNXX); // CR spectrum at z_i as a function of x
	}
	*/
}

//================================================================================================
// ncoCRPropFree										//
//												//
// Finish cosmic ray propagation code. Free all memory.						//
//												//
// Input:											//
//												//
//	None											//
//												//
// Output:											//
//												//
//	None											//
//												//
// Created: 13/08/2012										//
// Last modified: 13/08/2012									//
//================================================================================================

void ncoCRPropFree()
{
	int i;
	
	if (NCO_CRPROP_CMB_SCALING==1) 
	{
		ncoIPFree(&bpair_ip);
		ncoIPFree(&bpion_ip);
	}

	if (NCO_CRPROP_SOURCEDIST_CIB < 4) ncoIPFree(&cib_ip);

	if (NCO_CRPROP_SOURCEDIST_CIB == 4)
	{
		ncoIPFree(&cibfrancz00_ip);
		ncoIPFree(&cibfrancz02_ip);
		ncoIPFree(&cibfrancz04_ip);
		ncoIPFree(&cibfrancz06_ip);
		ncoIPFree(&cibfrancz08_ip);
		ncoIPFree(&cibfrancz10_ip);
		ncoIPFree(&cibfrancz12_ip);
		ncoIPFree(&cibfrancz14_ip);
		ncoIPFree(&cibfrancz16_ip);
		ncoIPFree(&cibfrancz18_ip);
		ncoIPFree(&cibfrancz20_ip);
	}
	
	ncoIPFree(&logEc_ip);
	
	for (i=0;i<NNZZ-1;i++)
	{
		ncoIPFree(&logCRSpectrumTwoCompAtFixedZ_ip[i]); // CR spectrum at z_i as a function of x
	}
	
	// Free interpolating functions used when the Inoue CIB is selected
	if ((NCO_CRPROP_SOURCEDIST_CIB >= 6) && (NCO_CRPROP_SOURCEDIST_CIB <= 8))
	{
		for (i=0; i<CIBInoueNumberZValues; i++)
		{
			ncoIPFree(&CIBInoue_ip[i]);
		}
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//// 											      ////
//// 	INPUT FUNCTIONS (NOT PART OF CR CODE)						      ////
//// 											      ////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//// 											      ////
//// 			BACKGROUND PROTON/PHOTON SPECTRA				      ////
//// 											      ////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

//================================================================================================
// CRInjectionSpectrumUniversal									//
//												//
// Returns the cosmic ray injection spectrum QCR ~ E^-gamma exp(-E/Epmax).			//
//												//
// Input:											//
//												//
//	x: (log of) proton energy in the comoving frame [GeV]					//
//	z: redshift										//
//												//
// Output:											//
//												//
//	CRInjectionSpectrumUniversal(x,z)							//
//	units: GeV^-1 Mpc^-3 s^-1								//
//												//
// Created: 13/08/2012										//
// Last modified: 13/08/2012									//
//================================================================================================

double CRInjectionSpectrumUniversal(double x, double z)
{
	double eprod,xprod,res,dampfactor,edamp;
	double norm;
	
        eprod=pow(10.0,x); 
	xprod = log10(eprod);
	
	edamp=pow(eprod/INJEPMAX,1.0);  // use single expotential cutoff to match Kotera et al.
	dampfactor=exp(-edamp);
	
	// This is normalised to give 10^44 erg Mpc^-3 yr^-1 for the local CR energy production rate
	if (NCO_CRPROP_ALPHA != 2.0)
	{
		norm = (5.e44*624.15) / (365.0*24.0*60.0*60.0) * (2.0-NCO_CRPROP_ALPHA) / ( pow(10.0,12.0*(2.0-NCO_CRPROP_ALPHA)) - pow(10.0,10.0*(2.0-NCO_CRPROP_ALPHA)) );
	} else
	{
		norm = (1.e44*624.15) / (365.0*24.0*60.0*60.0);
	}
	res = norm * pow(10.0,-NCO_CRPROP_ALPHA*xprod) * CRSourceDistribution(z) * dampfactor;

	// Unnormalised proton injection
	// res = pow(10.0,-NCO_CRPROP_ALPHA*xprod) * CRSourceDistribution(z) * dampfactor;
       
	return res;  
}

//================================================================================================
// CRInjectionSpectrumTwoComponentInterpolation							//
//												//
// Returns the cosmic ray injection spectrum Q_CR calculated in the two-component model. It	//
// reads an external file and interpolates on these data.					//
//												//
// Input:											//
//												//
//	x: (log of) proton energy in the comoving frame [GeV]					//
//	z: redshift										//
//												//
// Output:											//
//												//
//	CRInjectionSpectrumTwoComponentInterpolation(x,z)					//
//	units: GeV^-1 Mpc^-3 s^-1								//
//												//
// Created: 15/11/2012										//
// Last modified: 26/11/2012									//
//================================================================================================

double CRInjectionSpectrumTwoComponentInterpolation(double x, double z)
{
// 	double arrzcrspectrumtwocomp[239];
// 	double arrlogcrspectrumtwocomp[239][151];

	double arrLogCRSpectrumAtX[61];//[239];
	double logCRSpectrumAtXAtZ;

	nco_ip logCRSpectrumTwoCompAtX_ip;
	
	int i;
	
	if (z >= ZHOM)
	{
		for (i=0;i<NNZZ-1;i++)
		{
			arrLogCRSpectrumAtX[i] = ncoIP(&logCRSpectrumTwoCompAtFixedZ_ip[i],x);
		}
		ncoIPAllocLinear(&logCRSpectrumTwoCompAtX_ip,arrzcrspectrumtwocomp,arrLogCRSpectrumAtX,NNZZ); // CR spectrum at x as a function of z
		logCRSpectrumAtXAtZ = ncoIP(&logCRSpectrumTwoCompAtX_ip,z); 
////////////////////////////////////////////////////////////////////////////////////////////////////////
// 		logCRSpectrumAtXAtZ = log10(pow(10.0,ncoIP(&logCRSpectrumTwoCompAtX_ip,1.0))*CRSourceDistribution(z)/CRSourceDistribution(4.0));
////////////////////////////////////////////////////////////////////////////////////////////////////////

		ncoIPFree(&logCRSpectrumTwoCompAtX_ip);
		if (logCRSpectrumAtXAtZ < -250.0) return 1.e-250;
		else return pow(10.0,logCRSpectrumAtXAtZ);
	}
	else return 1.e-250;
}

//================================================================================================
// CIBPhotonSpectrumTwoGaussianInjection							//
//												//
// Returns the number density of CIB photons calculated under the assumption of a two-Gaussian	//
// CIB injection function fitted to the data from the Franceschini et al. model from z = 0 to	//
// z = 2.											//
//												//
// Input:											//
//												//
//	E: photon energy in the comoving frame [GeV]						//
//	z: redshift										//
//												//
// Output:											//
//												//
//	CIBPhotonSpectrumTwoGaussianInjection(E,z)						//
//	units: GeV^-1 cm^-3									//
//												//
// Ref.:											//
//												//
//	A. Franceschini et al., Astron. Astrphys. 487, 837 (2008) [arXiv:0805.1841]		//
//												//
// Created: 12/11/2012										//
// Last modified: 12/11/2012									//
//================================================================================================

double CIBPhotonSpectrumTwoGaussianInjection(double E, double z)
{
	static double sl0 = 1.002;
	static double sh0 = 1.421;
	static double ml0 = -11.155;
	static double mh0 = -9.015;
	static double Nl0 = 120.161;
	static double Nh0 = 148.535;
	static double n = 0.251;
	static double m = 0.412;
	static double p = 0.017;
	static double q = -0.094;
	static double r = 0.100;
	static double s = 0.581;
	static double ZMAXCIB = 8.0; // The parameters where obtained by integrating the CIB injection function up to ZMAXCIB
	
	const double PI = 3.14159265359;
  
	double integral;
	
	if (z > ZMAXCIB)
	{
		printf("CIBPhotonSpectrumTwoGaussianInjection: cannot use z > ZMAXCIB");
		ncoCRPropFree();
		ncoQuit();
		exit(0);
	}
	
	double logCIBPhotonInjectionTwoGaussian(double zz)
	{
		double zzp1 = 1.0+zz;
		double exponentl, gaussianl, exponenth, gaussianh;
		double x = log10(E*(1.0+zz));
		double result;

		exponentl = pow(zzp1,-2.0*n) * pow(x-ml0*pow(zzp1,p),2.0) / (2.0*sl0*sl0);
		gaussianl = exp(-exponentl) * Nl0 * pow(zzp1,-n+r) / (sqrt(2.0*PI)*sl0);
		exponenth = pow(zzp1,-2.0*m) * pow(x-mh0*pow(zzp1,q),2.0) / (2.0*sh0*sh0);
		gaussianh = exp(-exponenth) * Nh0 * pow(zzp1,-m+s) / (sqrt(2.0*PI)*sh0);
		result = gaussianl + gaussianh;
		
		if (result < -250.0) result = -250.0;
// 		if (result > 250.0) result = 250.0;
		return result;
	}
	
	double integrand(double zz)
	{
		return pow(10.0,logCIBPhotonInjectionTwoGaussian(zz))/HubbleParameter(zz);
	}
	
	integral = ncoIntegrate(&integrand,z,ZMAXCIB,21);
	
	return pow(1.0+z,2.0)*integral/pow(MpcToCm,3.0);
}


//================================================================================================
// CIBPhotonSpectrumInoue									//
//												//
// Returns the number density of CIB photons calculated according to the model by Inoue et al.	//
//												//
// Input:											//
//												//
//	E: photon energy in the comoving frame [GeV]						//
//	z: redshift (0 <= z <= 10)								//
//												//
// Output:											//
//												//
//	CIBPhotonSpectrumInoue(E,z)								//
//	units: GeV^-1 cm^-3									//
//												//
// Ref.:											//
//												//
//	Y. Inoue et al. [arXiv:1212.1683]							//
//												//
// Created: 08/04/2013										//
// Last modified: 10/04/2013									//
//================================================================================================

double CIBPhotonSpectrumInoue(double E, double z)
{ 
	int i;
	double x = log10(E);
	double CIBSpectrumFixedX[CIBInoueNumberZValues], result;
	nco_ip CIBInoueFixedXFixedZ_ip;
	
	// Returnn zero if the requested energy is outside the energy range of the data used for interpolation
	if ((x < CIBInoueXValues[0]) || (x > CIBInoueXValues[CIBInoueNumberXValues-1])) return 1.e-250;
	
	// If the requested z value is exactly one of the precomputed interpolating functions CIBInoue_ip,
	// then do not interpolate, just evaluate the corresponding interpolating function

	for (i=0; i<CIBInoueNumberZValues; i++)
	{
		if (fabs(z-CIBInoueZValues[i]) <= 1.e-8)
		{
// 			printf("no interp: %f %f %d\n",z,CIBInoueZValues[i],i);
			return pow(10.0,ncoIP(&CIBInoue_ip[i],x));
		}
	}
	
	// Otherwise, we need to perform a two-dimensional interpolation in redshift and energy (in x, actually).

	// First evaluate all of the CIBInoueNumberZValues different interpolating functions CIBInoue_ip at the 
	// requested value of x. 
	for (i=0; i<CIBInoueNumberZValues; i++)
	{
		CIBSpectrumFixedX[i] = ncoIP(&CIBInoue_ip[i],x);
	}
	
	// Now use the array CIBSpectrumFixedX to build an interpolating function which we will evaluate
	// at the requested value of z.
	ncoIPAllocLinear(&CIBInoueFixedXFixedZ_ip,CIBInoueZValues,CIBSpectrumFixedX,CIBInoueNumberZValues);
	result = pow(10.0,ncoIP(&CIBInoueFixedXFixedZ_ip,z));
	ncoIPFree(&CIBInoueFixedXFixedZ_ip);
	
	return result;
}


//================================================================================================
// CIBPhotonSpectrum										//
//												//
// Returns the redshift-scaled number density of CIB photons,					//
//												//
// Input:											//
//												//
//	E: photon energy in the comoving frame [GeV]						//
//	z: redshift										//
//												//
// Output:											//
//												//
//	CIBPhotonSpectrum(E,z)									//
//	units: GeV^-1 cm^-3									//
//												//
// Ref.:											//
//												//
//	Use here approximation from (see App. C)						//
//	M. Ahlers, L.A. Anchordoqui, and S. Sarkar, Phys. Rev. D 79, 083009 (2009) [0902.3993]	//
//												//
// Created: 13/08/2012										//
// Last modified: 08/04/2013									//
//================================================================================================

double CIBPhotonSpectrum(double E, double z)
{
	double v;
	
	// COMMENT [WW]: for different scaling assumptions or CIB spectra, define different functions here.
	// Implementing the scaling directly (such as for the CMB) does not buy anything in terms of efficiency

	if (NCO_CRPROP_SOURCEDIST_CIB < 4)
	{
		v = 1.0/(1.0+z)*ncoIP(&cib_scale_ip,z)*pow(10.0,ncoIP(&cib_ip,log10(E/(1.0+z)))); // at z=0, from interpolation function
	}
	else if (NCO_CRPROP_SOURCEDIST_CIB == 4) // CIB spectrum extracted from tables of Franceschini et al.
	{
		double arrcibfranc[11];
		double x = log10(E);
		nco_ip cibfranc_ip;
		
		arrcibfranc[0] = ncoIP(&cibfrancz00_ip,x);
		arrcibfranc[1] = ncoIP(&cibfrancz02_ip,x);
		arrcibfranc[2] = ncoIP(&cibfrancz04_ip,x);
		arrcibfranc[3] = ncoIP(&cibfrancz06_ip,x);
		arrcibfranc[4] = ncoIP(&cibfrancz08_ip,x);
		arrcibfranc[5] = ncoIP(&cibfrancz10_ip,x);
		arrcibfranc[6] = ncoIP(&cibfrancz12_ip,x);
		arrcibfranc[7] = ncoIP(&cibfrancz14_ip,x);
		arrcibfranc[8] = ncoIP(&cibfrancz16_ip,x);
		arrcibfranc[9] = ncoIP(&cibfrancz18_ip,x);
		arrcibfranc[10] = ncoIP(&cibfrancz20_ip,x);
		
		ncoIPAllocLinear(&cibfranc_ip,cibfrancz,arrcibfranc,ncibfrancz);
		v = pow(10.0,ncoIP(&cibfranc_ip,z));
		ncoIPFree(&cibfranc_ip);
	}
	else if (NCO_CRPROP_SOURCEDIST_CIB == 5) // CIB spectrum calculated from two-Gaussian model of the CIB injection fitted to Franceschini et al. model
	{
		v = CIBPhotonSpectrumTwoGaussianInjection(E,z);
	}
	else if ((NCO_CRPROP_SOURCEDIST_CIB >= 6) && (NCO_CRPROP_SOURCEDIST_CIB <= 8)) // CIB spectrum calculated for the baseline model of Inoue et al.
	{
		// NCO_CRPROP_SOURCEDIST_CIB = 6: baseline model
		// NCO_CRPROP_SOURCEDIST_CIB = 7: model with lower Pop-III limit
		// NCO_CRPROP_SOURCEDIST_CIB = 8: model with upper Pop-III limit
		v = CIBPhotonSpectrumInoue(E,z);
	}
	
	return v;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//// 											      ////
//// 	MAIN PROGRAM									      ////
//// 											      ////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
	double zmin = 0.0, zmax = 6.0; //6.0;//5.9; // zmax=8.0 as in Kotera et al. does not make a big difference compared to zmax=6.0!
	double wcas;

	nco_ip result;
	nco_ip nue;
	nco_ip numu;
	nco_ip nutau;

	nco_crinjspectrum CRInjSpecTest_ptr = &CRInjectionSpectrumUniversal;
// 	nco_crinjspectrum CRInjSpecTest_ptr = &CRInjectionSpectrumTwoComponentInterpolation;
// 	nco_targetphotons TargetPhotonSpecTest_ptr = &CIBPhotonSpectrumInoue;
	nco_targetphotons TargetPhotonSpecTest_ptr = &CIBPhotonSpectrum;  // here now CIB spectrum!
        
	ncoInit();


	NCO_ODESOLVER_METHOD = 4;   // use these here for this particular code; WW: I think that this is not needed
	NCO_INTEGR_METHOD = NCO_EXPERIMENTAL;
	NCO_PHOTO_IT=NCO_ALL_PION;  // ALWAYS include all pion interaction types/complete cross section for neutrino production

	ncoCRPropInit(zmin,zmax);
	
	// ------------------------------------------------------------	
	// routines to take input from argv
	// ------------------------------------------------------------
	
	int c;
	double optnumber;

	while((c = getopt(argc,argv,"a:e:z:m:")) !=-1)
	switch(c)
		{
		case 'a':
			optnumber = atof(optarg);
			if(optnumber < 4 && optnumber > 0.5)
			NCO_CRPROP_ALPHA = optnumber;
			else
			printf("\n Error: Alpha out of range (%f) \n\n",optarg);
			break;
		case 'e':
			optnumber = atof(optarg);
			if(optnumber < 14 && optnumber > 9)
			INJEPMAX = pow(10,optnumber);
			else
			printf("\n Error: Emax out of range (%f) \n\n",optarg);
			break;
		case 'z':
			optnumber = atof(optarg);
			if(optnumber <= 2*ZHOM && optnumber >= 0)
			ZSWITCH = optnumber;
			else
			printf("\n Error: ZSwitch out of range (%f) \n\n",optarg);
			break;
		case 'm':
			optnumber = atof(optarg);
			if(optnumber <= 20 && optnumber >= -10)
			NCO_CRPROP_COSMEVOL_CR = optnumber;
			else
			printf("\n Error: M out of range out of range (%f) \n\n",optarg);
			break;
		}
	
	// ------------------------------------------------------------
	

	// Input related Options:
	//-----------------------------------------------------------------------------------------------------------------------------------
	// INJEPMAX = 3.16228e11;  // Maximal proton energy [GeV] at source; 3.16228*10^11 GeV = 10^20.5 eV
	// NCO_CRPROP_SOURCEDIST_CR = 1;  // CR source evolution: if 0, no source evolution; if 1, the sources follow the star formation rate; if 2, the GRB rate; if 3, SFR*(1+z)^NCO_CRPROP_COSMEVOL_CR
	// NCO_CRPROP_COSMEVOL_CR = 1.8; // CR source evolution: exponent of the correction ~(1+z)^NCO_CRPROP_COSMEVOL_CR of the GRB comoving rate
	// NCO_CRPROP_ALPHA = 2;  // CR injection index
	// NCO_CRPROP_SOURCEDIST_CIB = 1;  // CIB source evolution: if 0, no source evolution; if 1, the sources follow the star formation rate; if 2, GRB rate; if 3, SFR*(1+z)^NCO_CRPROP_COSMEVOL_CIB
					  // if 4, use the tabulated CIB spectrum density from Franceschini et al.; if 5, use the CIB spectrum calculated with the two-Gaussian model of the CIB 
					  // injection; if 6, use the CIB by Inoue et al. (baseline model); if 7, use Inoue with lower Pop-III limit; if 8, use Inoue with upper Pop-III limit
	// NCO_CRPROP_COSMEVOL_CIB = 1.8; // CIB source evolution: exponent of the correction ~(1+z)^NCO_CRPROP_COSMEVOL_CIB of the GRB comoving rate
	//-----------------------------------------------------------------------------------------------------------------------------------

	double totaltime1 = clock();

	// ---------------------------------------------------------
	// This Block uses the parameters given as options
	// ---------------------------------------------------------

	printf("starting: \n alphap = %1.2f \n EPmax = 10^%2.2f \n Source_Ev_CR = %i \n Source_Ev_CIB = %i \n M = %1.2f \n Zhom = %1.2f \n Zswitch = %1.2f \n\n",
		NCO_CRPROP_ALPHA, log10(INJEPMAX), NCO_CRPROP_SOURCEDIST_CR, NCO_CRPROP_SOURCEDIST_CIB,NCO_CRPROP_COSMEVOL_CR,ZHOM,ZSWITCH);

	double time1=clock(); // WW ADDED
	ncoCRProp(zmin,zmax,CRInjSpecTest_ptr,TargetPhotonSpecTest_ptr,&result,&nue,&numu,&nutau,&wcas);
	double time2=clock(); // WW ADDED
	printf("\n\nTime: %g seconds\n\n",(time2-time1)/CLOCKS_PER_SEC); // WW ADDED

	printf("finished: \n alphap = %1.2f \n EPmax = 10^%2.2f \n Source_Ev_CR = %i \n Source_Ev_CIB = %i \n M = %1.2f \n Zhom = %1.2f \n Zswitch = %1.2f \n\n",
		NCO_CRPROP_ALPHA, log10(INJEPMAX), NCO_CRPROP_SOURCEDIST_CR, NCO_CRPROP_SOURCEDIST_CIB,NCO_CRPROP_COSMEVOL_CR,ZHOM,ZSWITCH);

	// Write here results to file
	char currentfilename[120] = "";
	sprintf(currentfilename,"protonflux_alphap_%1.2f_EPmax_e%2.2f_M_%1.2f.dat",NCO_CRPROP_ALPHA, log10(INJEPMAX), NCO_CRPROP_COSMEVOL_CR);
	ncoIPPrint(&result,currentfilename,NX-1);

	if (NCO_CRPROP_COSMNEUTRINOS > 0)
	{
		sprintf(currentfilename,"neutrinoflux_nue_alphap_%1.2f_EPmax_e%2.2f_M_%1.2f.dat"
			,NCO_CRPROP_ALPHA, log10(INJEPMAX), NCO_CRPROP_COSMEVOL_CR);
		ncoIPPrint(&nue,currentfilename,NXN-1);

		sprintf(currentfilename,"neutrinoflux_numu_alphap_%1.2f_EPmax_e%2.2f_M_%1.2f.dat"
			,NCO_CRPROP_ALPHA, log10(INJEPMAX), NCO_CRPROP_COSMEVOL_CR);	
		ncoIPPrint(&numu,currentfilename,NXN-1);

		sprintf(currentfilename,"neutrinoflux_nutau_alphap_%1.2f_EPmax_e%2.2f_M_%1.2f.dat"
			,NCO_CRPROP_ALPHA, log10(INJEPMAX), NCO_CRPROP_COSMEVOL_CR);
		ncoIPPrint(&nutau,currentfilename,NXN-1);
	}
	// Free memory space from interpolating functions
	ncoIPFree(&result);

	if (NCO_CRPROP_COSMNEUTRINOS > 0)
	{
		ncoIPFree(&nue);
		ncoIPFree(&numu);
		ncoIPFree(&nutau);
	}

	// ---------------------------------------------------------
	// This Block creates files looping over the parameters
	// ---------------------------------------------------------
	
	/*double EmaxCount = 12.5;
	double AlphaCount = 2.5;
	double MCount	= 0.;

	for(AlphaCount = 1.5; AlphaCount < 2.25+0.00001; AlphaCount += 0.05){
		NCO_CRPROP_ALPHA = AlphaCount;
	
	for(EmaxCount = 10.0; EmaxCount < 11.5+0.000001; EmaxCount += 0.1){
		INJEPMAX = pow(10,EmaxCount);
	
	for(MCount = -4; MCount < 6+0.000001; MCount += 0.2){
		NCO_CRPROP_COSMEVOL_CR = MCount;*/

		/*printf("starting: \n alphap = %1.2f \n EPmax = 10^%2.2f \n Source_Ev_CR = %i \n Source_Ev_CIB = %i \n M = %1.2f \n Zhom = %1.2f \n Zswitch = %1.2f \n\n",
			NCO_CRPROP_ALPHA, log10(INJEPMAX), NCO_CRPROP_SOURCEDIST_CR, NCO_CRPROP_SOURCEDIST_CIB,NCO_CRPROP_COSMEVOL_CR,ZHOM,ZSWITCH);

		double time1=clock(); // WW ADDED
		ncoCRProp(zmin,zmax,CRInjSpecTest_ptr,TargetPhotonSpecTest_ptr,&result,&nue,&numu,&nutau,&wcas);
	        double time2=clock(); // WW ADDED
	        printf("\n\nTime: %g seconds\n\n",(time2-time1)/CLOCKS_PER_SEC); // WW ADDED

		printf("finished: \n alphap = %1.2f \n EPmax = 10^%2.2f \n Source_Ev_CR = %i \n Source_Ev_CIB = %i \n M = %1.2f \n Zhom = %1.2f \n Zswitch = %1.2f \n\n",
			NCO_CRPROP_ALPHA, log10(INJEPMAX), NCO_CRPROP_SOURCEDIST_CR, NCO_CRPROP_SOURCEDIST_CIB,NCO_CRPROP_COSMEVOL_CR,ZHOM,ZSWITCH);

	        // Write here results to file
		char currentfilename[120] = "";
		sprintf(currentfilename,"CRPropagation5Output/protonflux_alphap_%1.2f_EPmax_e%2.2f_M_%1.2f.dat",NCO_CRPROP_ALPHA, log10(INJEPMAX), NCO_CRPROP_COSMEVOL_CR);
		ncoIPPrint(&result,currentfilename,NX-1);

		if (NCO_CRPROP_COSMNEUTRINOS > 0)
		{
			sprintf(currentfilename,"CRPropagation5Output/neutrinoflux_nue_alphap_%1.2f_EPmax_e%2.2f_M_%1.2f.dat"
				,NCO_CRPROP_ALPHA, log10(INJEPMAX), NCO_CRPROP_COSMEVOL_CR);
			ncoIPPrint(&nue,currentfilename,NXN-1);

			sprintf(currentfilename,"CRPropagation5Output/neutrinoflux_numu_alphap_%1.2f_EPmax_e%2.2f_M_%1.2f.dat"
				,NCO_CRPROP_ALPHA, log10(INJEPMAX), NCO_CRPROP_COSMEVOL_CR);	
			ncoIPPrint(&numu,currentfilename,NXN-1);

			sprintf(currentfilename,"CRPropagation5Output/neutrinoflux_nutau_alphap_%1.2f_EPmax_e%2.2f_M_%1.2f.dat"
				,NCO_CRPROP_ALPHA, log10(INJEPMAX), NCO_CRPROP_COSMEVOL_CR);
			ncoIPPrint(&nutau,currentfilename,NXN-1);
		}
		// Free memory space from interpolating functions
		ncoIPFree(&result);

		if (NCO_CRPROP_COSMNEUTRINOS > 0)
		{
			ncoIPFree(&nue);
			ncoIPFree(&numu);
			ncoIPFree(&nutau);
		}
	//}}} //close for loops*/

	// ---------------------------------------------------------


	// ---------------------------------------------------------
	// This Block creates files looping over Emax and Zhom
	// ---------------------------------------------------------

        /*double EmaxCount = 12.5;
	double ZhomCount = 0;
	NCO_CRPROP_ALPHA = 2.45;
	NCO_CRPROP_COSMEVOL_CR = 0.;
	
	for(EmaxCount = 13.0; EmaxCount < 15.+0.000001; EmaxCount += 0.1){
	for(ZhomCount = 0.; ZhomCount < 0.02+0.000001; ZhomCount += 0.0005){
		INJEPMAX = pow(10.,EmaxCount);
		ZHOM = ZhomCount;

		printf("starting: \n alphap = %1.2f \n EPmax = 10^%2.2f \n Source_Ev_CR = %i \n Source_Ev_CIB = %i \n M = %1.2f \n Zhom = %1.4f \n Zswitch = %1.2f \n\n",
			NCO_CRPROP_ALPHA, log10(INJEPMAX), NCO_CRPROP_SOURCEDIST_CR, NCO_CRPROP_SOURCEDIST_CIB,NCO_CRPROP_COSMEVOL_CR,ZHOM,ZSWITCH);
		double time1=clock(); // WW ADDED
		ncoCRProp(zmin,zmax,CRInjSpecTest_ptr,TargetPhotonSpecTest_ptr,&result,&nue,&numu,&nutau,&wcas);
        	double time2=clock(); // WW ADDED
        	printf("\n\nTime: %g seconds\n\n",(time2-time1)/CLOCKS_PER_SEC); // WW ADDED

		printf("finished: \n alphap = %1.2f \n EPmax = 10^%2.2f \n Source_Ev_CR = %i \n Source_Ev_CIB = %i \n M = %1.2f \n Zhom = %1.2f \n Zswitch = %1.2f \n\n",
			NCO_CRPROP_ALPHA, log10(INJEPMAX), NCO_CRPROP_SOURCEDIST_CR, NCO_CRPROP_SOURCEDIST_CIB,NCO_CRPROP_COSMEVOL_CR,ZHOM,ZSWITCH);

		// Write here results to file
		char currentfilename[120] = "";
		sprintf(currentfilename,"CRPropagation5Output_ZhomEmax/protonflux_alphap_%1.2f_EPmax_e%2.2f_Zhom_%1.4f.dat",NCO_CRPROP_ALPHA, log10(INJEPMAX), ZHOM);
		ncoIPPrint(&result,currentfilename,NX-1);

		if (NCO_CRPROP_COSMNEUTRINOS > 0)
		{
			sprintf(currentfilename,"CRPropagation5Output_ZhomEmax/neutrinoflux_nue_alphap_%1.2f_EPmax_e%2.2f_Zhom_%1.4f.dat"
				,NCO_CRPROP_ALPHA, log10(INJEPMAX), ZHOM);
			ncoIPPrint(&nue,currentfilename,NXN-1);

			sprintf(currentfilename,"CRPropagation5Output_ZhomEmax/neutrinoflux_numu_alphap_%1.2f_EPmax_e%2.2f_Zhom_%1.4f.dat"
				,NCO_CRPROP_ALPHA, log10(INJEPMAX), ZHOM);	
			ncoIPPrint(&numu,currentfilename,NXN-1);

			sprintf(currentfilename,"CRPropagation5Output_ZhomEmax/neutrinoflux_nutau_alphap_%1.2f_EPmax_e%2.2f_Zhom_%1.4f.dat"
				,NCO_CRPROP_ALPHA, log10(INJEPMAX), ZHOM);
			ncoIPPrint(&nutau,currentfilename,NXN-1);
		}
	
		// Free memory space from interpolating functions
		ncoIPFree(&result);

		if (NCO_CRPROP_COSMNEUTRINOS > 0)
		{
			ncoIPFree(&nue);
			ncoIPFree(&numu);
			ncoIPFree(&nutau);
		}

	}} //close for loops*/

	double totaltime2 = clock();	
	printf("\n\nTotal Time: %g minutes\n\n",(totaltime2-totaltime1)/CLOCKS_PER_SEC/60);	

	ncoCRPropFree();

	ncoQuit();
	exit(0);
}
