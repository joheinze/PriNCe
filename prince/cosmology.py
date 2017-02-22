'''
Created on Feb 22, 2017

@author: Anatoli Fedynitch
'''

from prince_config import config
import numpy as np

H0 = config['H_0s']
Omega_m = config['Omega_m']
Omega_Lambda = config['Omega_Lambda']

def H(z):
    """Expansion rate of the universe.

    :math:`H(z) = H_0 \\sqrt{\\Omega_m (1 + z)^3 + \\Omega_\\Lambda}`
    Args:
      z (float): redshift

    Returns:
      float: expansion rate in :math:`s^{-1}`
    """
    
    return H0*np.sqrt(Omega_m*(1 + z)**3 + Omega_Lambda)