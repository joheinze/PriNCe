"""
 Reads in particle data from two sources:
  - for Nuclides used the NeuCosmA files, which are data compiled from Mathematica built in functions
  - for elementary particles uses the ParticleDataTool (https://github.com/afedynitch/ParticleDataTool)
    which sources from PYTHIA data
 we use the NeuCosmA particle ids in the final dict
"""
import pickle
import sys
from os import path

import numpy as np
from scipy.constants import c as c_mps

fpath = path.dirname(path.abspath(__file__))

from particletools.tables import PYTHIAParticleData


# Part 1 NeuCosmA files
def parse_line(line):
    u_in_gev = 0.9314940954
    data = {}

    # data
    nco_id = int(line[0])
    mother_id = nco_id
    data['mass'] = u_in_gev * line[1]  # mass converted to GeV
    data['stable'] = bool(line[2])
    data['lifetime'] = np.inf if data['stable'] else line[3]
    data['incomplete'] = bool(line[8])
    data['charge'] = nco_id % 100

    #decay channel
    branching = line[4]
    if branching == 0.:
        data['branchings'] = []
    else:
        daughters = []
        beta_mode = int(line[5])
        if beta_mode == -1:
            # electron emitted
            mother_id += 1
            daughters += [12, 20]
        elif beta_mode == 1:
            # positron emitted
            mother_id -= 1
            daughters += [11, 21]
        # also take into account emitted baryons
        no_daughters = int(line[6])
        daughter_id = int(line[7])

        mother_id -= no_daughters * daughter_id
        daughters += no_daughters * [daughter_id]
        daughters += [mother_id]

        data['branchings'] = [(branching, daughters)]

    return nco_id, data


def gen_prince_db():
    """Generates from particle databases of Neucosma and
    MCEq a dictionary for quick lookup in PriNCe."""

    data_raw = np.loadtxt(
        path.join(fpath, '160302_BETA_MATHEMATICA.dat'), dtype=float)

    nuclide_data = {}

    for line in data_raw:
        nco_id, data = parse_line(line)
        if nco_id not in nuclide_data:
            nuclide_data[nco_id] = data
        else:
            nuclide_data[nco_id]['branchings'] += data['branchings']
            nuclide_data[nco_id]['branchings'].sort(reverse=True)

    # Part 2 ParticleDataTool
    pa_data = PYTHIAParticleData()

    id_mapping = [
        (0, 22),  # photon
        (2, 211),  # pi_plus
        (3, -211),  # pi_minus
        (4, 111),  # pi_zero
        (5, -13),  # mu_plus_l
        (6, -13),  # mu_plus_r
        (7, -13),  # mu_plus (both)
        (8, 13),  # mu_minus_l
        (9, 13),  # mu_minus_r
        (10, 13),  # mu_minus (both)
        (11, 12),  # nu_e
        (12, -12),  # nu_bar_e
        (13, 14),  # nu_mu
        (14, -14),  # nu_bar_mu
        (15, 16),  # nu_tau
        (16, -16),  # nu_bar_tau
        (20, 11),  # electron
        (21, -11),  # positron
        (50, 321),  # k_plus
        (51, -321),  # k_minus
        (100, 2112),  # neutron
        (101, 2212),  # proton
    ]
    id_mapping_dict = {}
    for nco_id, pdg_id in id_mapping:
        id_mapping_dict[pdg_id] = nco_id

    # Use scipy constants.
    c = 1e2 * c_mps  # convert to centimeters

    particle_data = {}
    # Add an entry containing the ids of the neucosma
    # particles which should be included by default incomputations
    particle_data["non_nuclear_species"] = sorted(id_mapping_dict.values())

    for nco_id, pdg_id in id_mapping:
        p = pa_data[pdg_id]
        branchings = pa_data.decay_channels(pdg_id)

        data_nco = {}
        data_nco['mass'] = p.mass
        data_nco['stable'] = True if p.ctau/c == np.inf else False
        data_nco['lifetime'] = p.ctau/c
        data_nco['incomplete'] = False
        data_nco['name'] = p.name
        data_nco['charge'] = p.charge

        # convert also ids in branching
        branchings_nco = []
        for ratio, daughter_ids in branchings:
            # hotfix bug in particletools
            for ida in range(len(daughter_ids)):
                if daughter_ids[ida] == -111:
                    daughter_ids[ida] = 111
            daughter_ids = [
                id_mapping_dict[daughter] for daughter in daughter_ids
            ]
            branchings_nco += [(ratio, daughter_ids)]
        branchings_nco.sort(
            reverse=
            True)  # sort them to start with the hightest branching ratio
        data_nco['branchings'] = branchings_nco

        particle_data[nco_id] = data_nco

    # Part 3 combine the data and save everything
    all_data = nuclide_data.copy()
    all_data.update(particle_data)

    file_path = path.join(fpath, '../data/particle_data.ppo')
    pickle.dump(all_data, open(file_path, 'wb'), protocol=-1)


if __name__ == "__main__":
    gen_prince_db()
