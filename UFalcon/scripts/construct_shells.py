# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Raphael Sgier and Jörg Herbel

# construct_shells.py

# This file contains example-functions to compute shells (maps with particle counts) using UFalcon

import argparse
import numpy as np
import yaml
import h5py
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const
import UFalcon


def get_redshifts(z_init, z_final, delta_z):
    """
    Generates array containing discrete redshift-steps
    :param z_init: start redshift
    :param z_final: end redshift
    :param delta_z: redshift interval between steps
    :return: array with redshift-steps between z_init and z_final with delta_z-sized steps
    """
    to_check = np.array([z_init, z_final, delta_z])
    to_check = to_check[to_check != 0]
    n_digits_round = int(np.ceil(np.amax(np.fabs(np.log10(to_check)))))
    z = np.around(np.arange(z_init, z_final + delta_z, delta_z), decimals=n_digits_round)
    return z


def main(path_config, dirpath_in, sim_type, boxsize, nside, path_out):
    """
    Computes and stores maps containing the particle counts (shells) from N-Body simulation output
    :param path_config: path to configuration yaml file
    :param dirpath_in: path to directory with N-Body simulation output
    :param sim_type: type of used N-Body simulation: 'l-picola' or 'pkdgrav' (up to now)
    :param boxsize: size of the box in Gigaparsec
    :param nside: nside of output maps
    :param path_out: path where shells will be stored
    :return: Computes and stores maps containing the particle counts
    """

    print('Config: {}'.format(path_config))
    print('Input directory: {}'.format(dirpath_in))
    print('Output directory: {}'.format(path_out))

    # load config
    with open(path_config, mode='r') as f:
        config = yaml.load(f)

    # get redshifts
    z = get_redshifts(config['z_init'], config['z_final'], config['delta_z'])

    # get cosmo instance
    cosmo_params = config.get('cosmology')
    cosmo = FlatLambdaCDM(H0=cosmo_params['H0'], Om0=cosmo_params['Om0'])

    # compute shells
    shells = UFalcon.shells.construct_shells(dirpath=dirpath_in,
                                             z_shells=z,
                                             boxsize=boxsize,
                                             cosmo=cosmo,
                                             const=const,
                                             nside=nside,
                                             file_format=sim_type)

    # store ouput
    with h5py.File(path_out, mode='w') as fh5:
        fh5.create_dataset(name='z', data=np.stack((z[:-1], z[1:]), axis=-1))
        fh5.create_dataset(name='shells', data=shells, compression='lzf')

    print('Wrote {}'.format(path_out))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Construct shells containing particle counts from N-Bodys',
                                     add_help=True)
    parser.add_argument('--path_config', type=str, required=True, help='configuration yaml file')
    parser.add_argument('--dirpath_in', type=str, required=True, help='path to simulation output')
    parser.add_argument('--sim_type', type=str, required=True, choices=('l-picola', 'pkdgrav'),
                        help='type of simulation')
    parser.add_argument('--boxsize', type=float, required=True, help='boxsize in Gpc')
    parser.add_argument('--nside', type=int, required=True, help='nside of output shells')
    parser.add_argument('--path_out', type=str, required=True, help='path where shells will be stored')
    args = parser.parse_args()

    main(args.path_config, args.dirpath_in, args.sim_type, args.boxsize, args.nside, args.path_out)
