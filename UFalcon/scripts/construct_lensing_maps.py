# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Raphael Sgier and Jörg Herbel

# construct_lensing_maps.py

# This file contains example-functions to compute weak lensing maps using UFalcon

import os
import argparse
import yaml
import numpy as np
import healpy as hp
import h5py
import UFalcon
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const


def get_single_source_redshifts(zs_str):
    """
    Generates an array with discrete redshift steps used for lightcone construction with single-source redshifts
    :param zs_str: string containing redshift range of the lightcone, i.e. 'z_init, z_final, dz'
    :return: numpy array containing redshifts from z_init to z_final with steps of size dz
    """
    z_init, z_final, dz = list(map(float, zs_str.split(',')))
    n_digits_round = int(np.ceil(np.amax(np.fabs(np.log10((z_init, z_final, dz))))))
    zs = np.around(np.arange(z_init, z_final + dz, dz), decimals=n_digits_round)
    return zs


def store_output(kappa_maps, single_source_redshifts, paths_out, combine_nz_maps=False):
    """
    Routine to store the computed kappa and derived gamma maps for single-source redshifts and n(z)-dist.
    :param kappa_maps: file containing the computed kappa-maps, shape ((len(lensing_weights), Npix)
    :param single_source_redshifts: array containing the single-source redshifts
    :param paths_out: path where to store the new files
    :param combine_nz_maps: bool: 'True' if all n(z) maps should be stored in a single file, 'False' otherwise
    :return: stores kappa and corresponding gamma1 and gamma2 maps in a single or separate files
    """

    # mean subtraction
    kappa_maps -= np.mean(kappa_maps, axis=1, keepdims=True)

    n_nz = len(kappa_maps) - len(single_source_redshifts)

    # maps from n(z)
    if n_nz > 0:

        gamma1_maps = []
        gamma2_maps = []

        for i in range(n_nz):

            print('Working on n(z) map {} / {}'.format(i + 1, n_nz), flush=True)

            gamma1, gamma2 = UFalcon.utils.kappa_to_gamma(kappa_maps[i])

            if combine_nz_maps:
                gamma1_maps.append(gamma1)
                gamma2_maps.append(gamma2)

            if not combine_nz_maps:
                hp.write_map(filename=paths_out[i], m=(kappa_maps[i], gamma1, gamma2),
                             fits_IDL=False,
                             coord='C',
                             overwrite=True)
                print('Wrote {}'.format(paths_out[i]))

        if combine_nz_maps:
            print('Storing all n(z) maps into single file')
            nz_maps = np.stack((kappa_maps[:n_nz], gamma1_maps, gamma2_maps), axis=1)
            np.save(paths_out[0], nz_maps)
            print('Wrote {}'.format(paths_out[0]))

    # single-source maps
    if len(single_source_redshifts) > 0:

        kappa_maps_single_source = kappa_maps[n_nz:]

        with h5py.File(paths_out[-1], mode='w') as fh5:

            fh5.create_dataset(name='z', data=single_source_redshifts)
            for name in ('kappa', 'gamma1', 'gamma2'):
                fh5.create_dataset(name=name,
                                   shape=kappa_maps_single_source.shape,
                                   dtype=kappa_maps_single_source.dtype,
                                   compression='lzf')

            for i, kappa_map in enumerate(kappa_maps_single_source):
                print('Storing single-source kappa map {} / {}'.format(i + 1, len(single_source_redshifts)), flush=True)
                gamma1, gamma2 = UFalcon.utils.kappa_to_gamma(kappa_map)
                fh5['kappa'][i] = kappa_map
                fh5['gamma1'][i] = gamma1
                fh5['gamma2'][i] = gamma2


def add_shells_h5(paths_shells, lensing_weighters, nside, boxsize, zs_low, zs_up, cosmo, const, n_particles):
    """
    Computes kappa-maps from precomputed shells in hdf5-format containing particles counts
    :param paths_shells: path to shells
    :param lensing_weighters: lensing weights for continuous n(z)-dist and single-source redshifts, in this order
    :param nside: nside of output maps
    :param boxsize: size of the box in Gigaparsec
    :param zs_low: lower redshift-bound for the lightcone construction
    :param zs_up: upper redshift-bound for the lightcone construction
    :param cosmo: Astropy.Cosmo instance, controls the cosmology used
    :param const: Astropy.Const instance, used for various constants
    :param n_particles: total number of parciles used in the N-Body simulation, i.e. (No. of part. per side)^3
    :return: kappa-maps with shape ((len(lensing_weights), Npix)
    """

    # add up shells
    kappa = np.zeros((len(lensing_weighters), hp.nside2npix(nside)), dtype=np.float32)

    zs_low_arr = np.arange(zs_low['min'], zs_low['max'], zs_low['delta_z'])
    zs_up_arr = np.arange(zs_up['min'], zs_up['max'], zs_up['delta_z'])

    for i_shells, path in enumerate(paths_shells):

        z_low = zs_low_arr[i_shells]
        z_up = zs_up_arr[i_shells]

        print('Processing shells {} / {}, path: {}'.format(i_shells + 1, len(paths_shells), path), flush=True)

        with h5py.File(path, mode='r') as fh5:

            # check if nside is ok
            nside_shells = hp.npix2nside(fh5['shells'].shape[1])
            assert nside <= nside_shells, 'Requested nside ({}) is larger than nside ({}) of input shells in file {}'. \
                format(nside, nside_shells, path)

            # select shells inside redshift range
            z_shells = fh5['z'][...]

            ind_shells = np.where((z_shells[:, 0] >= z_low) & (z_shells[:, 1] <= z_up))[0]

            for c, i_shell in enumerate(ind_shells):

                print('Shell {} / {}'.format(c + 1, len(ind_shells)), flush=True)

                # load shell
                shell = hp.ud_grade(fh5['shells'][i_shell], nside, power=-2).astype(kappa.dtype)
                z_shell_low, z_shell_up = z_shells[i_shell]

                shell *= UFalcon.lensing_weights.kappa_prefactor(n_pix=shell.size,
                                                                 n_particles=n_particles,
                                                                 boxsize=boxsize,
                                                                 cosmo=cosmo)

                # compute lensing weights and add to kappa maps
                for i_w, lensing_weighter in enumerate(lensing_weighters):
                    kappa[i_w] += shell * lensing_weighter(z_shell_low, z_shell_up, cosmo, const)

    return kappa


def add_shells_pkdgrav(dirpath, lensing_weighters_cont, single_source_redshifts, nside, cosmo, const, n_particles, boxsize):
    """
    Computes kappa-maps from precomputed shells in fits-format containing particles counts
    :param dirpath: path to directory containing the fits-files
    :param lensing_weighters_cont: lensing weights for a continuous, user-defined n(z) distribution
    :param single_source_redshifts: array containing single-source redshifts
    :param nside: nside of output maps
    :param cosmo: Astropy.Cosmo instance, controls the cosmology used
    :param const: Astropy.Const instance, used for various constants
    :param n_particles: total number of parciles used in the N-Body simulation, i.e. (No. of part. per side)^3
    :param boxsize: size of the box in Gigaparsec
    :return: kappa-maps with shape ((len(lensing_weights), Npix)
    """

    # get all shells
    file_list = list(filter(lambda fn: 'shell_' in fn and os.path.splitext(fn)[1] == '.fits', os.listdir(dirpath)))

    # extract redshifts
    z_shells_low = []
    z_shells_up = []

    for filename in file_list:
        filename_split = os.path.splitext(filename)[0].split('_')
        z_shells_low.append(float(filename_split[-1]))
        z_shells_up.append(float(filename_split[-2]))

    # adjust single-source redshifts
    z_shells = np.unique(z_shells_low + z_shells_up)

    for i, zs in enumerate(single_source_redshifts):
        single_source_redshifts[i] = z_shells[np.argmin(np.abs(z_shells - zs))]

    print('Adjusted single-source redshifts to: {}'.format(single_source_redshifts))

    lensing_weighters = lensing_weighters_cont + [UFalcon.lensing_weights.Dirac(zs) for zs in single_source_redshifts]

    # add up
    kappa = np.zeros((len(lensing_weighters), hp.nside2npix(nside)), dtype=np.float32)

    for i, filename in enumerate(file_list):

        path = os.path.join(dirpath, filename)
        print('Processing shell {} / {}, path: {}'.format(i + 1, len(file_list), path), flush=True)

        # load shell
        shell = hp.ud_grade(hp.read_map(path), nside, power=-2).astype(kappa.dtype)

        shell *= UFalcon.lensing_weights.kappa_prefactor(n_pix=shell.size,
                                                         n_particles=n_particles,
                                                         boxsize=boxsize,
                                                         cosmo=cosmo)

        # compute lensing weights and add to kappa maps
        for i_w, lensing_weighter in enumerate(lensing_weighters):
            kappa[i_w] += shell * lensing_weighter(z_shells_low[i], z_shells_up[i], cosmo, const)

    return kappa


def main(path_config, paths_shells, nside, paths_nz, single_source_redshifts, paths_out, combine_nz_maps=False):
    """
    main example-function; Constructs lensing maps from precomputed shells containing particle counts
    :param path_config: path to configuration yaml file
    :param paths_shells: path to shells
    :param nside: nside of output maps
    :param paths_nz: path to n(z) files
    :param single_source_redshifts: array containing the single-source redshifts (continuous shear)
    :param paths_out: paths to output files, must contain as many paths as maps are produced
    :param combine_nz_maps: bool: switch to store all n(z) maps into one file
    :return: computes and stores kappa, gamma1, gamma2 maps
    """

    print('Config: {}'.format(path_config))
    print('Shells: {}'.format(paths_shells))
    print('n(z): {}'.format(paths_nz))
    print('Single-source redshifts: {}'.format(single_source_redshifts))

    if combine_nz_maps:
        n_paths_out_nz = int(len(paths_nz) > 0)
    else:
        n_paths_out_nz = len(paths_nz)

    n_paths_out_single_source = int(len(single_source_redshifts) > 0)

    assert len(paths_out) == n_paths_out_nz + n_paths_out_single_source, \
        'Number of output paths does not match number of produced maps, should be {} for n(z) maps and {} for ' \
        'single-source maps'.format(n_paths_out_nz, n_paths_out_single_source)

    # load config
    with open(path_config, mode='r') as f:
        config = yaml.load(f)

    # get cosmo instance

    cosmo_params = config.get('cosmology')
    cosmo = FlatLambdaCDM(H0=cosmo_params['H0'], Om0=cosmo_params['Om0'])

    # get continuous lensing weighters
    try:
        z_lim_low = config['z_low']['min']
        z_lim_up = config['z_up']['max']
    except TypeError:
        z_lim_low = config['z_low']
        z_lim_up = config['z_up']

    lensing_weighters = [UFalcon.lensing_weights.Continuous(path_nz,
                                                            z_lim_low=z_lim_low,
                                                            z_lim_up=z_lim_up) for path_nz in paths_nz]

    # check if shells from h5 file(s) or shells from PKDGRAV
    if len(paths_shells) == 1 and os.path.isdir(paths_shells[0]):
        print('Got one input path which is a directory -- assuming shells stored by PKDGRAV in fits-format')
        kappa = add_shells_pkdgrav(paths_shells[0],
                                   lensing_weighters,
                                   single_source_redshifts,
                                   nside,
                                   cosmo,
                                   const,
                                   config['n_particles'],
                                   config['boxsize'])
    else:
        print('Assuming shells stored in hdf5-format')
        lensing_weighters.extend([UFalcon.lensing_weights.Dirac(zs) for zs in single_source_redshifts])
        kappa = add_shells_h5(paths_shells,
                              lensing_weighters,
                              nside,
                              config['boxsize'],
                              config['z_low'],
                              config['z_up'],
                              cosmo,
                              const,
                              config['n_particles'])

    # store results
    store_output(kappa, single_source_redshifts, paths_out, combine_nz_maps=combine_nz_maps)





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Construct lensing maps',
                                     add_help=True)
    parser.add_argument('--path_config', type=str, required=True, help='configuration yaml file')
    parser.add_argument('--paths_shells', type=str, nargs='+', required=True, help='paths of shells')
    parser.add_argument('--nside', type=int, required=True, help='nside of output maps')
    parser.add_argument('--paths_nz', type=str, nargs='+', default=[], help='paths to n(z) files')
    parser.add_argument('--single_source_redshifts', type=str, nargs='+', default=[], help='single-source redshifts')
    parser.add_argument('--paths_out', type=str, required=True, nargs='+', help='paths to output files, must contain '
                                                                                'as many paths as maps are produced')
    parser.add_argument('--combine_nz_maps', action='store_true', help='switch to store all n(z) maps into one file')
    args = parser.parse_args()

    if len(args.single_source_redshifts) == 1 and ',' in args.single_source_redshifts[0]:
        source_redshifts = get_single_source_redshifts(args.single_source_redshifts[0])
    else:
        source_redshifts = np.array(args.single_source_redshifts, dtype=np.float64)

    main(args.path_config,
         args.paths_shells,
         args.nside,
         args.paths_nz,
         source_redshifts,
         args.paths_out,
         combine_nz_maps=args.combine_nz_maps)
