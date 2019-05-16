import argparse
import os
import yaml
import numpy as np
import healpy as hp
import h5py
import UFalcon
import PyCosmo


def get_single_source_redshifts(zs_str):
    z_init, z_final, dz = list(map(float, zs_str.split(',')))
    n_digits_round = int(np.ceil(np.amax(np.fabs(np.log10((z_init, z_final, dz))))))
    zs = np.around(np.arange(z_init, z_final + dz, dz), decimals=n_digits_round)
    return zs


def kappa_to_gamma(kappa_map, lmax=None):

    nside = hp.npix2nside(kappa_map.size)

    if lmax is None:
        lmax = 3 * nside - 1

    kappa_alm = hp.map2alm(kappa_map, lmax=lmax)
    l = hp.Alm.getlm(lmax)[0]

    # Add the appropriate factor to the kappa_alm
    fac = np.where(np.logical_and(l != 1, l != 0), np.sqrt(((l + 2.0) * (l - 1))/((l + 1) * l)), 0)
    kappa_alm *= fac
    t, q, u = hp.alm2map([np.zeros_like(kappa_alm), kappa_alm, np.zeros_like(kappa_alm)], nside=nside)

    return q, u


def store_output(kappa_maps, paths_nz, single_source_redshifts, dirpath_out):

    nside = hp.npix2nside(kappa_maps.shape[1])

    # maps from n(z)
    for i, path_nz in enumerate(paths_nz):
        print('Storing kappa map from n(z) {} / {}'.format(i + 1, len(paths_nz)), flush=True)
        name_nz = os.path.splitext(os.path.split(path_nz)[1])[0]
        path_out = os.path.join(dirpath_out, 'lensing_maps.nside{}.{}.fits'.format(nside, name_nz))
        gamma1, gamma2 = kappa_to_gamma(kappa_maps[i])
        hp.write_map(filename=path_out, m=(kappa_maps[i], gamma1, gamma2), fits_IDL=False, coord='C', overwrite=True)

    # single-source maps
    filename_out = 'lensing_maps.nside{}.z_source_{}-{}.n_sources_{}.h5'.format(nside,
                                                                                *single_source_redshifts[[0, -1]],
                                                                                single_source_redshifts.size)
    path_out = os.path.join(dirpath_out, filename_out)

    with h5py.File(path_out, mode='w') as fh5:

        fh5.create_dataset(name='z', data=single_source_redshifts)
        for name in ('kappa', 'gamma1', 'gamma2'):
            fh5.create_dataset(name=name, shape=kappa_maps.shape, dtype=kappa_maps.dtype, compression='lzf')

        for i, kappa_map in enumerate(kappa_maps[len(paths_nz):]):
            print('Storing single-source kappa map {} / {}'.format(i + 1, len(single_source_redshifts)), flush=True)
            gamma1, gamma2 = kappa_to_gamma(kappa_map)
            fh5['kappa'][i] = kappa_map
            fh5['gamma1'][i] = gamma1
            fh5['gamma2'][i] = gamma2


def main(path_config, paths_shells, nside, paths_nz, single_source_redshifts, dirpath_out):

    print('Config: {}'.format(path_config))
    print('n(z): {}'.format(paths_nz))
    print('Single-source redshifts: {}'.format(single_source_redshifts))

    # load config
    with open(path_config, mode='r') as f:
        config = yaml.load(f)

    # get cosmo instance
    cosmo = PyCosmo.Cosmo(config.get('pycosmo_config'))
    cosmo.set(**config.get('cosmology', dict()))

    # get lensing weighters
    lensing_weighters = [UFalcon.lensing_weights.Continuous(path_nz,
                                                            z_lim_low=min(config['shells']['z_low']),
                                                            z_lim_up=max(config['shells']['z_up'])) for path_nz in paths_nz]
    lensing_weighters.extend([UFalcon.lensing_weights.Dirac(zs) for zs in single_source_redshifts])

    # add up shells
    kappa = np.zeros((len(lensing_weighters), hp.nside2npix(nside)), dtype=np.float32)

    for i_shells, path in enumerate(paths_shells):

        boxsize = config['shells']['boxsizes'][i_shells]
        z_low = config['shells']['z_low'][i_shells]
        z_up = config['shells']['z_up'][i_shells]

        print('Processing shells {} / {}, path: {}'.format(i_shells + 1, len(paths_shells), path), flush=True)

        with h5py.File(path, mode='r') as fh5:

            # check if nside is ok
            nside_shells = hp.npix2nside(fh5['shells'].shape[1])
            assert nside <= nside_shells, 'Requested nside ({}) is larger than nside ({}) of input shells in file {}'.\
                format(nside, nside_shells, path)

            # select shells inside redshift range
            z_shells = fh5['z'][...]

            ind_shells = np.where((z_shells[:, 0] >= z_low) & (z_shells[:, 1] <= z_up))[0]

            for c, i_shell in enumerate(ind_shells):

                print('Shell {} / {}'.format(c + 1, len(ind_shells)), flush=True)

                # load shell
                shell = hp.ud_grade(fh5['shells'][i_shell], nside, power=-2).astype(kappa.dtype)
                z_shell_low, z_shell_up = z_shells[i_shell]

                # divide by dimensionless comoving distance squared and apply prefactor
                shell /= UFalcon.utils.dimensionless_comoving_distance(0, (z_shell_low + z_shell_up) / 2, cosmo) ** 2
                shell *= UFalcon.lensing_weights.kappa_prefactor(n_pix=shell.size,
                                                                 n_particles=config['n_particles'],
                                                                 boxsize=boxsize,
                                                                 cosmo=cosmo)

                # compute lensing weights and add to kappa maps
                for i_w, lensing_weighter in enumerate(lensing_weighters):
                    kappa[i_w] += shell * lensing_weighter(z_shell_low, z_shell_up, cosmo)

    # store results
    store_output(kappa, paths_nz, single_source_redshifts, dirpath_out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Construct shells containing particle counts from N-Bodys',
                                     add_help=True)
    parser.add_argument('--path_config', type=str, required=True, help='configuration yaml file')
    parser.add_argument('--paths_shells', type=str, required=True, help='paths of shells')
    parser.add_argument('--nside', type=int, required=True, help='nside of output maps')
    parser.add_argument('--paths_nz', type=str, help='paths to n(z) files')
    parser.add_argument('--single_source_redshifts', type=str, help='single-source redshifts')
    parser.add_argument('--dirpath_out', type=str, required=True, help='path where maps will be stored')
    args = parser.parse_args()

    paths_shells = args.paths_nz.split(',')

    if args.paths_nz is not None:
        paths_nz = args.paths_shells.split(',')
    else:
        paths_nz = []

    if args.single_source_redshifts is not None:
        single_source_redshifts = get_single_source_redshifts(args.single_source_redshifts)
    else:
        single_source_redshifts = []

    main(args.path_config, paths_shells, args.nside, paths_nz, single_source_redshifts, args.dirpath_out)
