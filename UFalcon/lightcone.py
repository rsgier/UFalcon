import numpy as np
import healpy as hp
import PyCosmo
import os


from UFalcon import utils

def kappa_map(path2shells, output_name, weights, Np, NSIDE, z_low, z_up, deltaz, cosmo):

    Nsim = Np ** 3
    """no. of particles in the simulation"""

    Npix = hp.nside2npix(NSIDE)
    """no. of pixels in Healpix map"""

    z_arr = np.arange(z_low, z_up, deltaz)
    kappa_temp = np.zeros(Npix)

    for i in range(len(z_arr)):

        if 0.0 <= z_arr < 0.1:
            boxsize = 1
        elif 0.1 <= z_arr < 0.8:
            boxsize = 6
        else:
            boxsize = 9

        factor1 = (((boxsize * 1000.0) ** 3) / Nsim) * (Npix / (4.0 * np.pi))

        a = int(z_arr[i])
        b = str(round(z_arr[i] - int(z_arr[i]), 2))[2:]

        if len(b) == 1:
            z_file = '{}p{}00'.format(a, b)
        else:
            z_file = '{}p{}0'.format(a, b)

        if os.path.exists(path2shells + '/{}_z{}.npy'.format(output_name, z_file)):

            print('Start constructing lightcone for boxsize {}, redshift {}'.format(boxsize, z_arr[i]))

            npart_shell = np.load(path2shells + '/{}_z{}.npy'.format(output_name, z_file))
            npart_shell_c = npart_shell.copy()

            factor2 = (1.0 / utils.comoving_distance(0, (z_arr[i] + z_arr[i] + deltaz) / 2, cosmo) ** 2)

            # Cycle over weights
            kappa_temp += npart_shell_c * factor1 * factor2 * weights[i]
        else:
            pass

    return (3.0 * cosmo.params.omega_m / 2.0) * ((cosmo.params.H0 / cosmo.params.c) ** 3.0) * kappa_temp
