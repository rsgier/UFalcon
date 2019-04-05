import os
import numpy as np
import healpy as hp
from UFalcon import utils


def pos2ang(path, z_low, delta_z, boxsize, cosmo):
    """
    Reads in particle positions stored in a binary file and keeps only those particle within a given redshift shell.
    Returns the kept particle positions as healpix theta- and phi-coordinates.
    :param path: path to binary file holding particle positions
    :param z_low: lower limit of redshift shell
    :param delta_z: thickness of redshift shell
    :param boxsize: size of the box in Gigaparsec
    :param cosmo: PyCosmo.Cosmo instance, controls the cosmology used
    :return: theta- and phi-coordinates of particles inside the shell
    """

    as_float = np.fromfile(path, dtype=np.float32, count=-1)
    as_uint = as_float.view(np.uint32)  # same data, but different interpretation of bit patterns

    block_start = 0
    data_blocks = []

    while block_start < len(as_uint):
        block_size = as_uint[block_start + 1] * 7
        data_block = as_float[block_start + 4: block_start + 4 + block_size].reshape(-1, 7)
        data_blocks.append(data_block)
        block_start = block_start + block_size + 5  # 4 uint32 before the block and 1 afterwards

    data = np.vstack(data_blocks)

    pos_x = data[:, 0] / cosmo.params.h
    pos_y = data[:, 1] / cosmo.params.h
    pos_z = data[:, 2] / cosmo.params.h

    origin = boxsize * 500.0

    r = np.sqrt((pos_x - origin) ** 2 + (pos_y - origin) ** 2 + (pos_z - origin) ** 2)
    min_r = np.amin(r)
    max_r = np.amax(r)

    if (max_r < utils.comoving_distance(z_low, z_low + delta_z, cosmo)) or \
            (min_r > utils.comoving_distance(z_low, z_low + delta_z, cosmo)):
        theta = np.array([], np.float32)
        phi = np.array([], np.float32)

    else:
        shell = np.where(np.logical_and(r > utils.comoving_distance(z_low, z_low + delta_z, cosmo),
                                        r <= utils.comoving_distance(z_low, z_low + delta_z, cosmo)))[0]
        shell_x = pos_x[shell] - origin
        shell_y = pos_y[shell] - origin
        shell_z = pos_z[shell] - origin

        theta = np.pi / 2 - np.arctan2(shell_z, (np.sqrt(shell_x ** 2 + shell_y ** 2)))
        phi = np.pi + np.arctan2(shell_y, shell_x)

    return theta, phi


def n_part(theta, phi, nside):
    """
    Returns a healpix map with side length nside with particle counts according to particle input positions
    (theta, phi).
    Returns an array of length hp.nside2npix(nside) with the number of particles per pixel.
    :param theta: healpix theta-coordinate
    :param phi: healpix phi-coordinate
    :param nside: resolution of the healpix map
    :return: healpix map with number of particles as pixel values
    """
    particle_counts = np.zeros(hp.nside2npix(nside))
    pix_ind = hp.ang2pix(nside, theta, phi, nest=False)
    pix_binned = np.bincount(pix_ind)
    particle_counts[:len(pix_binned)] = pix_binned
    return particle_counts


def npart_map(dirpath, z_low, delta_z, boxsize, cosmo, nside):
    """
    Reads in all files in a given directory and extracts those particles within a given redshift shell.
    :param dirpath: path of directory, assumed to only contain binary files holding particle positions
    :param z_low: lower limit of redshift shell
    :param delta_z: thickness of redshift shell
    :param boxsize: size of the box in Gigaparsec
    :param cosmo: PyCosmo.Cosmo instance, controls the cosmology used
    :param nside: resolution of the healpix map holding the particle counts
    :return: healpix map with particle counts inside the shell
    """

    n_part_total = None
    filelist = list(os.listdir(dirpath))

    for i, filename in enumerate(filelist):

        filepath = os.path.join(dirpath, filename)

        print('extracting particles from file {} / {}, path: {}'.format(i + 1, len(filelist), filepath))

        theta, phi = pos2ang(filepath, z_low, delta_z, boxsize, cosmo)

        if n_part_total is None:
            n_part_total = n_part(theta, phi, nside)
        else:
            n_part_total += n_part(theta, phi, nside)

    print('shell construction is finished for z={}'.format(z_low))

    return n_part_total
