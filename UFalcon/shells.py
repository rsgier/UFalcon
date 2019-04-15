import os
import numpy as np
import healpy as hp
from UFalcon import utils


def read_lpicola(path, h, boxsize):
    """
    Reads in a binary data file produced by L-Picola.
    :param path: path to file
    :param h: dimensionless Hubble parameter
    :param boxsize: size of the box in Gigaparsec
    :return: 3-tuple containing (x, y, z) particle positions
    """

    # read in binary data
    as_float = np.fromfile(path, dtype=np.float32, count=-1)
    as_uint = as_float.view(np.uint32)  # same data, but different interpretation of bit patterns

    block_start = 0
    data_blocks = []

    while block_start < len(as_uint):
        block_size = as_uint[block_start + 1] * 7
        data_block = as_float[block_start + 4: block_start + 4 + block_size].reshape(-1, 7)
        data_blocks.append(data_block)
        block_start = block_start + block_size + 5  # 4 uint32 before the block and 1 afterwards

    data = np.vstack(data_blocks)[:, :3]

    # transform to Mpc and subtract origin
    origin = boxsize * 500.0
    data /= h
    data -= origin

    return data.T


def read_pkdgrav(path, boxsize, n_rows_per_block=int(1e6)):
    """
    Reads in a binary data file produced by PKDGRAV.
    :param path: path to file
    :param boxsize: size of the box in Gigaparsec
    :param n_rows_per_block: number of rows to read in one block, allows to limit memory consumption for large files
    :return: 3-tuple containing (x, y, z) particle positions
    """

    # get the total number of rows
    n_rows = os.stat(path).st_size // 7 // 4

    # initialize output
    data = np.empty((n_rows, 3), dtype=np.float32)

    # read in blocks
    n_block = int(7 * n_rows_per_block)
    n_rows_in = 0

    with open(path, mode='rb') as f:
        while True:
            block = np.fromfile(f, dtype=np.float32, count=n_block).reshape(-1, 7)[:, :3]

            if block.size == 0:
                break

            data[n_rows_in: n_rows_in + block.shape[0]] = block
            n_rows_in += block.shape[0]

    # transforms to Mpc
    data *= boxsize * 1000

    return data.T


def pos2ang(path, z_low, delta_z, boxsize, cosmo, file_format='l-picola'):
    """
    Reads in particle positions stored in a binary file and keeps only those particle within a given redshift shell.
    Returns the kept particle positions as healpix theta- and phi-coordinates.
    :param path: path to binary file holding particle positions
    :param z_low: lower limit of redshift shell
    :param delta_z: thickness of redshift shell
    :param boxsize: size of the box in Gigaparsec
    :param cosmo: PyCosmo.Cosmo instance, controls the cosmology used
    :param file_format: data format, either l-picola or pkdgrav
    :return: theta- and phi-coordinates of particles inside the shell
    """

    if file_format == 'l-picola':
        pos_x, pos_y, pos_z = read_lpicola(path, cosmo.params.h, boxsize)
    elif file_format == 'pkdgrav':
        pos_x, pos_y, pos_z = read_pkdgrav(path, boxsize)
    else:
        raise ValueError('Data format {} is not supported, choose either "l-picola" or "pkdgrav"')

    # compute comoving radius
    r = np.sqrt(pos_x ** 2 + pos_y ** 2 + pos_z ** 2)

    # select particles inside shell
    select_shell = (r > utils.comoving_distance(0.0, z_low, cosmo)) & \
                   (r <= utils.comoving_distance(0.0, z_low + delta_z, cosmo))

    shell_x = pos_x[select_shell]
    shell_y = pos_y[select_shell]
    shell_z = pos_z[select_shell]

    # convert to angles
    theta = np.pi / 2 - np.arctan2(shell_z, (np.sqrt(shell_x ** 2 + shell_y ** 2)))
    phi = np.pi + np.arctan2(shell_y, shell_x)

    return theta, phi


def ang2map(theta, phi, nside):
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


def construct_shell(dirpath, z_low, delta_z, boxsize, cosmo, nside, file_format='l-picola', verbose=False):
    """
    Reads in all files in a given directory and extracts those particles within a given redshift shell.
    :param dirpath: path of directory, assumed to only contain binary files holding particle positions
    :param z_low: lower limit of redshift shell
    :param delta_z: thickness of redshift shell
    :param boxsize: size of the box in Gigaparsec
    :param cosmo: PyCosmo.Cosmo instance, controls the cosmology used
    :param nside: resolution of the healpix map holding the particle counts
    :param file_format: data format, either l-picola or pkdgrav
    :param verbose: whether to information for each file read in
    :return: healpix map with particle counts inside the shell
    """

    n_part_total = None

    if file_format == 'l-picola':
        filelist = list(filter(lambda fn: os.path.splitext(fn)[1] != '.info', os.listdir(dirpath)))
    elif file_format == 'pkdgrav':
        filelist = list(filter(lambda fn: os.path.splitext(fn)[1] == '.out', os.listdir(dirpath)))
    else:
        raise ValueError('Data format {} is not supported, choose either "l-picola" or "pkdgrav"')

    for i, filename in enumerate(filelist):

        filepath = os.path.join(dirpath, filename)

        if verbose:
            print('extracting particles from file {} / {}, path: {}'.format(i + 1, len(filelist), filepath))

        theta, phi = pos2ang(filepath, z_low, delta_z, boxsize, cosmo, file_format=file_format)

        if n_part_total is None:
            n_part_total = ang2map(theta, phi, nside)
        else:
            n_part_total += ang2map(theta, phi, nside)

    if verbose:
        print('shell construction is finished for z={}'.format(z_low))

    return n_part_total
