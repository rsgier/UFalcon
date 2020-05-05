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
    :return: 3-tuple containing (x, y, z) particle positions in Megaparsec
    """

    n_rows_total = 0

    with open(path, mode='rb') as fh:

        # first get the total number of blocks, such that we can pre-allocate memory
        while True:
            header = np.fromfile(fh, dtype=np.uint32, count=4)

            if header.size == 0:
                break

            n_rows_current = header[1]
            n_rows_total += n_rows_current
            block_size = n_rows_current * 7
            fh.seek(block_size * 4 + 4, 1)   # data + endmarker

        # pre-allocate
        data = np.empty((n_rows_total, 3), dtype=np.float32)
        n_rows_read = 0

        # read out data
        fh.seek(0)
        while True:
            header = np.fromfile(fh, dtype=np.uint32, count=4)

            if header.size == 0:
                break

            n_rows_current = header[1]
            block_size = n_rows_current * 7
            data[n_rows_read: n_rows_read + n_rows_current] = np.fromfile(fh,
                                                                          dtype=np.float32,
                                                                          count=block_size).reshape(-1, 7)[:, :3]
            n_rows_read += n_rows_current

            fh.seek(4, 1)  # skip end marker

    # transform to Mpc and subtract origin
    origin = boxsize * 500.0
    data /= h
    data -= origin

    return data


def read_pkdgrav(path, boxsize, n_rows_per_block=int(1e6)):
    """
    Reads in a binary data file produced by PKDGRAV.
    :param path: path to file
    :param boxsize: size of the box in Gigaparsec
    :param n_rows_per_block: number of rows to read in one block, allows to limit memory consumption for large files
    :return: 3-tuple containing (x, y, z) particle positions in Megaparsec
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

    return data


def read_file(path, boxsize, cosmo, file_format='pkdgrav'):
    """
    Reads in particle positions stored in a binary file produced by either L-PICOLA or PKDGRAV.
    :param path: path to binary file holding particle positions
    :param boxsize: size of the box in Gigaparsec
    :param cosmo: PyCosmo.Cosmo instance, controls the cosmology used
    :param file_format: data format, either l-picola or pkdgrav
    :return: theta- and phi-coordinates of particles inside the shell
    """

    if file_format == 'l-picola':
        xyz = read_lpicola(path, cosmo.params.h, boxsize)
    elif file_format == 'pkdgrav':
        xyz = read_pkdgrav(path, boxsize)
    else:
        raise ValueError('Data format {} is not supported, choose either "l-picola" or "pkdgrav"')

    return xyz


def xyz_to_spherical(xyz_coord):
    """
    Transform from comoving cartesian (x, y, z)- to spherical coordinates (comoving radius, healpix theta, healpix phi).
    :param xyz_coord: cartesian coordinates, shape: (number of particles, 3)
    :return: comoving radius, theta, phi
    """

    x = xyz_coord[:, 0]
    y = xyz_coord[:, 1]
    z = xyz_coord[:, 2]
    spherical_coord = np.empty_like(xyz_coord)

    # comoving radius
    spherical_coord[:, 0] = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    # theta, phi
    spherical_coord[:, 1], spherical_coord[:, 2] = hp.vec2ang(xyz_coord)

    return spherical_coord


def thetaphi_to_pixelcounts(theta, phi, nside):
    """
    Transforms angular particle positions to counts in healpix pixels. The size of the output array equals the index of
    the last non-empty pixel (i.e. the largest healpix index with at least one count).
    :param theta: healpix theta-coordinate
    :param phi: healpix phi-coordinate
    :param nside: nside of the healpix map
    :return: counts in each pixel, maximum size: nside - 1
    """
    pix_ind = hp.ang2pix(nside, theta, phi, nest=False)
    counts = np.bincount(pix_ind)
    return counts


def construct_shells(dirpath, z_shells, boxsize, cosmo, nside, file_format='l-picola'):

    # find all files to process
    if file_format == 'l-picola':
        filelist = list(filter(lambda fn: '_lightcone.' in fn and os.path.splitext(fn)[1] != '.info',
                               os.listdir(dirpath)))
    elif file_format == 'pkdgrav':
        filelist = list(filter(lambda fn: '.lcp.' in fn, os.listdir(dirpath)))
    else:
        raise ValueError('Data format {} is not supported, choose either "l-picola" or "pkdgrav"')

    print('Will process {} files'.format(len(filelist)))

    # initialize shells
    shells = np.zeros((len(z_shells) - 1, hp.nside2npix(nside)), dtype=np.int32)

    # compute comoving distances of the shell boundaries
    com_shells = [utils.comoving_distance(0, z, cosmo) for z in z_shells]

    print('Processing file ', end='', flush=True)

    for i, filename in enumerate(filelist):

        print('{} '.format(i + 1), end='', flush=True)

        filepath = os.path.join(dirpath, filename)

        # read out cartesian coordinates
        coord = read_file(filepath, boxsize, cosmo, file_format=file_format)

        # transform to spherical coordinates
        coord[:] = xyz_to_spherical(coord)

        # sort by comoving radius
        coord[:] = coord[np.argsort(coord[:, 0])]

        # sort particles into shells
        ind_shells = np.searchsorted(coord[:, 0], com_shells, side='left')

        for i_shell in range(shells.shape[0]):
            i_low = ind_shells[i_shell]
            i_up = ind_shells[i_shell + 1]
            counts_shell = thetaphi_to_pixelcounts(coord[i_low: i_up, 1], coord[i_low: i_up, 2], nside)
            shells[i_shell, :counts_shell.size] += counts_shell

    return shells
