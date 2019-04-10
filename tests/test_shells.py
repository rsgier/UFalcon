import os
import numpy as np
import healpy as hp
import mock
import pytest
import PyCosmo
from UFalcon import shells
from UFalcon.utils import comoving_distance


def create_random_map(nside):

    npix = hp.nside2npix(nside)
    counts_map = np.zeros(npix)
    pix_ind = np.random.choice(npix, size=npix, replace=True)

    for i in pix_ind:
        counts_map[i] += 1

    return pix_ind, counts_map


def test_read_lpicola():

    n_particles = 40
    boxsize = 2.0
    h = 0.7

    # create test data
    data = np.random.rand(n_particles, 7).astype(np.float32) * boxsize * 1000

    # transform to binary format
    block = np.zeros(4 + 7 * n_particles, dtype=np.float32)
    block_int = block.view(np.uint32)
    block_int[:] = 0
    block_int[1] = n_particles
    block[4:] = data.reshape(-1, 1).flatten()

    # test
    with mock.patch('numpy.fromfile') as fromfile:
        fromfile.return_value = block

        x, y, z = shells.read_lpicola(None, h, boxsize)

    assert np.array_equal(x, (data[:, 0] / h) - boxsize * 500)
    assert np.array_equal(y, (data[:, 1] / h) - boxsize * 500)
    assert np.array_equal(z, (data[:, 2] / h) - boxsize * 500)


def test_read_pkdgrav():

    n_particles = 40
    boxsize = 2.0
    path = 'test.out'

    # create test data and write to disk
    data = np.random.rand(n_particles, 7).astype(np.float32)
    data.tofile(path)

    # test
    data *= boxsize * 1000

    x, y, z = shells.read_pkdgrav(path, boxsize, n_rows_per_block=15)
    assert np.array_equal(x, data[:, 0])
    assert np.array_equal(y, data[:, 1])
    assert np.array_equal(z, data[:, 2])

    x, y, z = shells.read_pkdgrav(path, boxsize, n_rows_per_block=100)
    assert np.array_equal(x, data[:, 0])
    assert np.array_equal(y, data[:, 1])
    assert np.array_equal(z, data[:, 2])

    # test empty file
    np.ones((0, 7), dtype=np.float32).tofile(path)
    x, y, z = shells.read_pkdgrav(path, boxsize)
    assert x.size == 0
    assert y.size == 0
    assert z.size == 0

    # remove test file
    os.remove(path)


def test_pos2ang():
    """
    Test the reading of a binary file and the selection of the particles inside a given redshift shell.
    """

    n_particles = 40
    boxsize = 2.0
    origin = boxsize * 500.0
    z_low = 0.105
    delta_z = 0.09
    cosmo = PyCosmo.Cosmo()

    # create test data
    np.random.seed(10)
    data = np.random.rand(n_particles, 3).astype(np.float32) * boxsize * 1000 - origin

    # find particles inside shell
    theta_in = []
    phi_in = []
    com_low = comoving_distance(0, z_low, cosmo)
    com_up = comoving_distance(0, z_low + delta_z, cosmo)

    for ip in range(n_particles):

        x = data[ip, 0]
        y = data[ip, 1]
        z = data[ip, 2]

        if com_low < np.sqrt(x ** 2 + y ** 2 + z ** 2) <= com_up:
            theta_in.append(np.pi / 2 - np.arctan2(z, (np.sqrt(x ** 2 + y ** 2))))
            phi_in.append(np.pi + np.arctan2(y, x))

    # test
    with mock.patch('UFalcon.shells.read_lpicola') as read_lpicola:
        read_lpicola.return_value = data[:, 0], data[:, 1], data[:, 2]

        theta_out, phi_out = shells.pos2ang(None, z_low, delta_z, boxsize, cosmo)

        assert np.allclose(theta_in, theta_out)
        assert np.allclose(phi_in, phi_out)

    # test pathological case where there are no particles inside the shell
    with mock.patch('UFalcon.shells.read_lpicola') as read_lpicola:
        read_lpicola.return_value = data[:, 0] * 0, data[:, 1] * 0, data[:, 2] * 0

        theta_out, phi_out = shells.pos2ang(None, z_low, delta_z, boxsize, cosmo)
        assert theta_out.size == 0
        assert phi_out.size == 0

    # test error raising
    with pytest.raises(ValueError):
        shells.pos2ang(None, z_low, delta_z, boxsize, cosmo, file_format='wrong')


def test_ang2map():
    """
    Test the conversion from healpix angular coordinates to number counts.
    """

    nside = 512

    # create map and draw random pixel indices
    pix_ind, counts = create_random_map(nside)

    # test
    assert np.array_equal(counts, shells.ang2map(*hp.pix2ang(nside, pix_ind), nside))


def test_construct_shell():
    """
    Test the construction of a shell from individual L-Picola output files.
    """

    nside = 512

    # create 3 random maps
    pix_inds = []
    counts_map = 0

    for _ in range(3):
        pix_ind, counts = create_random_map(nside)
        pix_inds.append(pix_ind)
        counts_map += counts

    def pos2ang_side_effect(*args, **kwargs):
        ind = int(args[0])
        return hp.pix2ang(nside, pix_inds[ind])

    # run function
    with mock.patch('os.listdir') as listdir:
        listdir.return_value = list(map(str, range(len(pix_inds))))  # as many "filenames" as we have maps

        with mock.patch('UFalcon.shells.pos2ang', side_effect=pos2ang_side_effect):
            assert np.array_equal(shells.construct_shell('', None, None, None, None, nside), counts_map)

    # test error raising
    with pytest.raises(ValueError):
        shells.construct_shell('', None, None, None, None, nside, file_format='wrong')
