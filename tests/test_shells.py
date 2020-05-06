# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Raphael Sgier and Jörg Herbel

import os
from unittest import mock
import numpy as np
import healpy as hp
import pytest
import PyCosmo
from UFalcon import utils, shells


def test_read_lpicola():
    """
    Tests the reading of a binary file produced by L-PICOLA.
    """

    n_particles = 40
    boxsize = 2.0
    h = 0.7
    path = 'test.out'

    # create test data
    data = np.random.rand(n_particles, 7).astype(np.float32) * boxsize * 1000

    # transform to binary format
    block = np.zeros(4 + 7 * n_particles, dtype=np.float32)
    block_int = block.view(np.uint32)
    block_int[:] = 0
    block_int[1] = n_particles
    block[4:] = data.reshape(-1, 1).flatten()
    block.tofile(path)

    # test
    xyz = shells.read_lpicola(path, h, boxsize)
    assert np.array_equal(xyz, data[:, :3] / h - boxsize * 500)

    # remove test file
    os.remove(path)


def test_read_pkdgrav():
    """
    Tests the reading of a binary file produced by PKDGRAV3.
    """

    n_particles = 40
    boxsize = 2.0
    path = 'test.out'

    # create test data and write to disk
    data = np.random.rand(n_particles, 7).astype(np.float32)
    data.tofile(path)

    # test
    data *= boxsize * 1000

    xyz = shells.read_pkdgrav(path, boxsize, n_rows_per_block=15)
    assert np.array_equal(xyz, data[:, :3])

    xyz = shells.read_pkdgrav(path, boxsize, n_rows_per_block=100)
    assert np.array_equal(xyz, data[:, :3])

    # test empty file
    np.ones((0, 7), dtype=np.float32).tofile(path)
    xyz = shells.read_pkdgrav(path, boxsize)
    assert xyz.size == 0

    # remove test file
    os.remove(path)


def test_read_file():
    """
    Tests the reading of a binary file produced by either L-PICOLA or PKDGRAV3.
    """

    with mock.patch('UFalcon.shells.read_lpicola') as read_lpicola:
        read_lpicola.return_value = 1
        assert shells.read_file(None, None, PyCosmo.Cosmo(), file_format='l-picola') == 1

    with mock.patch('UFalcon.shells.read_pkdgrav') as read_pkdgrav:
        read_pkdgrav.return_value = 1
        assert shells.read_file(None, None, None, file_format='pkdgrav') == 1

    with pytest.raises(ValueError):
        shells.read_file(None, None, None, file_format='wrong')


def test_xyz_to_spherical():
    """
    Tests the conversion from cartesian to spherical coordinates.
    """
    # sample random positions
    x_y_z = np.random.uniform(low=-1, high=1, size=(10, 3))
    # transform to spherical coordinates
    r_theta_phi = shells.xyz_to_spherical(x_y_z)
    # transform back
    x_y_z_out = hp.ang2vec(r_theta_phi[:, 1], r_theta_phi[:, 2]) * r_theta_phi[:, :1]
    # compare
    assert np.allclose(x_y_z_out, x_y_z)


def test_thetaphi_to_pixelcounts():
    """
    Test the conversion from angular coordinates to number counts in healpix pixels.
    """
    # create random healpix map
    nside = 512
    pix_ind = np.random.choice(hp.nside2npix(nside), size=hp.nside2npix(nside))
    m_in = np.zeros(hp.nside2npix(nside))

    for i in pix_ind:
        m_in[i] += 1

    # transform to angles
    theta, phi = hp.pix2ang(nside, pix_ind)

    # call function
    counts = shells.thetaphi_to_pixelcounts(theta, phi, nside)
    m_out = np.zeros_like(m_in)
    m_out[:counts.size] = counts

    # compare
    assert np.array_equal(m_in, m_out)


def test_construct_shells():
    """
    Test the construction of a shells from N-Body output files.
    """

    nside = 512
    z_shells = [0, 0.1, 0.5, 1]
    n_particles_per_shell = 100
    cosmo = PyCosmo.Cosmo()

    # compute comoving distances to the edges of the shells
    comoving_distances_shells = [utils.comoving_distance(0, z, cosmo) for z in z_shells]

    # create random sets of positions located inside one shell each
    pos = np.random.uniform(low=-1, high=1, size=(len(z_shells) - 1, n_particles_per_shell, 3))

    for i_shell in range(pos.shape[0]):
        f = np.random.uniform(low=comoving_distances_shells[i_shell],
                              high=comoving_distances_shells[i_shell + 1],
                              size=n_particles_per_shell) ** 2 / np.sum(pos[i_shell] ** 2, axis=-1)
        pos[i_shell] *= np.sqrt(f).reshape(f.size, 1)

    # shuffle positions around
    pos_randomized = pos.copy().reshape(-1, 3)
    pos_randomized = pos_randomized[np.random.permutation(pos_randomized.shape[0])]
    pos_randomized = pos_randomized.reshape(pos.shape)

    # run function
    def read_file_side_effect(*args, **kwargs):
        ind = int(args[0][0])
        return pos_randomized[ind]

    with mock.patch('os.listdir') as listdir:
        listdir.return_value = list(map(lambda x: str(x) + '_lightcone.', range(pos.shape[0])))  # as many "filenames" as we have sets of particle positions

        with mock.patch('UFalcon.shells.read_file', side_effect=read_file_side_effect):
            particle_shells = shells.construct_shells('', z_shells, None, cosmo, nside)

    # check results
    assert particle_shells.shape[0] == pos.shape[0]

    for i_shell in range(pos.shape[0]):
        m = np.zeros(hp.nside2npix(nside))
        r_theta_phi = shells.xyz_to_spherical(pos[i_shell])
        pix_ind = hp.ang2pix(nside, r_theta_phi[:, 1], r_theta_phi[:, 2])

        for i in pix_ind:
            m[i] += 1

        assert np.array_equal(m, particle_shells[i_shell])

    # test error raising
    with pytest.raises(ValueError):
        shells.construct_shells(None, None, None, None, None, file_format='wrong')
