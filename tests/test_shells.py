import numpy as np
import healpy as hp
import mock
import PyCosmo
from UFalcon import shells
from UFalcon.utils import comoving_distance


def test_pos2ang():
    """

    :return:
    """

    n = 40
    data = np.random.random((n, 7)).astype(np.float32)
    cosmo = PyCosmo.Cosmo()

    np.random.seed(10)
    pos_x = np.random.rand(n) * 2000.0
    pos_y = np.random.rand(n) * 2000.0
    pos_z = np.random.rand(n) * 2000.0

    boxsize = 2.0
    z_low = 0.105
    delta_z = 0.09

    origin = boxsize * 500.0

    r = np.sqrt((pos_x - origin) ** 2 + (pos_y - origin) ** 2 + (pos_z - origin) ** 2)
    min_r = np.amin(r)
    max_r = np.amax(r)

    if (max_r < comoving_distance(0.0, z_low, cosmo)) or \
            (min_r > comoving_distance(0.0, z_low + delta_z, cosmo)):
        theta = np.array([], np.float32)
        phi = np.array([], np.float32)

    else:
        shell = np.where(np.logical_and(r > comoving_distance(0.0, z_low, cosmo),
                                        r <= comoving_distance(0.0, z_low + delta_z, cosmo)))[0]
        shell_x = pos_x[shell] - origin
        shell_y = pos_y[shell] - origin
        shell_z = pos_z[shell] - origin

        theta = np.pi / 2 - np.arctan2(shell_z, (np.sqrt(shell_x ** 2 + shell_y ** 2)))
        phi = np.pi + np.arctan2(shell_y, shell_x)

    data[:, 0] = pos_x
    data[:, 1] =  pos_y
    data[:, 2] = pos_z

    block = np.zeros(4 + 7 * n, dtype=np.float32)
    block_int = block.view(np.uint32)
    block_int[:] = 0

    block_int[1] = n
    block[4:] = data.reshape(-1, 1).flatten()

    with mock.patch('numpy.fromfile') as fromfile:
        fromfile.return_value = block

        shells.pos2ang(None, 0.105, 0.09, 2.0, cosmo)




def create_random_map(nside):

    npix = hp.nside2npix(nside)
    counts_map = np.zeros(npix)
    pix_ind = np.random.choice(npix, size=npix, replace=True)

    for i in pix_ind:
        counts_map[i] += 1

    return pix_ind, counts_map


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

    def pos2ang_side_effect(*args):
        ind = int(args[0])
        return hp.pix2ang(nside, pix_inds[ind])

    # run function
    with mock.patch('os.listdir') as listdir:
        listdir.return_value = list(map(str, range(len(pix_inds))))  # as many "filenames" as we have maps

        with mock.patch('UFalcon.shells.pos2ang', side_effect=pos2ang_side_effect):
            assert np.array_equal(shells.construct_shell('', None, None, None, None, nside), counts_map)
