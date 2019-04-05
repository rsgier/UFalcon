import numpy as np
import healpy as hp
import mock
from UFalcon import shells


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
