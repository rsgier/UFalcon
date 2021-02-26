# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Raphael Sgier and Jörg Herbel

from unittest import mock
import pytest
import numpy as np
from scipy import stats
from astropy.cosmology import FlatLambdaCDM

from UFalcon import lensing_weights


@pytest.fixture()
def cosmo():
    omega_m = 0.3
    H0 = 70
    return FlatLambdaCDM(H0=H0, Om0=omega_m)


def test_dirac(cosmo):
    """
    Test the single-source lensing weights.
    """

    # source redshift above shell
    for z_source, z_low, z_up in [(0.3, 0.1,  0.11),
                                  (0.5, 0.4,  0.41),
                                  (0.7, 0.3,  0.31),
                                  (0.9, 0.8,  0.81),
                                  (1.1, 1.05, 1.06),
                                  (1.3, 1.2,  1.21),
                                  (1.5, 1.4,  1.41)]:

        assert lensing_weights.Dirac(z_source)(z_low, z_up, cosmo) > 0

    # source redshift below shell
    assert lensing_weights.Dirac(0.3)(0.5, 0.6, cosmo) == 0

    # source redshift inside shell
    with pytest.raises(NotImplementedError):
        lensing_weights.Dirac(0.3)(0.2, 0.6, cosmo)

    # check that weight amplitude is sensible (there is more lensing in the middle than at the beginning or the end)
    w_dirac = lensing_weights.Dirac(1)
    #not valid anymore due to new renormalization which is shell thickness dependent
    #assert w_dirac(0.1, 0.11, cosmo) < w_dirac(0.5, 0.51, cosmo)
    #assert w_dirac(0.7, 0.71, cosmo) < w_dirac(0.5, 0.51, cosmo)


def test_continuous_to_dirac(cosmo):
    """
    Test if lensing weights for continuous n(z) converge towards single-source weights for n(z) ~ Dirac-delta.
    """

    for z_source, z_low, z_up in [(0.3, 0.1, 0.11),
                                  (0.8, 0.4, 0.41),
                                  (1.5, 0.3, 0.31),
                                  (1.5, 0.8, 0.81),
                                  (1.5, 1.4, 1.41)]:

        # compute Dirac weight
        w_dirac = lensing_weights.Dirac(z_source)(z_low, z_up, cosmo)

        # compute continuous weight with a tabulated function
        with mock.patch('numpy.genfromtxt') as genfromtxt:
            # tabulated delta function
            genfromtxt.return_value = np.array([[z_source - 0.001, 0.0],
                                                [z_source, 1.0],
                                                [z_source + 0.001, 0.0]])
            # get the weights
            cont_weights = lensing_weights.Continuous(None, z_lim_low=0, z_lim_up=2)
            cont_weights_fast = lensing_weights.Continuous(None, z_lim_low=0, z_lim_up=2, fast_mode=True)

        w_cont = cont_weights(z_low, z_up, cosmo)
        print(w_cont)

        assert (w_dirac - w_cont) / w_cont < 0.01

        # compute continous weights with gaussian delta dist
        n_of_z = stats.norm(loc=z_source, scale=0.005).pdf  # set n(z) interpolator to approximate Dirac
        # get the weights
        cont_weights = lensing_weights.Continuous(n_of_z, z_lim_low=0, z_lim_up=2)
        w_cont = cont_weights(z_low, z_up, cosmo)
        assert (w_dirac - w_cont) / w_cont < 0.01

        n_of_z = stats.norm(loc=z_source, scale=0.5).pdf  # set n(z) interpolator to approximate Dirac
        # get the weights
        cont_weights = lensing_weights.Continuous(n_of_z, z_lim_low=0, z_lim_up=2)
        cont_weights_fast = lensing_weights.Continuous(n_of_z, z_lim_low=0, z_lim_up=2, fast_mode=True)
        w_cont = cont_weights(z_low, z_up, cosmo)
        w_cont_fast = cont_weights_fast(z_low, z_up, cosmo)

        assert (w_cont_fast - w_cont) / w_cont < 0.001


def test_dirac_to_continuous(cosmo):
    """
    Test if lensing weights for a continuous n(z) can be approximated by many single-source weights with source
    redshifts sampled from n(z).
    """

    # define n(z)
    mu = 0.6
    std = 0.1
    nz = stats.norm(loc=mu, scale=std)

    # define redshift interval to test
    z_low = 0.3
    z_up = 0.31

    # compute continuous lensing weight
    cont_weights = lensing_weights.Continuous(nz.pdf, z_lim_low=0, z_lim_up=2)

    w_cont = cont_weights(z_low, z_up, cosmo)

    # sample source redshifts
    zs_source = nz.rvs(size=1000)
    zs_source[(zs_source > z_low) & (zs_source < z_up)] = 0

    # compute single-source weights
    w_dirac = 0

    for i, z_source in enumerate(zs_source):
        w_dirac += lensing_weights.Dirac(z_source)(z_low, z_up, cosmo)

    w_dirac /= zs_source.size

    # compare
    assert (w_dirac - w_cont) / w_cont < 0.01


def test_kappa_prefactor(cosmo):
    """
    Test the computation of the prefactor to convert to convergence.
    """
    n_pix = 17395392
    n_particles = 1024 ** 3
    boxsize = 6
    f = lensing_weights.kappa_prefactor(n_pix, n_particles, boxsize, cosmo)
    assert '{:.18f}'.format(f) == str(0.001595227993431627)


def test_NLA(cosmo):
    # define n(z)
    mu = 0.6
    std = 0.1
    nz = stats.norm(loc=mu, scale=std)

    # define redshift interval to test
    z_low = 0.3
    z_up = 0.31

    # compute continuous lensing weight
    cont_weights = lensing_weights.Continuous(nz.pdf, z_lim_low=0, z_lim_up=2)
    w_cont_no_IA = cont_weights(z_low, z_up, cosmo)

    # check that with zero IA same result is returned
    cont_weights = lensing_weights.Continuous(nz.pdf, z_lim_low=0, z_lim_up=2,
                                              IA=0, eta=10, z_0=5)
    w_cont = cont_weights(z_low, z_up, cosmo)
    assert np.isclose(w_cont_no_IA, w_cont)

    # check that simple IA without redshift dependence works
    cont_weights = lensing_weights.Continuous(nz.pdf, z_lim_low=0, z_lim_up=2,
                                              IA=1, eta=0.0, z_0=0.0)
    w_cont_IA = cont_weights(z_low, z_up, cosmo)

    # check that IA with redshift dependence works
    cont_weights = lensing_weights.Continuous(nz.pdf, z_lim_low=0, z_lim_up=2,
                                              IA=1, eta=1.0, z_0=0.5)
    w_cont_IA = cont_weights(z_low, z_up, cosmo)
