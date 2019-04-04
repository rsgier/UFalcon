import pytest
import mock
import numpy as np
from scipy import stats
import PyCosmo

from UFalcon import lensing_weights


@pytest.fixture()
def cosmo():
    return PyCosmo.Cosmo()


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
    assert w_dirac(0.1, 0.11, cosmo) < w_dirac(0.5, 0.51, cosmo)
    assert w_dirac(0.7, 0.71, cosmo) < w_dirac(0.5, 0.51, cosmo)


def test_continuous_to_dirac(cosmo):
    """
    Test if lensing weights for continuous n(z) converge towards single-source weights for n(z) ~ dirac-delta
    """

    for z_source, z_low, z_up in [(0.3, 0.1, 0.11),
                                  (0.8, 0.4, 0.41),
                                  (1.5, 0.3, 0.31),
                                  (1.5, 0.8, 0.81),
                                  (1.5, 1.4, 1.41)]:

        # compute Dirac weight
        w_dirac = lensing_weights.Dirac(z_source)(z_low, z_up, cosmo)

        # compute continuous weight
        with mock.patch('numpy.genfromtxt') as genfromtxt:
            genfromtxt.return_value = np.zeros((2, 2))  # something which has the correct shape
            cont_weights = lensing_weights.Continuous(None, z_lim_low=0, z_lim_up=2)

        cont_weights.nz_intpt = stats.norm(loc=z_source, scale=0.005).pdf  # set n(z) interpolator to approximate Dirac
        cont_weights.nz_norm = 1

        w_cont = cont_weights(z_low, z_up, cosmo)

        assert (w_dirac - w_cont) / w_cont < 0.01