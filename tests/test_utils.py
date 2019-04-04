import pytest
import numpy as np
from scipy import integrate
import PyCosmo
from UFalcon import utils


@pytest.fixture()
def cosmo():
    return PyCosmo.Cosmo()


def test_e(cosmo):
    """
    Test the function E(z).
    """
    omega_m = 0.3
    omega_l = 0.7
    cosmo.set(omega_m=omega_m, omega_l_in=omega_l)

    for z in [0, 0.5, 1]:
        assert utils.e(z, cosmo) == np.sqrt(omega_m * (1 + z) ** 3 + omega_l)


def test_comoving_distance(cosmo):
    """
    Test the computation of the comoving distance.
    """

    z_low = np.arange(0, 1, 0.1)
    z_up = z_low + np.random.uniform(low=0, high=0.2, size=z_low.size)

    dimless_com_utils = utils.dimensionless_comoving_distance(z_low, z_up, cosmo)
    com_utils = utils.comoving_distance(z_low, z_up, cosmo)

    # check values
    for i, (zl, zu) in enumerate(zip(z_low, z_up)):
        dimless_com_direct = integrate.quad(lambda z: 1 / utils.e(z, cosmo), zl, zu)[0]
        com_direct = dimless_com_direct * cosmo.params.c / cosmo.params.H0
        assert (dimless_com_utils[i] - dimless_com_direct) / dimless_com_direct < 0.01
        assert (com_utils[i] - com_direct) / com_direct < 0.01

    # check that scalar input results in scalar output
    with pytest.raises(TypeError):
        len(utils.dimensionless_comoving_distance(0, 1, cosmo))
