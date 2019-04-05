import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d

from UFalcon import utils


class Continuous:
    """
    Computes the lensing weights for a continuous, user-defined n(z) distribution.
    """

    def __init__(self, path_nz, z_lim_low=0, z_lim_up=None):
        """
        Constructor.
        :param path_nz: path to file containing n(z), assumed to be a text file readable with numpy.genfromtext with
                        the first column containing z and the second column containing n(z).
        :param z_lim_low: lower integration limit to use for n(z) normalization, default: 0
        :param z_lim_up: upper integration limit to use for n(z) normalization, default: last z-coordinate in n(z) file
        """

        nz = np.genfromtxt(path_nz)

        if z_lim_up is None:
            z_lim_up = nz[-1, 0]

        self.z_lim_up = z_lim_up
        self.nz_intpt = interp1d(nz[:, 0], nz[:, 1], bounds_error=False, fill_value=0.0)
        self.nz_norm = integrate.quad(lambda x: self.nz_intpt(x), z_lim_low, self.z_lim_up)[0]

    def __call__(self, z_low, z_up, cosmo):
        """
        Computes the lensing weights for the redshift interval [z_low, z_up].
        :param z_low: lower end of the redshift interval
        :param z_up: upper end of the redshift interval
        :param cosmo: PyCosmo.Cosmo instance, controls the cosmology used
        :return: lensing weight
        """
        numerator = integrate.dblquad(self._integrand,
                                      z_low,
                                      z_up,
                                      lambda x: x,
                                      lambda x: self.z_lim_up,
                                      args=(cosmo,))[0]
        norm = utils.dimensionless_comoving_distance(z_low, z_up, cosmo) * self.nz_norm
        return numerator / norm

    def _integrand(self, y, x, cosmo):
        return self.nz_intpt(y) * \
               utils.dimensionless_comoving_distance(0, x, cosmo) * \
               utils.dimensionless_comoving_distance(x, y, cosmo) * \
               (1 + x) / \
               utils.e(x, cosmo) / \
               utils.dimensionless_comoving_distance(0, y, cosmo)


class Dirac:
    """
    Computes the lensing weights for a single-source redshift.
    """

    def __init__(self, z_source):
        """
        Constructor
        :param z_source: source redshift
        """
        self.z_source = z_source

    def __call__(self, z_low, z_up, cosmo):
        """
        Computes the lensing weights for the redshift interval [z_low, z_up].
        :param z_low: lower end of the redshift interval
        :param z_up: upper end of the redshift interval
        :param cosmo: PyCosmo.Cosmo instance, controls the cosmology used
        :return: lensing weight
        """

        # source is below the shell --> zero weight
        if self.z_source <= z_low:
            w = 0

        # source is inside the shell --> error
        elif self.z_source < z_up:
            raise NotImplementedError('Attempting to call UFalcon.lensing_weights.Dirac with z_low < z_source < z_up, '
                                      'this is not implemented')

        # source is above the shell --> usual weight
        else:

            numerator = integrate.quad(self._integrand,
                                       z_low,
                                       z_up,
                                       args=(cosmo,))[0]

            norm = utils.dimensionless_comoving_distance(z_low, z_up, cosmo) * \
                   utils.dimensionless_comoving_distance(0, self.z_source, cosmo)

            w = numerator / norm

        return w

    def _integrand(self, x, cosmo):
        return utils.dimensionless_comoving_distance(0, x, cosmo) * \
               utils.dimensionless_comoving_distance(x, self.z_source, cosmo) * \
               (1 + x) / \
               utils.e(x, cosmo)


def kappa_prefactor(n_pix, n_particles, boxsize, cosmo):
    """
    Computes the prefactor to transform from number of particles to convergence, see https://arxiv.org/abs/0807.3651,
    eq. (A.1).
    :param n_pix: number of healpix pixels used
    :param n_particles: number of particles
    :param boxsize: size of the box in Gigaparsec
    :param cosmo: PyCosmo.Cosmo instance, controls the cosmology used
    :return: 
    """
    convergence_factor = (3.0 * cosmo.params.omega_m / 2.0) * \
                         (n_pix / (4.0 * np.pi)) * \
                         (cosmo.params.H0 / cosmo.params.c) ** 3 * \
                         (boxsize * 1000.0) ** 3 / n_particles
    return convergence_factor
