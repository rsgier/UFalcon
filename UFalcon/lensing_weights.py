import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d

from UFalcon import utils


class Continuous:
    """
    Computes the lensing weights for a continuous, user-defined n(z) distribution.
    """

    def __init__(self, path_nz, z_lim_low=0, z_lim_up=None, shift_nz=0.0, IA=0.0):
        """
        Constructor.
        :param path_nz: path to file containing n(z), assumed to be a text file readable with numpy.genfromtext with
                        the first column containing z and the second column containing n(z).
        :param z_lim_low: lower integration limit to use for n(z) normalization, default: 0
        :param z_lim_up: upper integration limit to use for n(z) normalization, default: last z-coordinate in n(z) file
        :param shift_nz: Can shift the n(z) function by some redshift (intended for easier implementation of photo z bias)
        :param IA: Intrinsic Alignment. If unequal 0 computes the lensing weights for IA component
                        (needs to be added to the weights without IA afterwards)
        """

        nz = np.genfromtxt(path_nz)

        if z_lim_up is None:
            z_lim_up = nz[-1, 0]

        self.z_lim_up = z_lim_up
        self.z_lim_low = z_lim_low
        self.nz_intpt = interp1d(nz[:, 0] - shift_nz, nz[:, 1], bounds_error=False, fill_value=0.0)
        self.nz_norm = integrate.quad(lambda x: self.nz_intpt(x), z_lim_low, self.z_lim_up)[0]
        self.IA = IA
        self.lightcone_points = nz[np.logical_and(z_lim_low < nz[:,0], nz[:,0] < z_lim_up),0]

    def __call__(self, z_low, z_up, cosmo):
        """
        Computes the lensing weights for the redshift interval [z_low, z_up].
        :param z_low: lower end of the redshift interval
        :param z_up: upper end of the redshift interval
        :param cosmo: PyCosmo.Cosmo instance, controls the cosmology used
        :return: lensing weight
        """
        norm = utils.dimensionless_comoving_distance(z_low, z_up, cosmo) * self.nz_norm
        norm *= (utils.dimensionless_comoving_distance(0., (z_low + z_up)/2., cosmo) ** 2.)

        if np.isclose(self.IA, 0.0):
            # lensing weights without IA
            numerator = integrate.dblquad(self._integrand,
                                          z_low,
                                          z_up,
                                          lambda x: x,
                                          lambda x: self.z_lim_up,
                                          args=(cosmo,))[0]
        else:
            # lengsing weights for IA
            numerator = (2.0/(3.0*cosmo.params.omega_m)) * \
                        (cosmo.params.c/cosmo.params.H0) * \
                        w_IA(self.IA, z_low, z_up, cosmo, self.nz_intpt, self.lightcone_points, self.z_lim_low, self.z_lim_up)

        return numerator / norm

    def _integrand(self, y, x, cosmo):
        return self.nz_intpt(y) * \
               utils.dimensionless_comoving_distance(0, x, cosmo) * \
               utils.dimensionless_comoving_distance(x, y, cosmo) * \
               (1 + x) * \
               utils.one_over_e(x, cosmo) / \
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
            norm *= (utils.dimensionless_comoving_distance(0., (z_low + z_up)/2., cosmo) ** 2.)

            w = numerator / norm

        return w

    def _integrand(self, x, cosmo):
        return utils.dimensionless_comoving_distance(0, x, cosmo) * \
               utils.dimensionless_comoving_distance(x, self.z_source, cosmo) * \
               (1 + x) * \
               utils.one_over_e(x, cosmo)


def kappa_prefactor(n_pix, n_particles, boxsize, cosmo):
    """
    Computes the prefactor to transform from number of particles to convergence, see https://arxiv.org/abs/0807.3651,
    eq. (A.1).
    :param n_pix: number of healpix pixels used
    :param n_particles: number of particles
    :param boxsize: size of the box in Gigaparsec
    :param cosmo: PyCosmo.Cosmo instance, controls the cosmology used
    :return: convergence prefactor
    """
    convergence_factor = (3.0 * cosmo.params.omega_m / 2.0) * \
                         (n_pix / (4.0 * np.pi)) * \
                         (cosmo.params.H0 / cosmo.params.c) ** 3 * \
                         (boxsize * 1000.0) ** 3 / n_particles
    return convergence_factor


def F_NIA_model(z, IA, cosmo):
    """
    Calculates the NIA kernel used to calculate the IA shell weight
    :param z: Redshift where to evaluate
    :param IA: Galaxy intrinsic alignments amplitude
    :param cosmo: PyCosmo.Cosmo instance, controls the cosmology used
    :return: NIA kernel at redshift z
    """
    growth = lambda a: 1.0 / (a**3.0 * (cosmo.params.omega_m * a**-3.0 + (1.0 - cosmo.params.omega_m))**1.5)
    a = 1.0 / (1.0 + z)
    g = 5.0 * cosmo.params.omega_m / 2.0 * np.sqrt(cosmo.params.omega_m * a**-3.0 + (1.0 - cosmo.params.omega_m)) * integrate.quad(growth, 0, a)[0]

    # Calculate the growth factor today
    g_norm = 5.0 * cosmo.params.omega_m / 2.0 * integrate.quad(growth, 0, 1)[0]

    # divide out a
    g = g / g_norm

    #made to cancel with units
    G = 4.301e-9

    # critical density today
    rho_c = 3 * (cosmo.params.H0)**2 / (8 * np.pi * G)

    # Proportionality constant Msun^-1 Mpc^3
    C1 = 5e-14 / (cosmo.params.H0 / 100.0)**2

    return -IA * rho_c * C1 * cosmo.params.omega_m / g


def w_IA(IA, z_low, z_up, cosmo, nz_intpt, points, z_lower_bound, z_upper_bound):
    """
    Calculates the weight per slice for the NIA model given a 
    distribution of source redshifts n(z).
    :param IA: Galaxy Intrinsic alignment amplitude
    :param z_low: Lower redshift limit of the shell
    :param z_up: Upper redshift limit of the shell
    :param cosmo: PyCosmo.Cosmo instance, controls the cosmology used
    :param nz_intpt: nz function 
    :param points: Points in redshift where integrad is evaluated 
    :param z_lower_bound: Absolute lower bound for reshift
    :param z_upper_bound: Absolute upper bound for reshift
    :return: Shell weight for NIA model
    """
    def f(x, IA, cosmo, nz_intpt):
        return cosmo.params.H0 / cosmo.params.c * (F_NIA_model(x, IA, cosmo) * nz_intpt(x))

    points = points[np.logical_and(z_lower_bound < points, points < z_upper_bound)]
    dbl = integrate.quad(f, z_low, z_up, args=(IA, cosmo, nz_intpt), points=points[np.logical_and(z_low < points, points < z_up)])[0]

    return dbl
