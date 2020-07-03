# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Raphael Sgier and Jörg Herbel

import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d

from UFalcon import utils, constants




class Continuous:
    """
    Computes the lensing weights for a continuous, user-defined n(z) distribution.
    """

    def __init__(self, n_of_z, z_lim_low=0, z_lim_up=None, shift_nz=0.0, IA=0.0):
        """
        Constructor.
        :param n_of_z: either path to file containing n(z), assumed to be a text file readable with numpy.genfromtext
                        with the first column containing z and the second column containing n(z), or a callable that
                        is directly a redshift distribution
        :param z_lim_low: lower integration limit to use for n(z) normalization, default: 0
        :param z_lim_up: upper integration limit to use for n(z) normalization, default: last z-coordinate in n(z) file
        :param shift_nz: Can shift the n(z) function by some redshift (intended for easier implementation of photo z bias)
        :param IA: Intrinsic Alignment. If not None computes the lensing weights for IA component
                        (needs to be added to the weights without IA afterwards)
        """

        # we handle the redshift dist depending on its type
        if callable(n_of_z):
            if z_lim_up is None:
                raise ValueError("An upper bound of the redshift normalization has to be defined if n_of_z is not a "
                                 "tabulated function.")

            self.nz_intpt = n_of_z
            # set the integration limit and integration points
            self.lightcone_points = None
            self.limit = 1000
        else:
            # read from file
            nz = np.genfromtxt(n_of_z)

            # get the upper bound if necessary
            if z_lim_up is None:
                z_lim_up = nz[-1, 0]

            # get the callable function
            self.nz_intpt = interp1d(nz[:, 0] - shift_nz, nz[:, 1], bounds_error=False, fill_value=0.0)

            # points for integration
            self.lightcone_points = nz[np.logical_and(z_lim_low < nz[:, 0], nz[:, 0] < z_lim_up), 0]

            # check if there are any points remaining for the integration
            if len(self.lightcone_points) == 0:
                self.lightcone_points = None
                self.limit = 1000
            else:
                self.limit = 10 * len(self.lightcone_points)

        self.z_lim_up = z_lim_up
        self.z_lim_low = z_lim_low
        self.IA = IA
        # Normalization
        self.nz_norm = integrate.quad(lambda x: self.nz_intpt(x), z_lim_low, self.z_lim_up,
                                      points=self.lightcone_points, limit=self.limit)[0]

    def __call__(self, z_low, z_up, cosmo):
        """
        Computes the lensing weights for the redshift interval [z_low, z_up].
        :param z_low: lower end of the redshift interval
        :param z_up: upper end of the redshift interval
        :param cosmo: Astropy.Cosmo instance, controls the cosmology used
        :return: lensing weight
        """
        norm = utils.dimensionless_comoving_distance(z_low, z_up, cosmo) * self.nz_norm
        norm *= (utils.dimensionless_comoving_distance(0., (z_low + z_up)/2., cosmo) ** 2.)
        if abs(self.IA - 0.0) < 1e-10:
            # lensing weights without IA
            numerator = integrate.quad(self._integrand_1d, z_low, z_up, args=(cosmo,))[0]
        else:
            # lensing weights for IA
            numerator = (2.0/(3.0*cosmo.Om0)) * \
                        w_IA(self.IA, z_low, z_up, cosmo, self.nz_intpt, points=self.lightcone_points)

        return numerator / norm

    def _integrand_2d(self, y, x, cosmo):
        """
        The 2d integrant of the continous lensing weights
        :param y: redhsift that goes into the n(z)
        :param x: redshift for the Dirac part
        :param cosmo: Astropy.Cosmo instance, controls the cosmology used
        :return: the 2d integrand function
        """
        return self.nz_intpt(y) * \
               utils.dimensionless_comoving_distance(0, x, cosmo) * \
               utils.dimensionless_comoving_distance(x, y, cosmo) * \
               (1 + x) * \
               cosmo.inv_efunc(x) / \
               utils.dimensionless_comoving_distance(0, y, cosmo)

    def _integrand_1d(self, x, cosmo):
        """
        Function that integrates out y from the 2d integrand
        :param x: at which x (redshfit to eval)
        :param cosmo: Astropy.Cosmo instance, controls the cosmology used
        :return: the 1d integrant at x
        """
        if self.lightcone_points is not None:
            points = self.lightcone_points[np.logical_and(self.z_lim_low < self.lightcone_points,
                                                          self.lightcone_points < self.z_lim_up)]
            quad_y = lambda x: integrate.quad(lambda y: self._integrand_2d(y, x, cosmo), x, self.z_lim_up,
                                              limit=self.limit, points=points)[0]
        else:
            quad_y = lambda x: integrate.quad(lambda y: self._integrand_2d(y, x, cosmo), x, self.z_lim_up,
                                              limit=self.limit)[0]

        return quad_y(x)



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
        :param cosmo: Astropy.Cosmo instance, controls the cosmology used
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
               cosmo.inv_efunc(x)


def kappa_prefactor(n_pix, n_particles, boxsize, cosmo):
    """
    Computes the prefactor to transform from number of particles to convergence, see https://arxiv.org/abs/0807.3651,
    eq. (A.1).
    :param n_pix: number of healpix pixels used
    :param n_particles: number of particles
    :param boxsize: size of the box in Gigaparsec
    :param cosmo: Astropy.Cosmo instance, controls the cosmology used
    :return: convergence prefactor
    """
    convergence_factor = (3.0 * cosmo.Om0 / 2.0) * \
                         (n_pix / (4.0 * np.pi)) * \
                         (cosmo.H0.value / constants.c) ** 3 * \
                         (boxsize * 1000.0) ** 3 / n_particles
    return convergence_factor


def F_NIA_model(z, IA, cosmo):
    """
    Calculates the NIA kernel used to calculate the IA shell weight
    :param z: Redshift where to evaluate
    :param IA: Galaxy intrinsic alignments amplitude
    :param cosmo: Astropy.Cosmo instance, controls the cosmology used
    :return: NIA kernel at redshift z
    """
    OmegaM = cosmo.Om0
    H0 = cosmo.H0.value

    # growth factor calculation
    growth = lambda a: 1.0 / (a * cosmo.H(1.0 / a - 1.0).value)**3.0
    a = 1.0 / (1.0 + z)
    g = 5.0 * OmegaM / 2.0 * cosmo.efunc(z) * integrate.quad(growth, 0, a)[0]

    # Calculate the growth factor today
    g_norm = 5.0 * OmegaM / 2.0 * integrate.quad(growth, 0, 1)[0]

    # divide out a
    g = g / g_norm

    # critical density today = 3*H^2/(8piG)
    rho_c = cosmo.critical_density0.to("Msun Mpc^-3").value

    # Proportionality constant Msun^-1 Mpc^3
    C1 = 5e-14 / (H0/100.0) ** 2

    return -IA * rho_c * C1 * OmegaM / g


def w_IA(IA, z_low, z_up, cosmo, nz_intpt, points=None):
    """
    Calculates the weight per slice for the NIA model given a
    distribution of source redshifts n(z).
    :param IA: Galaxy Intrinsic alignment amplitude
    :param z_low: Lower redshift limit of the shell
    :param z_up: Upper redshift limit of the shell
    :param cosmo: Astropy.Cosmo instance, controls the cosmology used
    :param nz_intpt: nz function
    :param points: Points in redshift where integrand is evaluated (used for better numerical integration), can be None
    :return: Shell weight for NIA model

    """

    def f(x):
        return (F_NIA_model(x, IA, cosmo) * nz_intpt(x))

    if points is not None:
        dbl = integrate.quad(f, z_low, z_up, points=points[np.logical_and(z_low < points, points < z_up)])[0]
    else:
        dbl = integrate.quad(f, z_low, z_up)[0]

    return dbl
