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

    def __init__(self, n_of_z, interpolation_kind='linear', z_lim_low=0, z_lim_up=None, shift_nz=0.0, IA=0.0, eta=0.0,
                 z_0=0.5, fast_mode=False):
        """
        Constructor.
        :param n_of_z: either path to file containing n(z), assumed to be a text file readable with numpy.genfromtext
                       with the first column containing z and the second column containing n(z), or a callable that
                       is directly a redshift distribution
        :param interpolation_kind: This argument specifies type of interpolation used, if the redshift distribution is
                                   read from a file. It is directly forwarded to scipy.interpolate.interp1d and
                                   defaults to 'linear'
        :param z_lim_low: lower integration limit to use for n(z) normalization, default: 0
        :param z_lim_up: upper integration limit to use for n(z) normalization, default: last z-coordinate in n(z) file
        :param shift_nz: Can shift the n(z) function by some redshift (intended for easier implementation of photo z bias)
        :param IA: Intrinsic alignment amplitude for the NLA model.
        :param eta: Parameter for the redshift dependence of the NLA model.
        :param z_0: Pivot parameter for the redshift dependence of the NLA model
        :param fast_mode: Instead of using quad from scipy, use a simple simpson rule, note that this will drastically
                          decrease the runtime of the weight calculation if you n(z) is no continuous, while reducing
                          the accuracy and increasing the memory usage. This should not be used for highly oscillation
                          redshift distributions!
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
            self.nz_intpt = interp1d(nz[:, 0] - shift_nz, nz[:, 1], bounds_error=False, fill_value=0.0,
                                     kind=interpolation_kind)

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
        self.eta = eta
        self.z_0 = z_0
        self.fast_mode = fast_mode
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
        if self.IA is None or abs(self.IA - 0.0) < 1e-10:
            if self.fast_mode:
                z_vals, dz = np.linspace(z_low, z_up, 13, retstep=True)
                quad_y_vals = self._integrand_1d(z_vals, cosmo)
                numerator = integrate.simps(quad_y_vals, dx=dz, axis=0)
            else:
                # lensing weights without IA
                numerator = integrate.quad(self._integrand_1d, z_low, z_up, args=(cosmo,))[0]
        else:
            # lensing weights for IA
            numerator = (2.0/(3.0*cosmo.Om0)) * \
                        w_IA(self.IA, self.eta, z_low, z_up, cosmo, self.nz_intpt, z_0=self.z_0,
                             points=self.lightcone_points)

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
               utils.dimensionless_comoving_distance(0, x, cosmo, fast_mode=self.fast_mode) * \
               utils.dimensionless_comoving_distance(x, y, cosmo, fast_mode=self.fast_mode) * \
               (1 + x) * \
               cosmo.inv_efunc(x) / \
               utils.dimensionless_comoving_distance(0, y, cosmo, fast_mode=self.fast_mode)

    def _integrand_1d(self, x, cosmo):
        """
        Function that integrates out y from the 2d integrand
        :param x: at which x (redshfit to eval)
        :param cosmo: Astropy.Cosmo instance, controls the cosmology used
        :return: the 1d integrant at x
        """
        if self.fast_mode:
            def quad_y(x):
                x = np.atleast_1d(x)
                y_vals = np.geomspace(np.maximum(x, 1e-4), self.z_lim_up, 512+1)
                f_vals = np.nan_to_num(self._integrand_2d(y_vals, x, cosmo))
                return integrate.simps(f_vals, x=y_vals, axis=0)
        else:
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


def F_NLA_model(z, IA, eta, z_0, cosmo):
    """
    Calculates the NLA kernel used to calculate the IA shell weight
    :param z: Redshift where to evaluate
    :param IA: Galaxy intrinsic alignments amplitude
    :param eta: Galaxy Intrinsic alignment redshift dependence
    :param z_0: Pivot parameter for the redshift dependence of the NLA model
    :param cosmo: Astropy.Cosmo instance, controls the cosmology used
    :return: NLA kernel at redshift z
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

    # redshift dependece term
    red_dep = ((1 + z) / (1 + z_0))**eta

    return -IA * rho_c * C1 * OmegaM / g * red_dep


def w_IA(IA, eta, z_low, z_up, cosmo, nz_intpt, z_0=0.5, points=None):
    """
    Calculates the weight per slice for the NLA model given a
    distribution of source redshifts n(z).
    :param IA: Galaxy Intrinsic alignment amplitude
    :param eta: Galaxy Intrinsic alignment redshift dependence
    :param z_low: Lower redshift limit of the shell
    :param z_up: Upper redshift limit of the shell
    :param cosmo: Astropy.Cosmo instance, controls the cosmology used
    :param nz_intpt: nz function
    :param z_0: Pivot parameter for the redshift dependence of the NLA model
    :param points: Points in redshift where integrand is evaluated (used for better numerical integration), can be None
    :return: Shell weight for NLA model

    """

    def f(x):
        return (F_NLA_model(x, IA, eta, z_0, cosmo) * nz_intpt(x))

    if points is not None:
        dbl = integrate.quad(f, z_low, z_up, points=points[np.logical_and(z_low < points, points < z_up)])[0]
    else:
        dbl = integrate.quad(f, z_low, z_up)[0]

    return dbl
