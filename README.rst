==============================================
UFalcon (Ultra Fast Lightcone) - Version 0.2.0
==============================================

.. image:: https://cosmo-gitlab.phys.ethz.ch/cosmo_public/UFalcon/badges/master/coverage.svg
        :target: https://cosmo-gitlab.phys.ethz.ch/cosmo_public/UFalcon

.. image:: https://cosmo-gitlab.phys.ethz.ch/cosmo_public/UFalcon/badges/master/pipeline.svg
        :target: https://cosmo-gitlab.phys.ethz.ch/cosmo_public/UFalcon

.. image:: http://img.shields.io/badge/arXiv-1801.05745-orange.svg?style=flat
        :target: https://arxiv.org/abs/1801.05745

Package for constructing full-sky weak lensing maps from lightcones based on N-Body simulation output within a minimal runtime. Written in Python 3.

Introduced in `Sgier et al. 2019 <https://iopscience.iop.org/article/10.1088/1475-7516/2019/01/044>`_ and extended in `Sgier et al. 2020 <https://arxiv.org/abs/2007.05735>`_.

Why use UFalcon?
================

In order to accurately infer cosmological constraints from current and future weak lensing data, a large ensemble of survey simulations are required to model cosmological observables
and a well-converged covariance matrix. UFalcon applied to PKDGRAV3-output combines accuracy and minimal computational runtime: The simulation of the density field guarantees to satisfy a certain force accuracy and is therefore not an approximate N-Body code. Furthermore, the PKDGRAV3 code is highly efficient and can be run with graphics processing units (GPU) support. The subsequent post-processing with UFalcon can be parallelized on a computer cluster and has a runtime of less than 30 min walltime per convergence mass map. The package offers a high flexibility for the lightcone construction, such as user-specific redshift ranges, redshift distributions and single-source redshifts. Furthermore, UFalcon offers the possibility to compute the galaxy intrinsic alignment signal, which can be treated as an additive component to the cosmological signal.

Features
========

* Fast computation of fullsky Healpix maps (`Gorski et al. 2005 <https://iopscience.iop.org/article/10.1086/427976>`_) containing particle counts (shells).
* Fast construction of weak lensing maps (convergence, shear) for user-specific redshift distributions and single-source redshifts.
* Computation of galaxy intrinsic alignment (IA) signal (additive to the cosmological signal) based on the nonlinear intrinsic alignment model (NLA) (`Bridle et al. 2007 <https://arxiv.org/abs/0705.0166>`_ , `Hirata et al. 2004 <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.70.063526>`_ and `Joachimi et al. 2011 <https://www.aanda.org/articles/aa/abs/2011/03/aa15621-10/aa15621-10.html>`_) and applied in `Zürcher et al. 2020 <https://arxiv.org/abs/2006.12506>`_.

N-Body Simulations
==================

Currently supported N-Body simulation codes are:

* PKDGRAV3 (`Potter et al. 2016 <https://arxiv.org/abs/1609.08621>`_) available `here <https://bitbucket.org/dpotter/pkdgrav3/src/master/>`_.
* L-PICOLA (`Howlett et al. 2015 <https://arxiv.org/abs/1506.03737>`_) available `here <https://cullanhowlett.github.io/l-picola/>`__.

Note that UFalcon currently only supports post-processing of simulation output generated in lightcone mode.

Getting Started
===============

The following sections provide an overview of the UFalcon pipeline and some example-scripts, which you can use for your analysis.

Credits
=======

- If you use UFalcon for your research please cite `Sgier et al. 2019 <https://iopscience.iop.org/article/10.1088/1475-7516/2019/01/044>`_ and `Sgier et al. 2020 <https://arxiv.org/abs/2007.05735>`_.
- Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics, `Cosmology Group <https://cosmology.ethz.ch/>`_
- Authors: Raphael Sgier and Jörg Herbel
- Contact: Raphael Sgier raphael.sgier@phys.ethz.ch.
