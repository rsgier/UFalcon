=============================
UFalcon - Version 0.2.0
=============================

.. image:: http://img.shields.io/badge/arXiv-1801.05745-orange.svg?style=flat
        :target: https://arxiv.org/abs/1801.05745


UFalcon - Ultra Fast Lightcone

Package for constructing full-sky weak lensing maps from lightcones based on N-Body simulation output within a minimal runtime. Written in Python 3.

Introduced in `Sgier et al. 2019 <https://iopscience.iop.org/article/10.1088/1475-7516/2019/01/044>`_ and extended in Sgier et al. 2020 (in prep.).

N-Body Simulations
--------

Currently supported N-Body simulation codes are:

* PKDGRAV3 (`Potter et al. 2016 <https://arxiv.org/abs/1609.08621>`_) available `here <https://bitbucket.org/dpotter/pkdgrav3/src/master/>`_.
* L-PICOLA (`Howlett et al. 2015 <https://arxiv.org/abs/1506.03737>`_) available `here <https://cullanhowlett.github.io/l-picola/>`_.

Note that UFalcon currently only supports post-processing of simulation output generated in lightcone mode.

Why use UFalcon?
--------

In order to accurately infer cosmological constraints from current and future weak lensing data, a large ensemble of survey simulations are required to model cosmological observables
and a well-converged covariance matrix. UFalcon applied to PKDGRAV3-output combines accuracy and minimal computational runtime: The simulation of the density field guarantees to satisfy a certain force accuracy and is therefore not an approximate N-Body code. Furthermore, the PKDGRAV3 code is highly efficient and can be run with graphics processing units (GPU) support. The subsequent post-processing with UFalcon can be parallelized on a computer cluster and has a runtime of less than 30 min walltime per convergence mass map. Furthermore, it offers a high flexibility for the lightcone construction, such as user-specific redshift ranges and redshift distributions and single-source redshifts.


Features
--------

* Fast computation of fullsky Healpix maps (`Gorski et al. 2005 <https://iopscience.iop.org/article/10.1086/427976>`_) containing particle counts (shells)
* Fast construction of weak lensing maps (convergence, shear) for user-specific redshift distributions and single-source redshifts

Getting Started
--------

The files in the folder scripts contain some example-function you can use and adapt for your analysis. These are:

- construct_shells.py:

    Computes and stores healpy maps containing the particle counts (shells) from N-Body simulation output.

Some example-functions showing how to implement UFalcon for your analysis can be found in the files folder scripts.

Credits
--------

* Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics, `Cosmology Group <https://cosmology.ethz.ch/>`_
* Authors: Raphael Sgier and Jörg Herbel
* Contact: Raphael Sgier raphael.sgier@phys.ethz.ch.