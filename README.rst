=============================
UFalcon - Version 0.1.0
=============================

UFalcon - Ultra Fast Lightcone

Package for constructing full-sky maps from lightcones based on N-Body simulation output. Written in Python 3.

Introduced in `Sgier et al. 2019 <https://iopscience.iop.org/article/10.1088/1475-7516/2019/01/044>`_ and Sgier et al. 2020 (in prep.).

N-Body Simulations
--------

Currently supported N-Body simulation codes are:

* PKDGRAV3 (`Potter et al. 2016 <https://arxiv.org/abs/1609.08621>`_) available `here <https://bitbucket.org/dpotter/pkdgrav3/src/master/>`_.
* L-PICOLA (`Howlett et al. 2015 <https://arxiv.org/abs/1506.03737>`_) available `here <https://cullanhowlett.github.io/l-picola/>`_.

Note that UFalcon currently only supports post-processing of simulation output generated in <span style="font-weight:bold">lightcone mode</span>.

Features
--------

* Fast computation of fullsky Healpix maps (`Gorski et al. 2005 <https://iopscience.iop.org/article/10.1086/427976>`_) containing particle counts (shells)
* Fast construction of weak lensing maps (convergence, shear) for user-specific redshift distributions and single-source redshifts

Getting Started
--------

Some example-functions showing how to implement UFalcon for your analysis can be found in the folder scripts.

Credits
--------

* Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics, `Cosmology Group <https://cosmology.ethz.ch/>`_
* Authors: Raphael Sgier and Jörg Herbel
* Contact: Raphael Sgier raphael.sgier@phys.ethz.ch.