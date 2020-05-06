#!/usr/bin/env python

import os

from setuptools import find_packages, setup


requirements = ['numpy',
                'scipy',
                'healpy',
                'h5py',
                'pyyaml']  # during runtime


with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read().replace(".. :changelog", "")


doclink = """
Documentation
-------------

The full documentation can be generated with Sphinx"""


PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))


setup(
    name='UFalcon',
    version='1.0.0',
    description='Ultra Fast Lightcone',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Raphael Sgier',
    author_email='rsgier@phys.ethz.ch',
    url='https://cosmo-docs.phys.ethz.ch/UFalcon',
    packages=find_packages(include=["UFalcon"]),
    include_package_data=True,
    install_requires=requirements,
    license="MIT Licence",
    zip_safe=False,
    keywords="UFalcon",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
    ],
)
