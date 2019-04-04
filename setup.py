#!/usr/bin/env python

import os
import sys

from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


try:
    readme = open('README.rst').read()
except IOError:
    # when this package is installed as dependency some files might not be
    # present:
    readme = ""

doclink = """
Documentation
-------------

The full documentation can be generated with Sphinx"""

try:
    history = open('HISTORY.rst').read().replace('.. :changelog:', '')
    # when this package is installed as dependency some files might not be
    # present:
except IOError:
    history = ""

requires = []  # during runtime
tests_require = ['pytest>=2.3']  # for testing

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(
    name='UFalcon',
    version='0.1.0',
    description='Ultra Fast Lightcone',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Raphael Sgier',
    author_email='rsgier@phys.ethz.ch',
    url='https://cosmo-docs.phys.ethz.ch/UFalcon',
    packages=find_packages(PACKAGE_PATH, "test"),
    package_dir={'UFalcon': 'UFalcon'},
    include_package_data=True,
    install_requires=requires,
    license='Proprietary',
    zip_safe=False,
    keywords='UFalcon',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        "Intended Audience :: Science/Research",
        'Intended Audience :: Developers',
        'License :: Other/Proprietary License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    tests_require=tests_require,
    cmdclass={'test': PyTest},
)
