#!/usr/bin/env python

import os
import numpy as np
from setuptools import setup, find_packages
from Cython.Build import cythonize


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

print(find_packages())
setup(
    name = 'nnet',
    version = '0.1',
    author = 'Anders Boesen Lindbo Larsen',
    author_email = 'abll@dtu.dk',
    description = "Neural networks in NumPy/Cython",
    license = 'MIT',
    url = 'http://compute.dtu.dk/~abll',
    packages = find_packages(),
    install_requires = ['numpy', 'scipy', 'cython'],
    long_description = read('README.md'),
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
    ],
    ext_modules = cythonize(['nnet/convnet/conv.pyx',
                             'nnet/convnet/pool.pyx']),
    include_dirs = [np.get_include()]
)

