#!/usr/bin/env python
import numpy as np

#from distutils.core import setup
import setuptools

setuptools.setup(name='astrotools',
    version='1.0',
    author='Julien Lhermitte',
    description="Generic Deep Sky Astrophotography Tools",
    include_dirs=[np.get_include()],
    author_email='jrmlhermitte@gmail.com',
    keywords='Astronomy Deep Sky Astrophotography',
)
