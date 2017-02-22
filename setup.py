#!/usr/bin/env python

from distutils.core import setup

setup(name='PRiNCe',
      version='0.0',
      description='Propagation including Nuclear Cascade equations',
      author='Anatoli Fedynitch',
      author_email='afedynitch@gmail.com',
      url='https://github.com/afedynitch',
      packages=['prince'],
      py_modules=['prince_config'],
      requires=['numpy', 'scipy', 'matplotlib', 'jupyter', 'progressbar']
     )