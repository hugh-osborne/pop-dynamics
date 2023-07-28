# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='popdynamics',
    version='0.0.1',
    description='General purpose numerical population density solver.',
    long_description=readme,
    author='Hugh Osborne',
    author_email='hugh.osborne@gmail.com',
    url='https://github.com/hugh-osborne/pop-dynamics',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

