# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

setup(
    name="pyrpca",
    version="0.1.0",

    author="Alex Papanicolaou",
    author_email="alex.papanic@gmail.com",maintainer="Alex Papanicolaou",
    maintainer_email="alex.papanic@gmail.com",

    description="Python implementations of RPCA",
    long_description=readme,

    packages=find_packages(exclude=('tests', 'docs')),

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
)