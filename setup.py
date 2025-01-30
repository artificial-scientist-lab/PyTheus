#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import exists
from pathlib import Path
import re

from setuptools import setup, find_packages

author = 'artificial-scientist-lab'
email = 'cruizgo@proton.me, soeren.arlt@mpl.mpg.de, mario.krenn@mpl.mpg.de'
description = 'PyTheus, a highly-efficient inverse-design algorithm for quantum optical experiments'
dist_name = 'pytheusQ'
package_name = 'pytheus'
year = '2024'
url = 'https://github.com/artificial-scientist-lab/Pytheus'


def get_version():
    content = open(Path(package_name) / '__init__.py').readlines()
    return "1.2.9"


setup(
    name=dist_name,
    author=author,
    author_email=email,
    url=url,
    version=get_version(),
    packages=find_packages(),
    package_dir={dist_name: package_name},
    include_package_data=True,
    license='MIT',
    description=description,
    long_description=open('README.md').read() if exists('README.md') else '',
    long_description_content_type="text/markdown",
    install_requires=[
        'sphinx', 'numpy', 'scipy', 'matplotlib', 'termcolor', 'Click'
    ],
    python_requires=">=3.8",
    classifiers=['Operating System :: OS Independent',
                 'Programming Language :: Python :: 3',
                 ],
    platforms=['ALL'],
    py_modules=[package_name],
    entry_points={
        'console_scripts': [
            'pytheus = pytheus.cli:cli',
        ],
    }
)
