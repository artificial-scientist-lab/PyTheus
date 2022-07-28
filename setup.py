#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import exists
from setuptools import setup, find_packages

author = 'artificial-scientist-lab'
email = 'TODO'
description = 'Theseus, a highly-efficient inverse-design algorithm for quantum optical experiments'
name = 'theseuslab'
year = '2022'
url = 'https://github.com/artificial-scientist-lab/Theseus'
version = '0.0.1'

setup(
    name=name,
    author=author,
    author_email=email,
    url=url,
    version=version,
    packages=find_packages(),
    package_dir={name: name},
    include_package_data=True,
    license='None',
    description=description,
    long_description=open('README.md').read() if exists('README.md') else '',
    long_description_content_type="text/markdown",
    install_requires=['sphinx', 'numpy', 'scipy', 'matplotlib', 'termcolor'
                      ],
    python_requires=">=3.6",
    classifiers=['Operating System :: OS Independent',
                 'Programming Language :: Python :: 3',
                 ],
    platforms=['ALL'],
    py_modules=[name],
    entry_points={
        'console_scripts': [
            'theseus = theseuslab.main:run_main',
        ],
    }
)
