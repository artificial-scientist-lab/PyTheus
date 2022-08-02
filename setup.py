#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import exists
from pathlib import Path
import re

from setuptools import setup, find_packages

author = 'artificial-scientist-lab'
email = ''  # TODO: insert email
description = 'Theseus, a highly-efficient inverse-design algorithm for quantum optical experiments'
dist_name = 'theseuslab'
package_name = 'theseus'
year = '2022'
url = 'https://github.com/artificial-scientist-lab/Theseus'  # TODO: insert public repo URL


def get_version():
    content = open(Path(package_name) / '__init__.py').readlines()
    for line in content:
        match = re.match('^ *__version__ *= *[\'"]([^\'"]+)', line)
        if match:
            return match.group(0)
    raise Exception('Cannot extract version string.')


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
    python_requires=">=3.6",
    classifiers=['Operating System :: OS Independent',
                 'Programming Language :: Python :: 3',
                 ],
    platforms=['ALL'],
    py_modules=[package_name],
    entry_points={
        'console_scripts': [
            'theseus = theseus.cli:cli',
        ],
    }
)