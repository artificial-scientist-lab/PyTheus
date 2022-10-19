""" Module: cli.py

Theseus uses the Click library as a command line interface. Definition of click commands
and error handling should be contained in the cli.py module.

Conversely, to avoid tight coupling of code, all "business" logic, i.e. actions that shall be
performed when submitting a command, should be part of other modules, but NOT of cli.py.

In case of errors of any sort, the other modules should raise exceptions that cli.py
may handle as desired.

This decoupling makes sure that Theseus can not only be used as a command line application
but can also be imported as a package in normal Python code.
"""

import os
import sys
import json

import click
import pkg_resources

import pytheus
from pytheus.main import run_main
from pytheus.analyzer import get_analyse
from pytheus.graphplot import plotFromFile


@click.group()
def cli():
    pass


@cli.command()
@click.argument('filename')
@click.option('--example', is_flag=True, default=False, help='Load input file from examples directory.')
def run(filename, example):
    """Run an input file."""
    try:
        # run main routine (read config, set up target, run optimization/store results)
        run_main(filename, example)
    except IOError as e:
        click.echo('ERROR:' + str(e))
        sys.exit(1)


@cli.command()
@click.argument('filename')
@click.option('--pdf', default="", help='save output to pdf')
def plot(filename, pdf):
    """Plot a solution file."""
    try:
        plotFromFile(filename, outfile=pdf)
    except IOError as e:
        click.echo('ERROR:' + str(e))
        sys.exit(1)


@cli.command()
@click.option('-d', '--which-directory', default=None,
              help='choose folder to analyze')
@click.option('-one', '--all-weights-plus-minus-one', is_flag=True,
              show_default=True,
              help='map all weights to plus minus one')
@click.option('-pm', '--create-perfect-machting-pdf', is_flag=True,
              show_default=True,
              help='bool if create pdf with all pms')
@click.option('-i', '--which-infos', default=['norm', 'ent', 'k'],
              multiple=True,
              show_default=True,
              help='list of which infos appear in info plot')
def analyze(which_directory, all_weights_plus_minus_one,
            create_perfect_machting_pdf, which_infos):
    """Run anlyzer tool depending on inputs."""
    try:
        get_analyse(which_directory,
                    all_weights_plus_minus_one=all_weights_plus_minus_one,
                    create_perfect_machting_pdf=create_perfect_machting_pdf,
                    which_infos=which_infos)
    except IOError as e:
        click.echo('ERROR:' + str(e))
        sys.exit(1)


@cli.command()
def list():
    """List all included examples."""
    configs_dir = pkg_resources.resource_filename(pytheus.__name__, 'graphs')
    walk = os.walk(configs_dir)
    for root, dirs, files in walk:
        for file in files:
            if file.startswith('config'):
                click.echo(
                print(os.path.basename(root)))
