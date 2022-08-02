import os
import sys

import click
import pkg_resources

import theseus
from theseus.main import run_main
from theseus.analyzer import analyser

@click.group()
def cli():
    pass


@cli.command()
@click.argument('filename')
@click.option('--example', is_flag=True, default=False, help='Load input file from examples directory.')
def run(filename, example):
    """Run an input file."""
    try:
        run_main(filename, example)
    except IOError as e:
        click.echo('ERROR:' + str(e))
        sys.exit(1)

@cli.command()
@click.argument('directory')
def analyze(directory):
    """Run an input file."""
    try:
        #data_dir = pkg_resources.resource_filename(theseus.__name__, 'data',directory)
        a = analyser(directory)
        index = input(f'which state? (int from 0 - {len(a.files)} ) \n')
        a.info_statex(int(index))
    except IOError as e:
        click.echo('ERROR:' + str(e))
        sys.exit(1)


@cli.command()
def list():
    """List all included examples."""
    configs_dir = pkg_resources.resource_filename(theseus.__name__, 'configs')
    files = sorted(os.listdir(configs_dir))
    for file in files:
        click.echo(file.replace('.json', ''))
