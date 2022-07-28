import os
import sys

import click
import pkg_resources

import theseuslab
from theseuslab.main import run_main


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
def list():
    """List all included examples."""
    configs_dir = pkg_resources.resource_filename(theseuslab.__name__, 'configs')
    files = sorted(os.listdir(configs_dir))
    for file in files:
        click.echo(file.replace('.json', ''))
