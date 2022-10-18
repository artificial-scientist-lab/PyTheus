# PyTheus
PyTheus, a highly-efficient inverse-design algorithm for quantum optical experiments

## Installation

The package can be installed with

```
pip install pytheus
```

## Running PyTheus

To list the included examples, type

```
pytheus list
```

To run one of the included examples, type e.g.

```
pytheus run --example ghz_346
```

To run your own input file, type

```
pytheus run PATH_TO_YOUR_INPUT_FILE
```

Output of optimization is saved to a directory called `output`. Names of the subdirectories are specified by the name
and content of the config file.

To plot the graph corresponding to one result saved as a json file, execute 

```
pytheus plot PATH_TO_RESULT_FILE
```

To get help, add the `--help` option to any command. For instance

```
> pytheus run --help

Usage: pytheus run [OPTIONS] FILENAME

  Run an input file.

Options:
  --example  Load input file from examples directory.
  --help     Show this message and exit.
```

## Development

### Clone repository

```
git clone https://github.com/artificial-scientist-lab/PyTheus.git
```

### Create virtual environment

From the project root directory, submit

```
python -m venv venv
```

This will create a subfolder with your virtual environment.

To activate, type

```
. venv/bin/activate
```

Note the leading point!

### Local development installation

Submit

```
python setup.py develop
```

from the project root directory (where `setup.py` is located).
Any changes in the code will now automatically be reflected
in your local package installation.
