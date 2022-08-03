# Theseus
Theseus, a highly-efficient inverse-design algorithm for quantum optical experiments

## Installation

Using pip:

```
pip install theseuslab
```

Alternatively, from sources (after cloning the repository):

```
python setup.py install
```

## Running Theseus

To list the included examples, type

```
theseus list
```

To run one of the included examples, type e.g.

```
theseus run --example ghz_346
```

To run your own input file, type

```
theseus run PATH_TO_YOUR_INPUT_FILE
```

Output of optimization is saved to a directory called `output`. Names of the subdirectorie are specified by the name
and content of the config file.

To analyze a subdirectory corresponding to one run, type

```
theseus analyze outputs/ghz_346/ghz_346
```


To get help, add the `--help` option to any command. For instance

```
> theseus run --help

Usage: theseus run [OPTIONS] FILENAME

  Run an input file.

Options:
  --example  Load input file from examples directory.
  --help     Show this message and exit.
```

## Development

### Clone repository

```
git clone https://github.com/artificial-scientist-lab/Theseus.git
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


## Tests

### Run test suite

#### Running all tests

```
python -m unittest discover tests
```

#### Running only the fast tests

```
python -m unittest discover -s tests/fast
```

### Test coverage

Install `coverage`, if you have not yet done so:

```
pip install coverage
```

Then run coverage scan:

```
coverage run --source=theseus -m unittest discover tests 
```

After that, create the coverage report:

```
coverage report -m
```

