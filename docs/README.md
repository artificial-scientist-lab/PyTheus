# How to work with the Sphinx docs

## Install sphinx and dependencies

In project root directory install Theseus in development mode:

```
python setup.py develop
```

Then, in `./docs` do

```
pip install -r requirements
```

## Build documentation

```
make html
```

## Inspect generated docs

Open in browser:

```
docs/_build/html/index.html
```

