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


## The Rest

✅✅ ... found and saved in configs

✅🤔 ... exists but config not ready

🤔 ... not found (not sure if it works)

### GHZ
* 3 particle, 4 dimension ✅✅
* 3 particle, 5 dimension ✅✅
* 3 particle, 6 dimension ✅✅
* 4 particle, 4 dimension ("fake") ✅✅
* 4 particle, 4 dimension (HALO)
* 5 particle, 4 dimension ✅✅
* 6 particle, 3 dimension (HALO)

### Quantum Info
* BSSB4 state ✅✅
* Cluster states
  * 4 particle ✅✅
  * 5 particle ✅✅
  * 6 particle ✅✅
* Psi5 state ✅
* random matrix state 1 (3 qubits) ✅✅
* random matrix state 2 (3 qubits) ✅✅
* symmetric state
  * 3 particle, 3 dimension ✅✅
  * 4 particle, 3 dimension ✅✅
  * 5 particle, 2 dimension ✅✅
  * 6 particle, 2 dimension 🤔
* Schmidt rank vector
  * (5,5,4) ✅✅
  * (6,3,2) ✅✅
  * (6,5,5) ✅✅
  * (7,3,3) 🤔
* W state x W state ✅✅
* Steane Code 🤔
* Hyperdeterminant State ✅✅
* L state ✅✅
* Yeo Chua state ✅✅
* Higuchi Sudbery state ✅🤔

### k-uniform and AME states
* 5 particle, 2 dimension AME ✅🤔
* 6 particle, 2 dimension AME ("fake")  ✅🤔
* 6 particle, 2 dimension, k=2 uniform  ✅🤔
* 7 particle, 2 dimension, 'almost' k=2  ✅🤔
* 8 particle, 2 dimension, 'almost' k=3  ✅🤔

### Mixed States
* Werner State ✅✅
* Peres State 🤔
* more ?

### Measurements / Quantum Comm
* GHZ analyzer
  * 3 particle, 2 dimension ✅✅
  * 3 particle, 3 dimension ✅✅
  * 3 particle, 4 dimension 🤔
* Mean King 🤔
* 4d Entanglement swapping (HALO)

### Gates
* CNOT(2,2) (known)
* CNOT(2,3) ✅✅
* CNOT(2,4) ✅✅
* CNOT(3,3) ✅✅
* Toffoli (known?)
* controlled Z (known?)
* more ?

### Condensed Matter
* AKLT
  * 3 particle ✅✅
  * 4 particle 🤔
* Haldane states
  * 3 particle A 🤔
  * 3 particle B ✅✅
  * 3 particle C ✅✅
* Majumdar Gosh states
  * 4 particle ✅✅
  * 6 particle ✅✅
* N body
  * 3 particle ✅✅
  * 4 particle ✅✅
  * 5 particle ✅✅
  * 6 particle ✅✅
* weak Antiferrometric
  * 1 - 3 particle 🤔
  * 2 - 3 particle 🤔
  * 3 - 3 particle ✅✅
  * 4 - 3 particle ✅✅
* 3 particle spin3- ✅✅
* 3 particle spin3+ ✅✅
* 4 particle spin half ✅✅
* 3 particle spin1 ("fake") ✅✅
* 1d spin half wire ✅✅

### Other
* 4 qubit state that needs complex numbers ✅✅

### More ideas/inspiration
* maximize properties of mixed states
* graph theoretical properties, assembly index, etc.
* optimize quantum info inequalities (similar to CHSH)
* maximize robustness (similar to HS state)
* GKP states
* Fock states
* Heralded states
* experiments with interesting restrictions
* 9 ways of entangling 4 qubits
