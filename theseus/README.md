# Pytheus

# Outline for Chapter 1

## 1. Graphs & Experiments (2 pages)

### 1.1 Crystals (Probabilistic 2 photon sources)

* Equation & graph example with experiment & more general equation
* Perfect matchings (Post-selection)
* Heralding

### 1.2 Single Photon Sources

### 1.3 Number-resolving Detectors

### 1.4 Quantum Communication

### 1.5 Measurements

### 1.6 Quantum Gates

### 1.7 Fock Basis

# Ideas for Outline of Chapter 2 (not fixed) (2 pages)

## 2. Implementation of Pytheus

### 2.1 Fidelity and Count Rate

### 2.2 Other Loss Functions

* Entanglement
* Assembly Index & IIT

### 2.3 Topological Rules

* see documentation below

### 2.4 Post-selections Rules

* see documentation below

### 2.5 Running Config Files

* flowchart
* key classes: Graph, State, Optimizer, Saver

## Discovery for Diverse Experimental Resources

Our package allows for the discovery of quantum experiments for a range of experimental goals, constraints and
resources. Experiments that can be produced include:

* state creation (heralded or post-selected)
* quantum gates (heralded or post-selected)
* measurements of quantum states
* entanglement swapping
* (covered elsewhere: mixed state creation)

Sources for photons in these experiments can be SPDC sources, deterministic single-photon sources or a mix of the two.

Detectors can be photon-number-resolving or not.

Each of these experiments can be described with a graph. The interpretation of nodes and edges varies with the kind of
experiment.

### Rules for Loss Functions

With these varying interpretations (e.g. for single photon sources, input photons, entanglement swapping), different
constraints apply on what kind of graph can correspond to an experiment (Topological Rules).

With the different ways of performing the experiments (heralded/post-selected & number-resolving/non-number-resolving),
different events are selected out of all possibilities (post-selection rules).

#### Topological Rules

All experiments that our package is applied to can be described by a graph. When describing state creation using SPDC
each edge can be interpreted as a pair-creation. In this case all edges of the complete graph can be considered
physically legitimate. When describing other experiments edges can be interpreted differently. Not every edge will be
physically meaningful. Consequentially there are constraints on which connections of the complete graph are used in the
optimization.

*(A) Single Photon Sources and Input Photons*

Deterministic single photon sources and input photons (such as in gates) are described as (input) vertices in a graph.
An edge connecting an input vertex to a detector describes a path in which a photon can travel from the input into the
detector. This interpretation stems from the [Klyshko picture](https://arxiv.org/pdf/1805.06484.pdf). From this a
constraint on the graph follows. Two input vertices can not be connected by an edge. It could not be interpreted
physically.

*(B) Entanglement Swapping and Teleportation*

In entanglement swapping, photons are entangled that have not interacted before. If we want to design an entanglement
swapping experiment of two photons, the target is to discover a graph that produces an entangled state between the two
photons. However any edge between the corresponding vertices would translate into a common source crystal. A constraint
that ensures legitimate entanglement swapping is to remove any edge between the two parties

#### Post-Selection Rules

The rules for post-selecting coincidence events have been described in the
[Theseus paper](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.031044). Here, post-selection projects the
space of possibilities containing arbitrary combinations of crystals firing into the space of possibilities where only
crystals fire for which all detectors at the end of the experiment click. In the graph picture these combinations
correspond to the perfect matchings. A state is produced with fidelity one in post-selection if all possibilities of
coincidence events contribute to that state.

Other experimental settings (such as heralding) and additional experimental resources (such as number-resolving
detectors)
perform a different kind of projection on the space of possibilities by selecting for different events. This different
selection is reflected in the fidelity of the state. The products of the edge weights belonging to each possibility
contribute to the norm of the fidelity.

*(A) Heralding*

Heralding is a less strict form of selecting events. Instead of putting a detector in every path and selecting for
coincidence, only a subset of the paths are detected _heralding_ an output state in the unmeasured paths. This selection
rule not only allows for possibilities where one photon is in every path (perfect matchings) but also for other
possibilities (edge covers) as long as they cover the heralding detectors. This can lead to cross-terms that are not
present when post-selecting for coincidence in all paths. Consequentially it is more difficult to find a graph with
fidelity one, also requiring more experimental resources.

*(B) Single Photon Sources and Input Photons*

When describing heralded experiments (above) one has to consider edge covers instead of perfect matchings in the graph
for possible events. These possibilities include one edge being included twice in an edge cover, corresponding to a
crystal firing twice in an experiment. For single photon sources and other deterministic input photons such
possibilities do not exist. Only edge covers that cover the input vertices exactly once are considered for the norm of
the fidelity.

*(C) Photon Number-Resolving Detectors*

Photon number-resolving detectors are a valuable resource that can restrict the space of possibilities more than a
regular detector. When one can be certain that exactly one photon, and not two, has entered a detector it reduces the
number of events that could have led to this outcome, eliminating cross terms.

*(D) States in Fock Basis*

...

## Loss Functions For Target State Optimization

As explained above, the loss function depend largely on the different experimental conditions. Independent of these
conditions they fall into two categories.

* Fidelity
* Count Rate

A Fidelity of one ensures that an experiment has no unwanted cross terms. Every possibility that is selected for
contributes directly to the target outcome.

However, we have come to find that optimizing exclusively for fidelity in some cases can lead the optimization to scale
down the weights of the entire graph to minimize the contributions of crossterms. While the fidelity will be very close
to one in those cases the generally low edge weights would lead to very low count rates of successful events in actual
experiments.

To find solutions with higher weights we have introduced the _simplified count rate_ as a loss function.

# Config Examples

This section gives examples for config files showcasing the features for different kinds of experiments that can
searched for.

### Post-selected State creation

Here is an example for a config file optimizing for a graph that creates a three particle four-dimensional GHZ state.

```json
{
  "description": "Finding a setup for the creation of the three-particle four-dimensional GHZ state. It can be realized with three ancillary particles",
  "foldername": "ghz_346",
  "target_state": [
    "000",
    "111",
    "222",
    "333"
  ],
  "num_anc": 3,
  "loss_func": "cr",
  "thresholds": [
    0.25,
    0.1
  ],
  "samples": 10,
  "optimizer": "L-BFGS-B",
  "ftol": 1e-06,
  "edges_tried": 20,
  "tries_per_edge": 5
}

```

General info is given by `description`. With `foldername` one can give a custom name for the subfolder where solutions
are saved.

In the simple case of post-selected state creation with SPDC crystals, `target_state`, `num_anc` and `loss_func` are all
that are needed to define the objective of the optimization.

`thresholds` is necessary to decide whether a topological optimization step has been successful. In this case,
when `1-countrate < 0.25` and `1-fidelity < 0.1`.

### Post-Selected Quantum Gates

For an example how topological constraints on the starting graph of the optimization are dealt with, consider the
example of a post-selected CNOT(2,3) quantum gate.

```json
{
  "description": "Postselected CNOT between a qubit (control) and a qutrit (target). Two ancillary photons from SPDC.",
  "foldername": "cnot_23",
  "target_state": [
    "0000",
    "0101",
    "0202",
    "1011",
    "1112",
    "1210"
  ],
  "in_nodes": [
    0,
    1
  ],
  "out_nodes": [
    2,
    3
  ],
  "num_anc": 2,
  "loss_func": "cr",
  "thresholds": [
    0.3,
    0.1
  ],
  "samples": 10,
  "optimizer": "L-BFGS-B",
  "ftol": 1e-06,
  "edges_tried": 30,
  "tries_per_edge": 5
}
```

Here, `target_state` defines a logic table. It is defined through `in_nodes` and `out_nodes`, which entries belong to
incoming photons and which belong to outgoing photons. These definitions suffice to automatically put constraints on the
starting graph.

### Heralded Quantum Gates with Single Photon Sources

Two additional features are heralding and single photon sources. A simple example is a CNOT gate between two qubits
heralding on two ancillary detectors.

```json
{
  "description": "Heralded CNOT gate between two qubits with two single photon sources. Similar has been done: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.126.140501 could this have a better success probability?",
  "foldername": "cnot22sp",
  "target_state": [
    "0000",
    "0101",
    "1011",
    "1110"
  ],
  "num_anc": 2,
  "in_nodes": [
    0,
    1
  ],
  "out_nodes": [
    2,
    3
  ],
  "single_emitters": [
    4,
    5
  ],
  "heralding_out": true,
  "loss_func": "cr",
  "thresholds": [
    1,
    0.1
  ],
  "samples": 10,
  "optimizer": "L-BFGS-B",
  "ftol": 1e-06,
  "edges_tried": 30,
  "tries_per_edge": 5
}
```

Here, the two single photon sources given by `single_emitters` introduce further topological constraints on the starting
graph. As they have the role of ancillary photons in this case, we need to set `num_anc` accordingly. The graph
corresponding to this optimization will have a total of eight nodes. Two for incoming, two for outgoing, two for single
photon sources and two for ancillary detectors.

If `num_anc` is larger than the sum of lengths of `in_nodes` and `single_emitters`, the necessary amount of particles is
created through SPDC.

Additionaly `heralding_out` is set to `true` here. The photons corresponding to `out_nodes` will not be detected.

### Measurement

```json
{
  "description": "Measurement for three particle W state",
  "foldername": "W_measurement",
  "target_state": [
    "001",
    "010",
    "100"
  ],
  "in_nodes": [
    0,
    1,
    2
  ],
  "num_anc": 0,
  "loss_func": "cr",
  "samples": 10,
  "optimizer": "L-BFGS-B",
  "ftol": 1e-06,
  "thresholds": [
    0.3,
    0.1
  ],
  "tries_per_edge": 5,
  "edges_tried": 30
}

```

For further details, refer to the definition of the function `setup_for_target` in `main.py`.