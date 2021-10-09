# Sequential algorithmic modification with test data reuse

# Installation
We use `pip` to install things into a python virtual environment. Refer to `requirements.txt` for package requirements.
The R code for computing significance thresholds requires installation of the R package `mvtnorm`.
We use `nestly` + `SCons` to run simulations.

# File descriptions

`generate_data.py` -- Generate data.

`create_modeler.py` -- Creates an adaptive model developer as specified by the `--simulation` argument (options are `adversary` and `online`).

`create_mtp_mechanism.py` -- Create the multiple testing procedure for approving modifications.

`main.py` -- Given simulated test and training data, the approval mechanism (i.e. the multiple hypothesis testing procedure), and the adaptive model developer, this will simulate the approval procedure.

# Reproducing simulation results

The `simulation_adversary` folder contains the first set of simulations with an "adversarial" model developer who proposes deleterious modifications. The `simulation_reuse` folder contains the second set of simulations where the model developer generally proposes beneficial modifications. To run the simulations, run `scons <simulation_folder_name>`.
