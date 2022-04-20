# Overview
This is the official code for the paper [A new approach for cross-silo federated learning and its privacy risks](https://ieeexplore.ieee.org/document/9647753)

# HOLDA
TBD
# Usage
The main entry point of the system is the `holda.py` file.

`python holda.py --file architectures/holda_not_hier.xml --pers`

The command shown above executes the following steps:
  - It reads the .xml file passed in input (--file), which describes the architecture of the federation that has to be adopted during the training procedure. More details about the structure of the .xml file is provided in a following section.
  -  It executes `HOLDA` according to the information provided in the .xml file.
  -  At the end of the main training procedure, it personalises the (intermediate / local) models, throughout a fine-tuning procedure.
  -  The final models are located into the file, whose path is specified in the xml file. (tag: `ckpt\_best`)

This repository contains two simple working examples

