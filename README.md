# Overview
This is the official code for the paper [A new approach for cross-silo federated learning and its privacy risks](https://ieeexplore.ieee.org/document/9647753)

# HOLDA
![ezcv logo](./images/holda.png)
# Usage
The main entry point of the system is the `holda.py` file.

`python holda.py --file architectures/holda_not_hier.xml --pers`

The command shown above executes the following steps:
  - It reads the .xml file passed in input (--file), which describes the architecture of the federation that has to be adopted during the training procedure. More details about the structure of the .xml file is provided in a following section.
  -  It executes `HOLDA` according to the information provided in the .xml file.
  -  At the end of the main training procedure, it personalises the (intermediate / local) models, throughout a fine-tuning procedure (if the --pers flag is activated).
  -  The final models are located into the file, whose path is specified in the xml file. (tag: `ckpt_best`)

This repository contains two simple working examples, located into the `scripts` folder.
  - `execute_example.sh` : it executes a non-hierarchical training on the adult dataset (located into `examples/dataset` folder). The federation comprises one server and 4 clients, directly connected to the main server. Before starting the training, it creates the xml file, starting from the baseline architecture, stored into the  `architecture/baselines/example_not_hier_holda_base.xml` file. The final architecture is stored into `architecture/holda/holda_not_hier.xml`

  - `execute_hier_example.sh` : it executes a hierarchical training on the adult dataset (located into `examples/dataset` folder). The federation comprises one server, 2 proxies and 4 clients, like the one depicted in the picture. Before starting the training, it creates the xml file, starting from the baseline architecture, stored into the  `architecture/baselines/example_hier_holda_base.xml` file. The final architecture is stored into `architecture/holda/holda_hier.xml`

  
## Structure of the xml file
This file contains the description of:
  - the architecture of the federation that has to be adopted in order to execute the training procedure
  - the metadata of each single node of the federation
  - the training parameters for each node (server, proxy, client)

At the high level, the file has to include the following tags:
- `root_node`: it is the root of the tree
- `task`: It describes the task that has to be solved.
- `model`: It describes how to build the model that has to be trained during the execution of `HOLDA`.
- `metrics` : It describes what are the metrics that have to be computed during the training.
- `setting` : It specifies some setting parameters.
- `architecture` : It describes the structure of the federation.
### The `task` node


### The `model` node

### The `metrics` node

### The `architecture` node
