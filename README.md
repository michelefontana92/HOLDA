# Overview
This is the official code for the paper [A new approach for cross-silo federated learning and its privacy risks](https://ieeexplore.ieee.org/document/9647753)

# HOLDA
![holda logo](./images/holda.png)
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
It describes the task that has to be solved. 
The node has to contain the following tags:
-`n_features` : the number of features of the dataset
-`n_classes` : the number of target classes of the dataset
-`target` : the name of the target column of the dataset

An example is the following one:
```
<task>
        <n_features> 103 </n_features>
        <n_classes>2</n_classes>
        <target>income</target>
</task>
```
Here, we are saying that our dataset has 103 attributes, 2 possible classes and the target column is named `income`.

### The `model` node
It describes how to build the model that has to be trained during the execution of `HOLDA`.
The node has to contain the following tags:
-`model_fn` : the function that has to be used to instantiate the model (it needs the whole path).
-`params` : this tag has one child for each parameters needed by the `model_fn` function. The tag of the child has to be the name of the correspondig parameter.

An example is the following one.
Assume that we construct the model using this function

```
def create_simple_net(hidden_1, dropout, output, input):
    net = SimpleNet(hidden_1, dropout, output, input)
    return net
```

where `SimpleNet` is defined as:

```
class SimpleNet(nn.Module):
    def __init__(self, hidden_1, dropout, output, input):
        super(SimpleNet, self).__init__()

        self.fc1 = nn.Linear(input, hidden_1)
        self.fc2 = nn.Linear(hidden_1, output)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x
```

Then, the following tag,
```
<model>
  <model_fn>models.nn.create_simple_net</model_fn>
   <params>
      <input>103</input>
      <hidden_1>200</hidden_1>
      <dropout>0.2</dropout>
      <output>2</output>
   </params>
</model>
```
is just like calling `models.nn.simple_net(input=103,hidden_1=200,dropout=0.2,output=2)`

This structure is kept the same whenever we need to perform any function invocation during the trainign algorithm.
In particular this is the schema adopted for describing the loss function and the local optimizer.

### The `metrics` node
It describes what are the metrics that have to be computed during the training, other than the loss function.
The system currently supports the following metrics: 
- `accuracy`
- `precision`
- `recall`
- `f1`

I highly recommend to keep at least the `f1` metric, since it is the one used to choose the best generalizing model.

```
<metrics>
  <metric>accuracy</metric>
  <metric>precision</metric>
  <metric>recall</metric>
  <metric>f1</metric>
</metrics>
```

### The `architecture` node
It describes the structure of the federation.
