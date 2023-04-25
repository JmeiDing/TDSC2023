# TDSC2023
This repo is a python implementation of user abnormal behavior detection using graph neural networks.

## Requirements
### Required Packages
* **python**3 or above
* **PyTorch**1.0.0
* **numpy**1.18.2
* **sklearn** for model evaluation

Run the following script to install the required packages.
```
pip install --upgrade pip
pip install torch==1.5.1
pip install numpy==1.24.2
pip install scikit-learn
```


### Dataset
For each dataset, we randomly pick 80% of the data as the training set while the remaining are utilized for the testing set. 
In the comparison, metrics accuracy, recall, precision, F1-score, TPR, FPR and AUC are all involved.
The original dataset can be found in [CERT](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset)

The normalized graph data can be found in [DATA](https://drive.google.com/file/d/1lKbeIeQ1EkjHzhgmLVYDfWrdG0Qrqa19/view?usp=sharing),
each category of abnormal behaviour dataset consists of three sub-datasets, as follows:
* UBA1_0 is session graph data extracted from the original data
* UBA1_1g is the augmented session graph data
* UBA1_3ss is the association graph data

#### Dataset structure in this project
The graph data of AB-I in the following structure respectively，AB-II and AB-III is the same as AB-I.
```
${TDSC2023}
├── data
    ├── demo
    │    └── graph_A.txt
    │    └── graph_indicator.txt
    │    └── graph_labels.txt
    │    └── node_labels.txt
    │    └── node_attributes.txt
	├── rawdata
	      ├── UBA1_3ss
	    		  └── generation_data_batch
	    		  └── session_node
	    		  └── user_0
	    		  └── user_1

```
* graph_A.txt (m lines):sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)
* graph_indicator.txt (n lines):column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i
* graph_labels.txt (N lines):class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i
* node_labels.txt (n lines)column vector of node labels,
    the value in the i-th line corresponds to the node with node_id i
* node_attributes.txt (n lines):matrix of node attributes,
    the comma seperated values in the i-th line is the attribute vector of the node with node_id i
* user_0 file:activity-Level feature dataset for normal users
* user_1 file:activity-Level feature dataset for abnormal users
* session_node file:session feature dataset for all users
* generation_data_batch file:graph data for each user

### Code Files
The tools for extracting graph features (vectors) are as follows:
```
${TDSC2023}
├── graph_construction
			├── user1
			│		└── 1.build_folder.py
			│		└── 2.session_feature.py
			│		└── 3.graph_data.py 
			│		└── 4.all_graph.py
			├── user2
			│		└── 1.build_folder.py
			│		└── 2.session_feature.py
			│		└── 3.graph_data.py 
			│		└── 4.all_graph.py
			├── user3
					└── 1.build_folder.py
					└── 2.session_feature.py
					└── 3.graph_data.py 
					└── 4.all_graph.py
```

```
AOD-ASG.py
```
* All graph data are automatically split and stored.
* Find the relationships between activity.
* Extract all user behavior data into the corresponding graph consisting of nodes and edges.
* association graph into vectors.


## Running project
* To run program, use this command: python AOD-ASG.py.
* In addition, you can use specific hyperparameters to train the model. All the hyper-parameters can be found in `parser.py`.

Examples:
```shell
python AOD-ASG.py --dataset ./data/graphdata
python AOD-ASG.py --dataset ./data/graphdata --model gcn --n_hidden 192 --lr 0.001 -f 64,64,64 --dropout 0.1 --vector_dim 100 --epochs 50 --lr_decay_steps 10,20 
```
Using script：
Repeating 10 times for different seeds with `train.sh`.
```shell
for i in $(seq 1 10);
do seed=$(( ( RANDOM % 10000 )  + 1 ));
python AOD-ASG.py --model gcn --seed $seed | tee logs/smartcheck_"$i".log;
done
```

### Reference
1. The code borrows from [graph_unet](https://github.com/bknyaz/graph_nn)
2. Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017
