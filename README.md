# TAS2RsPredictor

Both the models take as input a .txt file containing the SMILES of the query molecules
Change the PATH variable in the evaluation scripts to input your own SMILES

## Setting up the environment

1. Create a new conda environment
```
conda create -n TAS2RPred python=3.9 pip
```

2. Activate the environment
```
conda activate TAS2RPred
```

3. Install the required packages
```
pip install -r requirements.txt
```

## Running the training scripts


### TML model

1. Train the Traditional Machine Learning model

```
python TML/TML_Train.py
```

2. Carry out the classification task with the trained Traditional Machine Learning model

```
python TML/TML_Eval.py
```

### GCN model

1. Train the Graph Convolutional Neural Network model

```
python GCN/GCN_Train.py
```

2. Carry out the classification task with the trained Convolutional Neural Network model
```
python GCN/GCN_Eval.py
```

---

Example Notebooks are available in both TML and GCN directory
