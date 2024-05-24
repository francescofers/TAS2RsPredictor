# TAS2RsPredictor
---

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

3. Install the reuqired packages
```
pip install -r requirements.txt
```

## Running the training scripts


### TML model

1. Training the Traditional Machine Learning model

```
python TML/TML_Train.py
```

2. Evaluation

```
python TML/TML_Eval.py
```

Change the PATH variable in evaluation scripts to input your own SMILES

### GCN model

1. Training

```
python GCN/GCN_Train.py
```

2. Evaluation
```
python GCN/GCN_Eval.py
```


---

Example Notebooks are available in both TML and GCN directory
