# TAS2RsPredictor

## Setting up the environment
---
'''
conda create -n TAS2RPred python=3.9 pip
'''

'''
conda activate TAS2RPred
'''

'''
pip install -r requirements.txt
'''

## Running the training scripts
---

### TML model

1. Training

'''
python TML/TML_Train.py
'''

2. Evaluation

'''
python TML/TML_Eval.py
'''

Change the PATH variable in evaluation scripts to input your own SMILES

### GCN model

1. Training

'''
python GCN/GCN_Train.py
'''

2. Evaluation
'''
python GCN/GCN_Eval.py
'''

Change the PATH variable in evaluation scripts to input your own SMILES
