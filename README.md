# TAS2RsPredictor

Both the models take as input a .txt file containing the SMILES of the query molecules.
Change the PATH variable in the evaluation scripts to input your own SMILES.

## Setting up the environment

### 1. Create a new conda environment
```
conda create -n TAS2RPred python=3.9
```

### 2. Activate the environment
```
conda activate TAS2RPred
```

### 3. Install the basic packages
```
conda install --yes -c conda-forge pandas=2.1.3 scipy=1.11.4 matplotlib=3.8.2
```

### 4. Install VIRTUOUS library dependecies

ChEMBL pipeline and RDKit
```
conda install --yes -c conda-forge rdkit=2023.09.4 chembl_structure_pipeline=1.2.0
```

Mordred
``` 
conda install --yes -c mordred-descriptor mordred=1.2.0
```

Others
``` 
conda install --yes -c conda-forge tqdm=4.66.1 knnimpute=0.1.0 joblib=1.3.2 cython=3.0.10 scikit-learn=1.3.2 xmltodict=0.13.0 pyfiglet
pip install pyenchant 
```

### TML packages

### 5.a CatBoost
``` 
conda install --yes -c conda-forge catboost=1.2.5
```

### GCN packages

### 5.b Pytorch, PyG, NetworkX and rdkit_heatmaps

If CUDA is not available on your OS:
``` 
conda install --yes pytorch::pytorch torchvision torchaudio -c pytorch
```

If your OS supports CUDA:
```
conda install --yes pytorch torchvision torchaudio pytorch-cuda=<CUDA_VERSION> -c pytorch
```
Replace <b><CUDA_VERSION></b> with your installed CUDA driver version number (e.g. pytorch-cuda=12)

Pyg
```
conda install --yes pyg -c pyg
```

NetworkX
```
conda install --yes -c conda-forge networkx=3.2.1
```

rdkit_heatmaps
```
pip install git+https://github.com/c-feldmann/rdkit_heatmaps
```

## Running the training and evaluation

### Clone the GitHub Repository

```
git clone https://github.com/francescofers/TAS2RsPredictor
```

### TML model

### 1. Train the Traditional Machine Learning model

```
python TML/TML_Train.py
```

### 2. Carry out the classification task with the trained Traditional Machine Learning model

```
python TML/TML_Eval.py
```

### GCN model

### 1. Train the Graph Convolutional Neural Network model

```
python GCN/GCN_Train.py
```

### 2. Carry out the classification task with the trained Convolutional Neural Network model
```
python GCN/GCN_Eval.py
```

---

Example Notebooks are available in both TML and GCN directory.




# Troubleshooting
On MacOS, if you encounter some errors related to pyenchants, you can try this:
```
brew install enchant
export PYENCHANT_LIBRARY_PATH=$(brew --prefix enchant)/lib/libenchant-2.dylib
```
