# TAS2RsPredictor
Official repo of the TAS2Rs Predictor developed in the framework of the EU-funded VIRTUOUS project

[![Virtuous button][Virtuous_image]][Virtuous link]

[Virtuous_image]: https://virtuoush2020.com/wp-content/uploads/2021/02/V_logo_h.png
[Virtuous link]: https://virtuoush2020.com/


### Repo Structure
The repository is organized in the following folders:

- **TML/**
Including the Traditional Machine Learning model
    - TML_Train.py: code to train your own model using a novel dataset
    - TML_Eval.py: code to evaluate the model trained on our dataset
    - Virtuous.py: library for general processing functions

- **GCN/**
Including the Graph Convolutional Neural Network model
    - GCN_Train.py: code to train your own model using a novel dataset
    - GCN_Eval.py: code to evaluate the model trained on our dataset
    - Virtuous.py: library for general processing functions

- **data/**
Collecting the training and the test sets of the model and an example txt file to run the code

- **notebooks/**
Including example notebooks to run the code and understand the workflow

### Authors
1. [Francesco Ferri](https://github.com/francescofers)
2. [Marco Cannariato](https://github.com/marcocannariato)
2. [Lorenzo Pallante](https://github.com/lorenzopallante)

----------------
## Prerequisites
----------------
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

### 5. TML packages - CatBoost
``` 
conda install --yes -c conda-forge catboost=1.2.5
```

### 6. GCN packages - Pytorch, PyG, NetworkX and rdkit_heatmaps

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

### 7. Others

notebook widgets
```
conda install ipywidgets
```

**Troubleshooting**
>On MacOS, if you encounter some errors related to pyenchants, you can try this:
```
brew install enchant
export PYENCHANT_LIBRARY_PATH=$(brew --prefix enchant)/lib/libenchant-2.dylib
```


### 8. Clone the GitHub Repository

```
git clone https://github.com/francescofers/TAS2RsPredictor
```

----------------
## How to run the code
----------------

### TML model

The main code to run the Traditional Machine Learning model is `TML_Eval.py` within the TML/ folder.

To learn how to run, just type:

    python TML/TML_Eval.py --help

And this will print the help message of the program:

    usage: TML_Eval.py [-h] (-c COMPOUND | -f FILE) [-t TYPE] [-d DIRECTORY] [-v] [-g]

    optional arguments:
    -h, --help            show this help message and exit
    -c COMPOUND, --compound COMPOUND
                            query compound (allowed file types are SMILES, FASTA, Inchi, PDB, Sequence, Smarts, pubchem name)
    -f FILE, --file FILE  text file containing the query molecules
    -t TYPE, --type TYPE  type of the input file (SMILES, FASTA, Inchi, PDB, Sequence, Smarts, pubchem name). If not specified, an automatic recognition of the input format will be
                            tried
    -d DIRECTORY, --directory DIRECTORY
                            name of the output directory
    -v, --verbose         Set verbose mode
    -g, --ground_truth    Set to TRUE if you want to check if the input SMILES are already present in the ground truth dataset

Baically, the user can run the code by providing a SMILES string (-c input) or a file containing SMILES strings (-f input). The code will return the prediction of the model in the output folder defined using the -d option (folder 'results' otherwise).


###### If you want to train the TML model on your dataset, you can use the `TML_Train.py` script.


### GCN model


