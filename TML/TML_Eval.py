"""
Python code to predict the association between bitter molecules and TAS2Rs using a Traditional Machine Learning (TML) model.

The code is part of the Virtuous package and is used to evaluate the association between bitter molecules and TAS2Rs using a TML model. 
The code takes as input a SMILES string or a file containing SMILES strings and returns the predicted association between the input molecules and the TAS2Rs. 

----------------
Acknowledgements
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement Action (GA No. 872181)
----------------

----------------
Version history:
- Version 1.0 - 30/05/2024
----------------

"""

__version__ = '1.0'
__author__ = 'Francesco Ferri, Marco Cannariato, Lorenzo Pallante'

import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
import ast
pd.options.mode.chained_assignment = None
import numpy as np
from Virtuous import Calc_Mordred, ReadMol, Standardize, TestAD 
import os
import argparse
import pyfiglet
from rdkit import Chem


# Defining functions
def Std(input_molecules, verbose=False, type=None):
    '''
    Standardize input SMILES using Virtuous package
    input_smiles: list of SMILES strings
    return: list of standardized SMILES strings
    '''

    # Sanitizing SMILES
    mol_list = [ReadMol(mol,verbose=verbose, type=type) for mol in input_molecules]

    # Standardizing molecule
    mol_list_std = [Standardize(mol) for mol in mol_list]
    parent_smi = [i[2] for i in mol_list_std]

    return parent_smi

def min_max_scaling(series,col,min_max):
    return (series - min_max[col].iloc[0]) / (min_max[col].iloc[1] - min_max[col].iloc[0])

def eval_tml(cpnd, verbose=False, format=None):
    ''''
    Evaluate the association between bitter molecules and TAS2Rs using a Traditional Machine Learning (TML) model
    cpnd: list of compounds (e.g. SMILES strings)
    verbose: if TRUE, the code will print additional information during the execution
    format: type of the input file (SMILES, FASTA, Inchi, PDB, Sequence, Smarts, pubchem name). If not specified, an automatic recognition of the input format will be tried
    return: dataframe with the predicted association between the input molecules and the TAS2Rs
    '''

    # CHECK if only one smile was given instead of a list
    if type(cpnd) != list:
        cpnd = [cpnd]

    # ---------------
    # Defining parameters and loading files (model, applicability domain, min-max scaling)

    # setting the paths for the code, data and src folders
    code_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.dirname(code_path)
    data_path = os.path.join(root_path, 'data')
    src_path  = os.path.join(code_path, 'src')

    # Receptors that the model is trained to evaluate over
    hTAS2R = [1, 3, 4, 5, 7, 8, 9, 10, 13, 14, 16, 38, 39, 40, 41, 42, 43, 44, 46, 47, 49, 50]

    # Applicability Domain file path
    AD_file = os.path.join(src_path, 'AD.pkl')

    # Load the final model
    model = pickle.load(open(os.path.join(src_path, 'TML_model.pkl'),'rb'))

    # Importing min and max value to min-max Mordred descriptors
    min_max = pd.read_csv(os.path.join(src_path, 'min_max_mord_fs.csv'),header=0,index_col=0)
    selected_columns = pd.read_csv(os.path.join(src_path, 'TML_sel_feature_per_iter.csv'),header=0).iloc[131,3]
    selected_columns = ast.literal_eval(selected_columns)

    # Importing dataset for checks using the inchi
    tdf = pd.read_csv(os.path.join(data_path, 'dataset.csv'), header=0, index_col=0)
    tdf.columns = tdf.columns.astype(int)
    tdf_inchi = pd.read_csv(os.path.join(data_path, 'dataset_inchi.csv'), header=0, index_col=0)

    # ---------------
    # STANDARDIZATION
    if verbose:
        print('[INFO  ] Standardizing molecules')
    std_smiles = Std(cpnd, verbose=verbose, type=format)

    # CHECK if in Applicability Domain
    if verbose:
        print('[INFO  ] Checking Applicability Domain')
    check_AD = [TestAD(smi, filename=AD_file, verbose = False, sim_threshold=0.2, neighbors = 5, metric = "tanimoto") for smi in std_smiles]

    test       = [i[0] for i in check_AD]
    # score      = [i[1] for i in check_AD]
    # sim_smiles = [i[2] for i in check_AD]

    # Calculating Mordred descriptors and selecting the important features
    if verbose:
        print('[INFO  ] Calculating descriptors')
    mord_header = Calc_Mordred(std_smiles[0])[0]
    mol_mord_df = pd.DataFrame([Calc_Mordred(mol)[1] for mol in std_smiles], index=std_smiles, columns=mord_header)
    mord_fs_df = mol_mord_df[mol_mord_df.columns.intersection(selected_columns)]

    # Min-Max scaling mordred descriptors
    for col in mord_fs_df.iloc[:,:].columns:
        mord_fs_df[col] = min_max_scaling(mord_fs_df[col],col, min_max)
    
    data = mord_fs_df

    # Creating duplicate molecule rows per receptor
    data_enc = pd.DataFrame(np.repeat(data.values, len(hTAS2R), axis=0))

    # Adding one hot encoding of each receptor
    if verbose:
        print('[INFO  ] Adding Receptor features')
    rec_col = hTAS2R * len(data.index)
    data_enc['receptor'] = rec_col
    encoded = pd.get_dummies(data_enc['receptor']).astype(int)
    data_enc = pd.concat([data_enc, encoded], axis=1)
    data_enc = data_enc.drop(columns=['receptor'])

    # Setting the right column names
    data_enc.columns = list(mord_fs_df.columns.tolist() + hTAS2R)

    # Predicting probabilities with pretrained model
    if verbose:
        print('[INFO  ] Making predictions')
    y_pred = model.predict_proba(data_enc)[:,1]

    # Wrapping up results per molecule
    results = []
    for r in range(0, len(y_pred), len(hTAS2R)):
        results.append(y_pred[r:r + len(hTAS2R)])
    
    # Wrapping up results in predicted dataframe
    if verbose:
        print('[INFO  ] Wrapping up results')
    results_df = pd.DataFrame(results,index=std_smiles,columns=hTAS2R)
    results_df = round(results_df,2)
    results_df.insert(loc=0, column='Check AD', value=test)
    results_df.insert(loc=0, column='Standardized SMILES',value=results_df.index)
    results_df = results_df.reset_index(drop=True)
    
    return results_df

def code_name():
    # print the name of the code "TAS2R Predictor" using ASCII art fun
    code_title = pyfiglet.figlet_format("TAS2R Predictor")
    # subtitle in a smaller font
    code_subtitle = "Code to predict the association between bitter molecules and TAS2Rs using a Traditional Machine Learning (TML) model"
    
    print ("\n")
    print(code_title)
    print(code_subtitle)

    # print authors and version
    print ("\n")
    print ("Version: " + __version__)
    print ("Authors: " + __author__ )
    print ("\n")


if __name__ == '__main__':

    # clear the screen
    os.system('clear')

    # --- Parsing Input ---
    parser = argparse.ArgumentParser(description=code_name(), )
    # the user cannot provide both a compound and a file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-c','--compound',help="query compound (allowed file types are SMILES, FASTA, Inchi, PDB, Sequence, Smarts, pubchem name)",default=None)
    group.add_argument('-f','--file',help="text file containing the query molecules",default=None)
    parser.add_argument('-t', '--type', help="type of the input file (SMILES, FASTA, Inchi, PDB, Sequence, Smarts, pubchem name). If not specified, an automatic recognition of the input format will be tried", default=None)
    parser.add_argument('-d','--directory',help="name of the output directory",default=None)
    parser.add_argument('-v','--verbose',help="If present, the code will print additional information during the execution",default=False, action='store_true')
    args = parser.parse_args()

    # check if the input is a file or a single compound
    if args.compound:
        cpnd = args.compound
    elif args.file:
        PATH = os.path.abspath(args.file)
        with open(PATH) as f:
            cpnd = f.read().splitlines()
    else:
        print('[ERROR ] No input provided. Please provide a molecule with the -c option or a file containing a list of molecules with the -f option')
        exit()

    # check if the output directory is provided and if it exists
    if args.directory:
        if not os.path.exists(args.directory):
            os.makedirs(args.directory)
        output_path = os.path.abspath(args.directory)
    else:
        output_path = os.path.join(os.getcwd(), 'results')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    # --- Evaluating SMILES ---
    f_df = eval_tml(cpnd, verbose=args.verbose, format=args.type)
    f_df.to_csv(os.path.join(output_path, 'TML_output.csv'),sep=',')
    print('[DONE  ] Prediction task completed.')

    if args.verbose:
        print(f"[INFO  ] Output saved in {output_path}/TML_output.csv\n\n")
        print(f_df)