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

# setting the paths
code_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(code_path)
data_path = os.path.join(root_path, 'data')
src_path = os.path.join(code_path, 'src')
PATH = os.path.join(data_path, 'test.txt') #'PATH/TO/SMILES/FILE.txt'


GT = True # TRUE for Ground Truth Check

# Load SMILES to predict
with open(PATH) as f:
    smiles = f.read().splitlines()

# Applicability Domain file path
AD_file = os.path.join(src_path, 'AD.pkl')

# Load the final model
model = pickle.load(open(os.path.join(src_path, 'TML_model.pkl'),'rb'))

# Importing min and max value to min-max Mordred descriptors
min_max = pd.read_csv(os.path.join(src_path, 'min_max_mord_fs.csv'),header=0,index_col=0)
selected_columns = pd.read_csv(os.path.join(src_path, 'TML_sel_feature_per_iter.csv'),header=0).iloc[131,3]
selected_columns = ast.literal_eval(selected_columns)

# Importing dataset for checks
tdf = pd.read_csv(os.path.join(data_path, 'dataset.csv'), header=0, index_col=0)
tdf.columns = tdf.columns.astype(int)

# Receptors that the model is trained to evaluate over
hTAS2R = [1, 3, 4, 5, 7, 8, 9, 10, 13, 14, 16, 38, 39, 40, 41, 42, 43, 44, 46, 47, 49, 50]

def Std(input_smiles):

    # Sanitizing SMILES
    mol_list = [ReadMol(mol,verbose=False) for mol in input_smiles]

    # Standardizing molecule
    mol_list_std = [Standardize(mol) for mol in mol_list]
    parent_smi = [i[2] for i in mol_list_std]

    return parent_smi

def min_max_scaling(series,col):
    return (series - min_max[col].iloc[0]) / (min_max[col].iloc[1] - min_max[col].iloc[0])

def eval_smiles(smiles, ground_truth):

    # CHECK if only one smile was given instead of a list
    if type(smiles) != list:
        smiles = [smiles]

    # STANDARDIZATION
    print('[INFO  ] Standardizing molecules')
    std_smiles = Std(smiles)

    # CHECK IF ALREADY KNOWN / INCOMPLETE -> 1/0 on know and prediction on unknown
    # TO REMOVE THIS CHECK set ground_truth to FALSE
    if ground_truth:
        # Firstly, we check if input smiles are already present in ground truth dataset
        std_known_smiles = tdf.loc[tdf.index.isin(std_smiles)]
        # If the molecule is present BUT there are unknow associations we pass them through evaluation
        # Molecules with all known association will NOT be passed to evaluation and ground truth will be added at the end
        std_incomplete_smiles = std_known_smiles[std_known_smiles.isna().any(axis=1)]
        std_clean_smiles = [x for x in std_smiles if x not in std_known_smiles.index.astype(str)[:]]
        std_clean_smiles += std_incomplete_smiles.index.astype(str)[:].to_list()
        std_fullyknown_smiles = std_known_smiles.dropna(how='any')

        # If all molecules passed are already present and fully known then simply return ground truth
        if len(std_fullyknown_smiles.index) == len(std_smiles):
            comp = ['Fully Known'] * len(std_fullyknown_smiles.index)
            std_fullyknown_smiles['Ground Truth'] = comp
            return round(std_fullyknown_smiles,2)
    else:
        std_clean_smiles = std_smiles
    
    # CHECK if in Applicability Domain
    print('[INFO  ] Checking Applicability Domain')
    check_AD = [TestAD(smi, filename=AD_file, verbose = False, sim_threshold=0.2, neighbors = 5, metric = "tanimoto") for smi in std_clean_smiles]
    test       = [i[0] for i in check_AD]
    score      = [i[1] for i in check_AD]
    sim_smiles = [i[2] for i in check_AD]

    # Calculating Mordred descriptors and selecting the important features
    print('[INFO  ] Calculating descriptors')
    mord_header = Calc_Mordred(std_clean_smiles[0])[0]
    mol_mord_df = pd.DataFrame([Calc_Mordred(mol)[1] for mol in std_clean_smiles], index=std_clean_smiles, columns=mord_header)
    mord_fs_df = mol_mord_df[mol_mord_df.columns.intersection(selected_columns)]

    # Min-Max scaling mordred descriptors
    for col in mord_fs_df.iloc[:,:].columns:
        mord_fs_df[col] = min_max_scaling(mord_fs_df[col],col)
    
    data = mord_fs_df

    # Creating duplicate molecule rows per receptor
    data_enc = pd.DataFrame(np.repeat(data.values, len(hTAS2R), axis=0))

    # Adding one hot encoding of each receptor
    print('[INFO  ] Adding Receptor features')
    rec_col = hTAS2R * len(data.index)
    data_enc['receptor'] = rec_col
    encoded = pd.get_dummies(data_enc['receptor']).astype(int)
    data_enc = pd.concat([data_enc, encoded], axis=1)
    data_enc = data_enc.drop(columns=['receptor'])

    # Setting the right column names
    data_enc.columns = list(mord_fs_df.columns.tolist() + hTAS2R)

    # Predicting probabilities with pretrained model
    print('[INFO  ] Making predictions')
    y_pred = model.predict_proba(data_enc)[:,1]

    # Wrapping up results per molecule
    results = []
    for r in range(0, len(y_pred), len(hTAS2R)):
        results.append(y_pred[r:r + len(hTAS2R)])
    
    # Wrapping up results in predicted dataframe
    print('[INFO  ] Wrapping up results')
    results_df = pd.DataFrame(results,index=std_clean_smiles,columns=hTAS2R)
    results_df = round(results_df,2)

    # Adding molecules with partial ground truth if present
    unk = ['Absent'] * len(results_df.index)
    results_df['Ground Truth'] = unk
    results_df.insert(loc=0, column='Check AD', value=test)
    if ground_truth:
        inc = ['Partially Known'] * len(std_incomplete_smiles.index)
        std_incomplete_smiles['Ground Truth'] = inc

        results_df.update(std_incomplete_smiles)
        

        # Adding input smiles that already have ground thruth for every receptor
        comp = ['Fully Known'] * len(std_fullyknown_smiles.index)
        std_fullyknown_smiles['Ground Truth'] = comp
        test_dummy = [True] * len(std_fullyknown_smiles.index)
        std_fullyknown_smiles.insert(loc=0, column='Check AD', value=test_dummy)

    # Wrapping up results in final dataframe
        final_results_df = pd.concat([results_df,std_fullyknown_smiles],axis=0)
        col = final_results_df.pop('Ground Truth')
        final_results_df.insert(0, col.name, col)
    else:
        final_results_df = results_df
        final_results_df = final_results_df.drop(columns=['Ground Truth'])
    
    final_results_df.insert(loc=0, column='Standardized SMILES',value=final_results_df.index)
    final_results_df = final_results_df.reset_index(drop=True)
    

    return final_results_df

f_df = eval_smiles(smiles,ground_truth=GT)
f_df.to_csv('TML_output.csv',sep=',')
print('[DONE  ] Prediction task completed.')