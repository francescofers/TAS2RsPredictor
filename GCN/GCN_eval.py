import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem.EState import EState
from rdkit.Chem import Lipinski
from torch_geometric.utils.convert import from_networkx
from sklearn.preprocessing import MinMaxScaler
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import os
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GATv2Conv
from torch.nn import Linear, Dropout, ReLU
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils.mask import index_to_mask
from chembl_structure_pipeline import standardizer
from rdkit_heatmaps import mapvalues2mol
from rdkit_heatmaps.utils import transform2png
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdmolops import RemoveAllHs
from Virtuous import TestAD

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# What to investigate?
PATH = '../data/test.txt' #'PATH/TO/SMILES/FILE.txt'

# Overrides naming of molecules as molecule_N, if False the standardized SMILES will be used
NAME_OVERRIDE = False

# Activates plotting of prediction with UGrad-CAM
PLOT_UGRADCAM = False

# If TRUE checks if evaluated pairs have in-vitro verified data and overwrites prediction with Ground Truth
GT = True

# Load SMILES to predict
with open(PATH) as f:
    input_molecules = f.read().splitlines()

# CHECK if only one smile was given instead of a list
if type(input_molecules) != list:
    input_molecules = [input_molecules]

# Standardize molecules
def Std(input_smiles):
    mol_list = [Chem.MolFromSmiles(mol) for mol in input_smiles]
    std_mol_list = [standardizer.standardize_mol(mol) for mol in mol_list]
    parent = [standardizer.get_parent_mol(std_mol)[0] for std_mol in std_mol_list]
    parent_smi = [Chem.MolToSmiles(parent) for parent in parent]

    return parent_smi

molecules = Std(input_molecules)

# write a file of name Input_data.csv in raw
# the format is ready to be predicted by the model
with open('./raw/Input_data.csv', 'w') as f:
    f.write(',1,3,4,5,7,8,9,10,13,14,16,38,39,40,41,42,43,44,46,47,49,50\n')
    recs = np.zeros(22,dtype=int)
    rec2idx = {'1':0,'3':1,'4':2,'5':3,'7':4,'8':5,'9':6,'10':7,'13':8,'14':9,
               '16':10,'38':11,'39':12,'40':13,'41':14,'42':15,'43':16,'44':17,
               '46':18,'47':19,'49':20,'50':21}
    for m in molecules:
        for r in range(22):
            recs[r] = 1
            f.write(m+','+','.join(recs.astype(str))+'\n')
            recs[r] = 0

class MolDataset_exp(InMemoryDataset):
    """Class that defines the dataset for the model."""
    def __init__(self, root="", transform=None, pre_transform=None):
        self.node_attrs = ['mass','logP','mr','estate','asa','tpsa','partial_charge','degree','imp_val','nH','arom',
                           'tas1','tas3','tas4','tas5','tas7','tas8','tas9','tas10','tas13','tas14','tas16','tas38','tas39','tas40', 'tas41','tas42','tas43','tas44','tas46','tas47','tas49','tas50']
        self.num_rec = 22
        self.edge_attrs = ['is_single','is_double','is_triple','is_aromatic']
        super(MolDataset_exp, self).__init__(root, transform, pre_transform)
        print("[INFO   ] Root directory of dataset is: ", self.root)
        self.data, self.slices = torch.load(self.processed_paths[0]) # This form is required by PyTorch Geometric with version < 2.4
        
    @property
    def raw_file_names(self):
        return ['Input_data.csv'] #CHECK THIS
    
    @property
    def processed_file_names(self):
        return ['Input_data.pt'] #CHECK THIS
    
    def download(self):
        # no data needs to be downloaded
        # functions as a placeholder
        pass

    def process(self):
        '''
        The data is processed from the association matrix with missing values. Missing associations are removed, then one graph is created for each molecule.
        The information of the receptor is repeated in each node of the graph as the last 16 attributes.
        '''
       
        def one_hot_encoding(x,all_list):
            return list(map(lambda s: x == s, all_list))
        
        def mol_to_nx(mol,row,min_f,max_f):
            G = nx.Graph()
            # min/max index list
            # 0 logP
            # 1 MR
            # 2 EState
            # 3 ASA
            # 4 TPSA
            # 5 partial_charge
            mol_logP = [(i[0] - min_f[0])/(max_f[0] - min_f[0]) for i in Chem.rdMolDescriptors._CalcCrippenContribs(mol)]
            mol_mr = [(i[1] - min_f[1])/(max_f[1] - min_f[1]) for i in Chem.rdMolDescriptors._CalcCrippenContribs(mol)]
            mol_EState = [(i - min_f[2]) / (max_f[2] - min_f[2]) for i in list(EState.EStateIndices(mol))]
            mol_ASA = [(i - min_f[3]) / (max_f[3] - min_f[3]) for i in Chem.rdMolDescriptors._CalcLabuteASAContribs(mol)[0]]
            mol_TPSA = [(i - min_f[4]) / (max_f[4] - min_f[4]) for i in list(Chem.rdMolDescriptors._CalcTPSAContribs(mol))]
            for ndx, atom in enumerate(mol.GetAtoms()):
                # skip hydrogens
                if atom.GetAtomicNum() != 1:
                    G.add_node(atom.GetIdx(),
                            mass = atom.GetMass() / 126.9, #Iodium mass 0
                            logP = mol_logP[ndx], #1
                            mr = mol_mr[ndx], #2
                            estate = mol_EState[ndx], #3
                            asa = mol_ASA[ndx], #4
                            tpsa = mol_TPSA[ndx], #5
                            partial_charge = (atom.GetDoubleProp('_GasteigerCharge') - min_f[5]) / (max_f[5] - min_f[5]), # Atom partial normalized with min-max normalization 6
                            degree = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), # The degree of an atom is defined to be its number of directly-bonded neighbors 7-17
                            imp_val = one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), #Returns the number of implicit Hs on the atom 18-28
                            nH = one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), # 29-39
                            arom = atom.GetIsAromatic(), #40
                            # 41-end
                            tas1 = row['1'],
                            tas3 = row['3'],
                            tas4 = row['4'],
                            tas5 = row['5'],
                            tas7 = row['7'],
                            tas8 = row['8'],
                            tas9 = row['9'],
                            tas10 = row['10'],
                            tas13 = row['13'],
                            tas14 = row['14'],
                            tas16 = row['16'],
                            tas38 = row['38'],
                            tas39 = row['39'],
                            tas40 = row['40'],
                            tas41 = row['41'],
                            tas42 = row['42'],
                            tas43 = row['43'],
                            tas44 = row['44'],
                            tas46 = row['46'],
                            tas47 = row['47'],
                            tas49 = row['49'],
                            tas50 = row['50'],
                            )
                
            for bond in mol.GetBonds():
                # skip hydrogens
                if bond.GetBeginAtom().GetAtomicNum() != 1 and bond.GetEndAtom().GetAtomicNum() != 1:
                    G.add_edge(bond.GetBeginAtomIdx(),
                            bond.GetEndAtomIdx(),
                            is_single = int(bond.GetBondType()==Chem.rdchem.BondType.SINGLE),
                            is_double = int(bond.GetBondType()==Chem.rdchem.BondType.DOUBLE),
                            is_triple = int(bond.GetBondType()==Chem.rdchem.BondType.TRIPLE),
                            is_aromatic = int(bond.GetBondType()==Chem.rdchem.BondType.AROMATIC)
                            )
                
            return G

        # Read the input data (already in the format for prediction)
        data = pd.read_csv(self.raw_paths[0],index_col=0)

        min_max = pd.read_csv('Data/GCN/min_max.csv',index_col=0,header=0)
        min_f = min_max.iloc[0, :].values.tolist()
        max_f = min_max.iloc[1, :].values.tolist()
        data_list = []
        # Convert each row to a graph
        for smiles,row in data.iterrows():
            mol = Chem.MolFromSmiles(smiles)
            # add hydrogens
            mol = Chem.AddHs(mol)
            # try with sanitization process. If it fails, skip the molecule and notify
            try:
                Chem.SanitizeMol(mol)
            except Exception as err:
                print(f'[NOTICE ] Skipping molecule {row+1:.0f} due to sanitization error: ', err)
                continue
            # try with Gasteiger charges. If it fails, skip the molecule and notify
            try:
                ComputeGasteigerCharges(mol)
            except Exception as err:
                print(f'[NOTICE ] Skipping molecule {row+1:.0f} due to Gasteiger charge error: ', err)
                continue
            g = mol_to_nx(mol,row,min_f,max_f)
            # create the data object
            data = from_networkx(g,group_node_attrs=self.node_attrs,group_edge_attrs=self.edge_attrs)
            data.x = data.x.to(torch.float32)
            data.edge_attr = data.edge_attr.to(torch.float32)
            data.y = torch.tensor(2, dtype=torch.float32)
            # add to the list
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0]) # collate() is not included in save in PyTorch Geometric with version < 2.4

class BitterGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_rec=22):
        self.num_rec = num_rec
        super(BitterGCN, self).__init__()
        # Note: the last num_rec features are the receptor features, to be removed at the beginning and added after global pooling

        # Graph Convolutional layers
        # Define net to process graph features:
        self.relu = ReLU()
        self.conv1 = GATv2Conv(num_node_features,32,dropout=0.1,edge_dim=num_edge_features)
        self.conv2 = GATv2Conv(32,8,dropout=0.1,edge_dim=num_edge_features)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(8)

        self.dropout = Dropout(p=0.1)
        self.dropout_output = Dropout(p=0.2)
        # Output layer: takes pooled embedding and outputs koff prediction (regression)
        self.fc1 = Linear(8+num_rec, 32)          # First hidden layer         
        self.fc2 = Linear(32, 16)
        self.fc3 = Linear(16, 8)
        self.fc4 = Linear(8, 4)
         
        # Output layer for classification:
        self.output = Linear(4, 2)         # Output layer with 1 neuron for regression

        # Placeholder for Explainability
        self.gradients = None
        self.activations = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        all_node_feat, edge_index, edge_attr, batch = x, edge_index, edge_attr, batch
        # Remove receptor features from node features, and discard last column (cluster_id)
        try:
            x = all_node_feat[:,:-self.num_rec]
        except IndexError:
            x = all_node_feat[:-self.num_rec]
        
        try:
            rec_feat = torch.zeros((torch.max(batch)+1,self.num_rec))
        except TypeError:
            rec_feat = torch.zeros((1,self.num_rec))
        # now, use the information in batch to get the receptor features, one node for each graph in the batch


        for i,b in enumerate(torch.unique(batch)):
            rec_feat[i,:] = all_node_feat[batch==b,:][0,-self.num_rec:]

        rec_feat = rec_feat.to(torch.float32)
        rec_feat = rec_feat.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.sigmoid(x)
        x = self.bn2(x)

        # Save activations
        self.activations = x

        h = x.register_hook(self.activations_hook)

        # Pool node features:
        x = global_mean_pool(x, batch) # Mean pooling
        #x = self.dropout(global_mean_pool(x, batch)) # Mean pooling

        # Add receptor features
        x = torch.cat((x,rec_feat),dim=1)
        
        # Output layer
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout_output(x)
        x = self.output(x)

        return x.squeeze()

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.activations

# Define the functions to transform activations and gradients into weights per molecule
# UGrad-CAM;
def ugrad_cam(mol, activations, gradients):
    node_heat_map = []
    alphas = torch.mean(gradients, axis=0) # mean gradient for each neuron
    for n in range(activations.shape[0]): # nth node
        node_heat = (alphas @ activations[n]).item()
        node_heat_map.append(node_heat)

    node_heat_map = np.array(node_heat_map[:mol.GetNumAtoms()]).reshape(-1, 1)
    pos_node_heat_map = MinMaxScaler(feature_range=(0,1)).fit_transform(node_heat_map*(node_heat_map >= 0)).reshape(-1,)
    neg_node_heat_map = MinMaxScaler(feature_range=(-1,0)).fit_transform(node_heat_map*(node_heat_map < 0)).reshape(-1,)
    return pos_node_heat_map + neg_node_heat_map

# Function to Plot on Molecule
def plot_on_mol(mol,name,receptor,pred,activations, gradients):
    mol = Chem.MolFromSmiles(mol)
    test_mol = Draw.PrepareMolForDrawing(mol)
    test_mol = RemoveAllHs(test_mol) #IMPORTANT, RDKit PrepareMolForDrawing adds chiral Hs (bugged), so this patches the issue

    if not os.path.exists('UGradCAM'):
        os.makedirs('UGradCAM')

    if not os.path.exists(f'UGradCAM/{name}'):
        os.makedirs(f'UGradCAM/{name}')

    if not os.path.exists(f'UGradCAM/{name}/{pred}'):
        os.makedirs(f'UGradCAM/{name}/{pred}')

    # Plot UGrad-CAM
    ugrad_cam_weights = ugrad_cam(mol, activations, gradients)
    # if pred == 0:
        # ugrad_cam_weights = [x * -1 for x in ugrad_cam_weights]
    colorscale = 'bwr_r' if pred == 0 else 'bwr'
    atom_weights = ugrad_cam_weights
    bond_weights = np.zeros(len(test_mol.GetBonds()))
    limit=[max(atom_weights)*(-1),max(atom_weights)]
    canvas = mapvalues2mol(test_mol, atom_weights, bond_weights,color=colorscale,value_lims=limit)
    img = transform2png(canvas.GetDrawingText())
    img.save(f'UGradCAM/{name}/{pred}/TAS2R{receptor}.png') 

b_s = 22

data = MolDataset_exp(root='./')
test_loader =  DataLoader(data, batch_size=b_s, shuffle=False, num_workers=0, persistent_workers=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model = 'Data/GCN/GCN_model.pt'
model = BitterGCN(data.num_node_features - 22, data.num_edge_features)
model.load_state_dict(torch.load(best_model,map_location=torch.device('cpu')),strict=False)
model = model.to(device)

model.eval()

receptors = [1, 3, 4, 5, 7, 8, 9, 10, 13, 14, 16, 38, 39, 40, 41, 42, 43, 44, 46, 47, 49, 50]

# Create df for final predictions
final_results_df = pd.DataFrame(np.zeros(shape=(len(molecules),len(receptors))) ,columns=receptors)

for nmol, data in enumerate(test_loader):

    # get the entry from the dataloader
    data = data.to(device)

    # get the most likely prediction of the model
    output = model(data.x, data.edge_index, data.edge_attr, data.batch)

    output_prob = F.softmax(output, dim=1)
    preds = output_prob.argmax(dim=1)
    
    for position, receptor  in enumerate(receptors):
        # Retrieve the prediction per pair
        pred = preds[position].item()
        prob = output_prob[position,pred].item()
        final_results_df.iloc[nmol,position] = output_prob[position,1].item()
       
        if NAME_OVERRIDE:
            name = f'molecule_{nmol+1}'
        else:
            name = molecules[nmol]

        #print(f'[INFO  ] {name} - TAS2R{receptor} prediction: {pred} with probability {prob:.3f}')

        if PLOT_UGRADCAM or len(input_molecules)==1:
            # Get the gradient of the output with respect to the parameters of the model 
            if position + 1 == len(receptors):
                output[position,pred].backward()
            else:
                output[position,pred].backward(retain_graph=True)
            
            # pull the gradients out of the model
            gradients = model.get_activations_gradient()
            # get the activations of the last convolutional layer
            activations = model.get_activations(data.x)

            plot_on_mol(molecules[nmol],name,receptor,pred,activations,gradients)

final_results_df = final_results_df.round(decimals=2)
final_results_df.index = molecules
# CHECK IF ALREADY KNOWN / INCOMPLETE -> 1/0 on know and prediction on unknown
# TO REMOVE THIS CHECK set ground_truth to FALSE
if GT:
    unk = ['Unknown'] * len(molecules)
    final_results_df.insert(loc=0, column='Ground Truth', value=unk)
    # Importing dataset for checks
    tdf = pd.read_csv('Data/expanded_dataset/expanded_noOrph_22.csv', header=0, index_col=0)
    tdf.columns = tdf.columns.astype(int)

    # Check if input smiles are already present in ground truth dataset
    # Discriminate between fully known pairs, incomplete pairs and unknown
    std_known_smiles = tdf.loc[tdf.index.isin(molecules)]
    std_incomplete_smiles = std_known_smiles[std_known_smiles.isna().any(axis=1)]
    std_fullyknown_smiles = std_known_smiles.dropna(how='any')

    inc = ['Partially Known'] * len(std_incomplete_smiles.index)
    std_incomplete_smiles['Ground Truth'] = inc

    known = ['Fully Known'] * len(std_fullyknown_smiles)
    std_fullyknown_smiles['Ground Truth'] = known

    final_results_df.update(std_incomplete_smiles)
    final_results_df.update(std_fullyknown_smiles)

# CHECK if in Applicability Domain
AD_file = 'Data/GCN/AD.pkl'
check_AD = [TestAD(smi, filename=AD_file, verbose = False, sim_threshold=0.2, neighbors = 5, metric = "tanimoto") for smi in molecules]
test       = [i[0] for i in check_AD]
final_results_df.insert(loc=0, column='Check AD', value=test)

col = final_results_df.pop('Ground Truth')
final_results_df.insert(0, col.name, col)
final_results_df.insert(loc=0, column='Standardized SMILES',value=final_results_df.index)
final_results_df = final_results_df.reset_index(drop=True)

final_results_df.to_csv("GCN_output.csv", sep=",")
print('Prediction and explanation completed.')
