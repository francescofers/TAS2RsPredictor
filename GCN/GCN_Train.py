import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.EState import EState
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import global_mean_pool, GATv2Conv
from torch.nn import Linear, Dropout, ReLU
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import os, shutil
from time import time
from sklearn.metrics import pairwise_distances, roc_curve, auc

import warnings
warnings.simplefilter("ignore", UserWarning)

class MolDataset(InMemoryDataset):
    """Class that defines the dataset for the model."""
    def __init__(self, root="", transform=None, pre_transform=None):
        self.node_attrs = ['mass','logP','mr','estate','asa','tpsa','partial_charge','degree','imp_val','nH','arom',
                           'tas1','tas3','tas4','tas5','tas7','tas8','tas9','tas10','tas13','tas14','tas16','tas38','tas39','tas40', 'tas41','tas42','tas43','tas44','tas46','tas47','tas49','tas50']
        self.num_rec = 22
        self.edge_attrs = ['is_single','is_double','is_triple','is_aromatic']
        super(MolDataset, self).__init__(root, transform, pre_transform)
        print("[INFO   ] Root directory of dataset is: ", self.root)
        self.data, self.slices = torch.load(self.processed_paths[0]) # This form is required by PyTorch Geometric with version < 2.4
        
    @property
    def raw_file_names(self):
        return ['BitterData.csv']
    
    @property
    def processed_file_names(self):
        return ['BitterData.pt']
    
    def download(self):
        # no data needs to be downloaded
        # functions as a placeholder
        pass

    def process(self):
        '''
        The data is processed from the association matrix with missing values. Missing associations are removed, then one graph is created for each molecule.
        The information of the receptor is repeated in each node of the graph as the last 22 attributes.
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
                            mass = atom.GetMass() / 126.9, #Iodium mass
                            logP = mol_logP[ndx],
                            mr = mol_mr[ndx],
                            estate = mol_EState[ndx],
                            asa = mol_ASA[ndx],
                            tpsa = mol_TPSA[ndx],
                            partial_charge = (atom.GetDoubleProp('_GasteigerCharge') - min_f[5]) / (max_f[5] - min_f[5]), # Atom partial normalized with min-max normalization
                            degree = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), # The degree of an atom is defined to be its number of directly-bonded neighbors
                            imp_val = one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), #Returns the number of implicit Hs on the atom
                            nH = one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                            arom = atom.GetIsAromatic(),
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
        
        def Get_tanimoto_matrix(df):
            mol_fps = [np.array(GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles),2,nBits=1024,useChirality=True),dtype=bool) for smiles,_ in df.iterrows()]
            mol_fps = np.vstack(mol_fps)
            tanimoto_matrix = pairwise_distances(mol_fps, metric='jaccard')
            return tanimoto_matrix
        
        def Cluster(tanimoto_matrix):
            # Selecting best n_clusters
            range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25, 30]
            silhouette_lists = []

            for n_clusters in range_n_clusters:
                aggls = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
                aggls = aggls.fit(tanimoto_matrix)
                labels = aggls.labels_
                kdf = pd.DataFrame()
                kdf['aggls'] = labels
                min_nelem = min(kdf['aggls'].value_counts())
                if min_nelem < 10:
                    silhouette_avg = 0
                else:
                    silhouette_avg = silhouette_score(tanimoto_matrix, labels, metric='precomputed')
                
                silhouette_lists.append(silhouette_avg)
            best_n_clusters = range_n_clusters[np.argmax(silhouette_lists)]

            # Apply AgglomerativeClustering with best n_clusters for receptor
            aggls = AgglomerativeClustering(n_clusters=best_n_clusters, metric='precomputed', linkage='complete')
            aggls = aggls.fit(tanimoto_matrix)
            labels = aggls.labels_
            return labels

        data_list = []
        # Read the data with smiles and associations (with missing values!!)
        raw_data = pd.read_csv(self.raw_paths[0],index_col=0,header=0)
        # unpivot the table and remove missing values
        data = pd.melt(raw_data.assign(ligand=raw_data.index), id_vars='ligand', var_name='receptor',
                       value_name='association').dropna(axis=0, how='any').drop_duplicates(keep="first").set_index(['ligand'])

        encoded_rec = pd.get_dummies(data['receptor']).astype(int)
        data = pd.concat([encoded_rec,data['association']],axis=1)

        # cluster the ligands using Agglomerative clustering of tanimoto similarity of morgan fingerprints, then put the cluster as feature for the stratified split
        try:
            test = np.loadtxt(self.processed_paths[0]+'.cluster_ids')
        except FileNotFoundError:
            cluster_ids = Cluster(Get_tanimoto_matrix(data))
            np.savetxt(self.processed_paths[0]+'.cluster_ids', cluster_ids, delimiter=',', fmt='%d')

        # Pre calculate min and max values of features for min-max normalization
        logPs, mrs, estates, asas, tpsas, partial_charges = [], [], [], [], [], []
        for smiles,row in data.iterrows():
            mol = Chem.MolFromSmiles(smiles)

            logPs.extend([i[0] for i in Chem.rdMolDescriptors._CalcCrippenContribs(mol)])
            mrs.extend([i[1] for i in Chem.rdMolDescriptors._CalcCrippenContribs(mol)])
            estates.extend(list(EState.EStateIndices(mol)))
            asas.extend([i for i in Chem.rdMolDescriptors._CalcLabuteASAContribs(mol)[0]])
            tpsas.extend(list(Chem.rdMolDescriptors._CalcTPSAContribs(mol)))

            # add hydrogens
            mol = Chem.AddHs(mol)
            try:
                ComputeGasteigerCharges(mol)
                for atom in mol.GetAtoms():
                    # skip hydrogens
                    if atom.GetAtomicNum() != 1:
                        partial_charges.append(atom.GetDoubleProp('_GasteigerCharge'))
            except Exception as err:
                continue

        min_f = [min(logPs), min(mrs), min(estates), min(asas), min(tpsas), min(partial_charges)]
        max_f = [max(logPs), max(mrs), max(estates), max(asas), max(tpsas), max(partial_charges)]

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
            data.y = torch.tensor([row['association']], dtype=torch.float32)
            data.x = data.x.to(torch.float32)
            data.edge_attr = data.edge_attr.to(torch.float32)
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
        self.conv1 = GATv2Conv(num_node_features,32,dropout=DROPOUT,edge_dim=num_edge_features)
        self.conv2 = GATv2Conv(32,8,dropout=DROPOUT,edge_dim=num_edge_features)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(8)

        self.dropout = Dropout(p=DROPOUT)
        self.dropout_output = Dropout(p=0.2)
        # Output layer: takes pooled embedding and outputs koff prediction (regression)
        self.fc1 = Linear(8+num_rec, 32)          # First hidden layer         
        self.fc2 = Linear(32, 16)
        self.fc3 = Linear(16, 8)
        self.fc4 = Linear(8, 4)
         
        # Output layer for classification:
        self.output = Linear(4, 2) 

        # Placeholder for Explainability
        self.gradients = None
        self.activations = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        all_node_feat, edge_index, edge_attr, batch = x, edge_index, edge_attr, batch
        # Remove receptor features from node features, and discard last column (cluster_id)
        x = all_node_feat[:,:-self.num_rec]
        
        rec_feat = torch.zeros((torch.max(batch)+1,self.num_rec))
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

        # Register the hook
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

# Creating Early stopper class
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.stuck_counter = 0
        self.min_validation_loss = float('inf')

    # Function to avoid overfitting
    def early_stop(self, validation_loss, epoch):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta) and epoch >= 50:
            self.counter += 1
            if self.counter >= self.patience:
                self.counter = 0
                self.min_validation_loss = float('inf')
                return True
        return False
    
    # Function to kill fold if it's not learning
    def stuck(self, validation_loss):
        if round(validation_loss,3) != round(self.min_validation_loss,3):
            self.stuck_counter = 0
        elif round(validation_loss,3) == round(self.min_validation_loss,3):
            self.stuck_counter += 1
            if self.stuck_counter >= 20 and validation_loss > 0.5:
                self.stuck_counter = 0
                self.min_validation_loss = float('inf')
                return True
        return False

def get_rates(y_test,probs):
    fpr, tpr, thresholds = roc_curve(y_test, probs, drop_intermediate=False)
    auc_value = auc(fpr,tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return fpr,tpr,auc_value,optimal_threshold

dataset = MolDataset(root='./src/train')

NUM_REC = 22
N_EPOCHS = 200
LR = 0.0005
REDUCE_LR_ON_PLATEAU = True
BATCH_SIZE = 16
NUM_WORKERS = 0
DROPOUT = 0.1
WD = 0.0005
TEST_SIZE = 0.2
OUTPUTDIR = os.getcwd() + os.sep + 'Models'

# split the dataset into train and test
kfold = StratifiedKFold(n_splits=10,random_state=42,shuffle=True)
if os.path.exists(OUTPUTDIR):
    shutil.rmtree(OUTPUTDIR)
os.makedirs(OUTPUTDIR)

cluster_ids = np.loadtxt(dataset.processed_paths[0]+'.cluster_ids', delimiter=',', dtype=int)
train_indexes, test_indexes = train_test_split(np.array(dataset.indices()), test_size=TEST_SIZE, random_state=42,stratify=cluster_ids)
# isolate train and test datasets
training_set = dataset[torch.tensor(train_indexes)]
test_set = dataset[torch.tensor(test_indexes)]

aucs = []
train_aucs = []
start_time = time()
fpr_points = np.linspace(0,1,500)
tpr_train, tpr_val = [], []

# Define the device:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO   ] Device: {device}")
weights = torch.tensor([1, 1.5]).type(torch.FloatTensor).to(device)
for fold, (train_ids, val_ids) in enumerate(kfold.split(training_set,training_set.y)):
    model_name = OUTPUTDIR + os.sep + f'fold_{fold+1}.pt'

    # Print split information:
    print(f'[INFO   ] Total Number of graphs: {len(train_ids)+len(val_ids)+len(test_indexes)}')
    print(f"[INFO   ] Split information:") 
    print(f'[INFO   ] Number of training graphs: {len(train_ids)}')
    print(f'[INFO   ] Number of validation graphs: {len(val_ids)}')
    print(f'[INFO   ] Number of test graphs: {len(test_indexes)}')

    # split the dataset according to the fold indices:
    train_ids = torch.tensor(train_ids)
    val_ids = torch.tensor(val_ids)
    train_dataset = training_set[train_ids]
    val_dataset = training_set[val_ids]   
    
    # Create the dataloaders:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=False)
    val_loader =  DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=False)


    # Create the model:
    model = BitterGCN(train_dataset.num_node_features - NUM_REC, train_dataset.num_edge_features)

    # Define loss function and optimizer
    model.criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    # Define Reduce LR on Plateau:
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=15, min_lr=0.00005, verbose=True)

    # Define early stopper:
    early_stopper = EarlyStopper(patience=6, min_delta=0.03)

    # Move the model to the device:
    model = model.to(device)

    # Call forward with a dummy batch to initialize the parameters:
    dummy_batch = next(iter(train_loader))

    # Call forward with dummy batch:
    model(dummy_batch.to(device).x,dummy_batch.to(device).edge_index,dummy_batch.to(device).edge_attr,dummy_batch.to(device).batch)
    # Print model info:
    print("[INFO   ] Model information:")
    print('[INFO   ] ====================')
    print(f'[INFO   ] Number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'[INFO   ] Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}') 

    # Define the training function:
    def train():
        model.train()
        loss_all = 0.0
        for data in train_loader:
            # Move to the device
            data = data.to(device)
            # Reset gradients
            optimizer.zero_grad()
            # Predict output and calculate loss:
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            if data.num_graphs > 1:
                loss = model.criterion(output, data.y.type(torch.LongTensor).to(device))
            else:
                loss = model.criterion(output.unsqueeze(0), data.y.type(torch.LongTensor).to(device))
            # Backpropagation
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            # Step:
            optimizer.step()
            del data
        return loss_all / len(train_dataset)

    # Define the test function:
    def test(loader):   
        model.eval()
        loss_all = 0
        # correct = 0
        for data in loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            if data.num_graphs > 1:
                loss = model.criterion(output, data.y.type(torch.LongTensor).to(device))
            else:
                loss = model.criterion(output.unsqueeze(0), data.y.type(torch.LongTensor).to(device))
            loss_all += data.num_graphs * loss.item()
            del data
        return loss_all / len(loader.dataset)

    # Train the model:
    print("[INFO   ] Training the model...")

    # Initialize arrays to store train loss and test loss:
    train_loss_list = []
    val_loss_list = []
    for epoch in range(1, N_EPOCHS+1):
        train_loss = train()
        test_loss = test(val_loader)
        last_epoch = epoch
        #if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Validation Loss: {test_loss:.3f}')
        # Append to array for plotting:
        train_loss_list.append(train_loss)
        val_loss_list.append(test_loss)

        # Early stopping if val_loss diverges
        if early_stopper.early_stop(test_loss,epoch=epoch):
            print(f'Early stopping at Epoch {epoch:03d} to avoid overfitting, Train Loss: {train_loss:.3f}, Validation Loss: {test_loss:.3f}')      
            break
        if early_stopper.stuck(test_loss):
            print(f'Early stopping at Epoch {epoch:03d}, Fold is not learning') 
            break

        # Reduce lr on plateau
        if REDUCE_LR_ON_PLATEAU:
            scheduler.step(test_loss)

    # Save the model:
    torch.save(model.state_dict(), model_name)

    # Evaluate the model with the roc curve:
    model.eval()
    y_train, train_probs = [], []
    for data in train_loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        try:
            train_probs.append(F.softmax(output,dim=1)[:,1].cpu().detach().numpy())
        except IndexError:
            output = output.unsqueeze(0)
            train_probs.append(F.softmax(output,dim=1)[:,1].cpu().detach().numpy())
        y_train.append(data.y.cpu().detach().numpy())
        del data
    train_probs = np.concatenate(train_probs)
    y_train = np.concatenate(y_train)
    del y_train, train_probs

    y_val, probs = [], []
    for data in val_loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        probs.append(F.softmax(output,dim=1)[:,1].cpu().detach().numpy())
        y_val.append(data.y.cpu().detach().numpy())
        del data
    probs = np.concatenate(probs)
    y_val = np.concatenate(y_val)
    fpr,tpr,auc_value,_ = get_rates(y_val,probs)
    aucs.append(auc_value)
    del y_val, probs
    print(f'[INFO   ] Model {fold+1} saved.')

aucs = np.array(aucs)

# Remove non-learning folds
train_aucs_clean = [i for i in train_aucs if i > 0.55]
aucs_clean = [i for i in aucs.tolist() if i > 0.55]


print(f'[INFO   ] Mean AUC on Training: {np.mean(train_aucs_clean):.2f} +/- {np.std(train_aucs_clean):.2f}')
print(f'[INFO   ] Mean AUC on Validation: {np.mean(aucs_clean):.2f} +/- {np.std(aucs_clean):.2f}')

end_time = time()
print('[INFO   ] Finished Training in {} minutes, num_workers={}'.format(round((end_time - start_time)/60,2), NUM_WORKERS) )
# get the best model and load it:
best_model_ids = np.argmax(aucs)
best_model_name = OUTPUTDIR + os.sep + f'fold_{best_model_ids+1}.pt'
model.load_state_dict(torch.load(best_model_name))
model.to(device)
os.rename(OUTPUTDIR + os.sep + f'fold_{best_model_ids+1}.pt', OUTPUTDIR + os.sep + f'fold_{best_model_ids+1}_BEST.pt')
print(f'[INFO   ] Best Model is at fold {best_model_ids+1}')

# Evaluate the model on the test set:
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=False)
y_test, probs, y_pred = [], [], []

model.eval()
for data in test_loader:
    data = data.to(device)
    output = F.softmax(model(data.x, data.edge_index, data.edge_attr, data.batch),dim=1)
    _, pred = torch.max(output,dim=1)
    probs.append(F.softmax(output,dim=1)[:,1].cpu().detach().numpy())
    y_test.append(data.y.cpu().detach().numpy())
    y_pred.append(pred.cpu().detach().numpy())
    del data
probs = np.concatenate(probs)
y_test = np.concatenate(y_test)
y_pred = np.concatenate(y_pred)
print('[DONE   ] Model trained. All folds were saved to "Models" folder')
