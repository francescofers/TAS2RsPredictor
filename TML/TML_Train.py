import pandas as pd
pd.options.mode.chained_assignment = None

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from Virtuous import Calc_fps, Calc_Mordred

from catboost import CatBoostClassifier

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cluster import AgglomerativeClustering
#from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold, train_test_split

import matplotlib as mpl
mpl.rcParams['font.size'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['legend.frameon'] = False

from sklearn.metrics import roc_curve, auc, silhouette_score, RocCurveDisplay, average_precision_score, precision_recall_curve, classification_report
NUM_REC = 22
PATH = f'../Data/expanded_dataset/expanded_noOrph_{NUM_REC}.csv'


# Importing the expanded dataset without all receptors that bind to less than 5 ligands
df = pd.read_csv(PATH, header=0, index_col=0)

# Min-max scaling function
def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())

def tanimoto_distance(bitstring1, bitstring2):
    """ This function takes two bit strings and calculates the tanimoto distance between them.
    The bit strings must be of the same length."""
    # Calculate the tanimoto distance between the two bit strings (without using RDKit):
    # Convert the bit strings to numpy arrays:
    bitstring1 = np.array(list(bitstring1),dtype=int)
    bitstring2 = np.array(list(bitstring2),dtype=int)
    # Print them for debug:
    #print(f"[DEBUG  ] bitstring1: {bitstring1}")
    #print(f"[DEBUG  ] bitstring2: {bitstring2}")
    # Calculate the dot product of the two arrays:
    dot_product = np.dot(bitstring1,bitstring2)
    # Calculate the magnitude of the two arrays:
    magnitude = np.sum(bitstring1) + np.sum(bitstring2) - np.sum(dot_product)
    # Calculate the tanimoto distance:
    tanimoto_distance = 1 - (dot_product / magnitude)
    # Return the tanimoto distance:
    return tanimoto_distance

def remove_corr_features(mord_df, threshold):

    # Calculate the correlation matrix
    corr_matrix = mord_df.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                #print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    mord_df = mord_df.drop(columns=drops)

    return mord_df

# Features generation

# Pre-calculate fps and descriptors for the ligands and then associate them with the pairs to avoid a lot of useless and time-consuming calculations
# Pre-calculated features are only for training purposes, when evaluating we will generate them impromptu

ndx = df.index.format()[:]

# Fingerprints
def Calc_MFps():
    try:
        fps_df = pd.read_csv(f'{PATH.split(".c")[0]}_fps.csv', header=0, index_col=0)
            
    except FileNotFoundError:
        print("Precalculated fingerprints file not found. Calculating Morgan fingerprints. This will take a while...")
        fps_df = df
        for col in fps_df.columns:
            fps_df = fps_df.drop(col,axis=1)
        mol_fps = [Calc_fps(mol,1024,2) for mol in ndx]
        mol_fps_np = [np.frombuffer(fp.ToBitString().encode(), 'u1') - ord('0') for fp in mol_fps]
        for row, array in zip(fps_df.index, mol_fps_np):
            fps_df.loc[row, [f'f_{i+1}' for i in range(len(array))]] = array
        fps_df = fps_df.astype(int)
        fps_df.to_csv(f'{PATH.split(".c")[0]}_fps.csv')

    return fps_df

# Mordred
def Calc_mord(norm=True):
    try:
        mord_df = pd.read_csv(f'{PATH.split(".c")[0]}_mord.csv', header=0, index_col=0)
            
    except FileNotFoundError:
        print("Precalculated descriptors file not found. Calculating Mordred descriptors. This will take a while...")
        mord_df = df
        for col in mord_df.columns:
            mord_df = mord_df.drop(col,axis=1)
        mol_mord = [Calc_Mordred(mol)[1] for mol in ndx]
        mord_header = Calc_Mordred(ndx[1])[0]
        for row, array in zip(mord_df.index, mol_mord):
            mord_df.loc[row, [f'{desc}' for desc in mord_header]] = array
        mord_df = mord_df.dropna(axis=1, how='any')
        mord_df.drop(columns=mord_df.columns[mord_df.sum()==0], inplace=True)
        # Removing correlated features
        remove_corr_features(mord_df,0.90)
        mord_df.to_csv(f'{PATH.split(".c")[0]}_mord.csv')
    
    # Min-Max normalization
    if norm:
        for col in mord_df.iloc[:,:].columns:
            mord_df[col] = min_max_scaling(mord_df[col])

    return mord_df

# Clustering
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
    print(silhouette_lists)
    best_n_clusters = range_n_clusters[np.argmax(silhouette_lists)]

    # Apply Agglomerative clustering with best n_clusters for receptor
    aggls = AgglomerativeClustering(n_clusters=best_n_clusters, metric='precomputed', linkage='complete')
    aggls = aggls.fit(tanimoto_matrix)
    labels = aggls.labels_
    print(labels)
    print(len(labels))
    return labels

def Get_tanimoto_matrix(df):
    grp_ndx = df.index.format()[:]
    mol_fps = [Calc_fps(mol,1024,2) for mol in grp_ndx]
    try:
        tanimoto_matrix = np.loadtxt(f'../Data/expanded_dataset/tanimoto/tanimoto_simmat_tot.txt')
    except OSError:
        # Initialize tanimoto matrix:
        print("Calculating Tanimoto similarity matrices for whole dataset of ligands. This will take a while...")
        tanimoto_matrix = np.zeros((len(mol_fps),len(mol_fps)))
        # Iterate over all pairs of fingerprints and calculate the tanimoto similarity using the helper function:
        for i in range(len(mol_fps)):
            for j in range(len(mol_fps)):
                tanimoto_matrix[i,j] = tanimoto_distance(mol_fps[i],mol_fps[j])
        print(f"    Total Tanimoto simmat saved")
        np.savetxt(f'../Data/expanded_dataset/tanimoto/tanimoto_simmat_tot.txt',tanimoto_matrix)
    
    return tanimoto_matrix

# Custom train_test_split based on clusters
def clust_train_test_split(pairs):
    
    # Get Tanimoto similarity matrix of the ligands of the receptor
    tanimoto_matrix = Get_tanimoto_matrix(df)
    # Clustering with KMeans
    cluster_labels = Cluster(tanimoto_matrix)
    cluster_labels = pd.DataFrame(cluster_labels,index=df.index,columns=['cluster'])
    pairs = pd.merge(pairs,cluster_labels,left_index=True,right_index=True)
    
    # Assembling Training test and Test set
    x_col = pairs.iloc[:,2:]
    x_train = pd.DataFrame(columns=x_col.columns)
    x_test = pd.DataFrame(columns=x_col.columns)
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()

    pairs = pairs.groupby(pairs.cluster)
    
    for clusters, elements in pairs:
        x_t = elements.iloc[:,2:]
        y_t = elements.iloc[:,1]
        try:
            x_t_train, x_t_test, y_t_train, y_t_test = train_test_split(x_t,y_t,test_size=0.2, random_state=42, stratify=y_t)
        except ValueError:
            x_t_train, x_t_test, y_t_train, y_t_test = train_test_split(x_t,y_t,test_size=0.2, random_state=42)
        x_train = pd.concat([x_train,x_t_train])
        x_test = pd.concat([x_test,x_t_test])
        y_train = pd.concat([y_train,y_t_train])
        y_test = pd.concat([y_test,y_t_test])

    x_train = x_train.drop(columns=['cluster'])
    x_test = x_test.drop(columns=['cluster'])
    return x_train, x_test, y_train, y_test

# Define model
def Get_model(n_iter, lr):

    model = CatBoostClassifier(loss_function='Logloss', 
                            eval_metric='Logloss', 
                            depth=6,
                            random_seed=42,
                            iterations=n_iter,
                            learning_rate=lr,
                            leaf_estimation_iterations=4,
                            l2_leaf_reg=3,
                            subsample=0.7,
                            boosting_type='Plain',
                            verbose=False,
                            class_weights=[1, 1.5],
                            )

    return model
def get_rates(y_test,probs):
    fpr, tpr, thresholds = roc_curve(y_test, probs, drop_intermediate=False)
    auc_value = auc(fpr,tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return fpr,tpr,auc_value,optimal_threshold

def skf_crossval(x, y, ax):
    n_fold = 10
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    tprs = []
    aucs = []
    models = []
    mean_fpr = np.linspace(0, 1, 500)
    fpr_points = np.linspace(0,1,500)
    for fold, (train, val) in enumerate(skf.split(x, y)):
        model_cv = Get_model(1000,0.1)
        model_cv.fit(x.iloc[train], y.iloc[train])
        probs = model_cv.predict_proba(x.iloc[val])[:,1]
        fpr, tpr, auc_value, _ = get_rates(y.iloc[val],probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc_value)
        models.append(model_cv)

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    best_model = models[np.argmax(aucs)]

    ax.plot(fpr_points, mean_tpr, color='darkgreen',
            label=r'Val (AUC = {} $\pm$ {})'.format(np.round(mean_auc,2),np.round(std_auc,2)),alpha=1)
    ax.fill_between(fpr_points, mean_tpr - std_tpr, mean_tpr + std_tpr, facecolor='darkgreen', edgecolor=None, alpha=0.2)

    return best_model

# Stratified K-fold cross validation function
def OLDskf_crossval(x, y):
    
    n_fold = 10
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    model_cv = Get_model(1000,0.1)
    tprs = []
    aucs = []
    models = []
    mean_fpr = np.linspace(0, 1, 100)
    for fold, (train, val) in enumerate(skf.split(x, y)):
        model_cv.fit(x.iloc[train], y.iloc[train])
        viz = RocCurveDisplay.from_estimator(
            model_cv,
            x.iloc[val],
            y.iloc[val],
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=axes,
            plot_chance_level=(fold == n_fold - 1),
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        models.append(model_cv)

    no_overfitted_aucs = [0.1 if x == 1 else x for x in aucs]
    best_model = models[np.argmax(no_overfitted_aucs)]

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    axes.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    axes.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    axes.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"GB on DT (10-Fold CV) on Expanded noOrph (LB) || ROC curves",
    )
    axes.axis("square")
    #axes[p_row, p_col].legend(loc="center left", bbox_to_anchor=(1,0.5))

    return best_model

# Preparing pairs dataset
print('Initializing pairs datasets')
fps_df = Calc_MFps()
fps_df = fps_df.loc[:, fps_df.any()]
mord_df = Calc_mord()
pairs = pd.melt(df.assign(ligand=df.index), id_vars='ligand', var_name='receptor', value_name='association').dropna(axis=0, how='any').drop_duplicates(keep="first").set_index(['ligand'])
pairs_df = pd.merge(pairs,fps_df,left_index=True,right_index=True)
pairs_df = pd.merge(pairs_df,mord_df,left_index=True,right_index=True)
pairs_df = pairs_df.dropna(axis=1, how='any')
pairs_df['receptor'] = pairs_df['receptor'].astype(int)

# Add one hot encoded receptors associated as features
encoded = pd.get_dummies(pairs_df['receptor']).astype(int)
pairs_df = pd.concat([pairs_df, encoded], axis=1)

# Clustered division on train and test set
print('Generating training and test with clustering...')
x_train, x_test, y_train, y_test = clust_train_test_split(pairs_df)

import ast
nf = 17
selectedfeatures_df = pd.read_csv('ModelSelection_MC/LB_Whole_22_sel_feature_per_iter.csv', header=0, index_col=0)
selected_columns = selectedfeatures_df.loc[selectedfeatures_df['n_selected_features'] == nf, 'selected_features'].values[0]
selected_columns = ast.literal_eval(selected_columns)

# extract the selected features from the training and test sets
ni = len(x_train.columns)
x_train_fs_1 = x_train[x_train.columns.intersection(selected_columns)]
x_train_fs_2 = x_train.iloc[:,ni-NUM_REC:]
x_train = pd.concat([x_train_fs_1, x_train_fs_2], axis=1)

x_test_fs_1 = x_test[x_test.columns.intersection(selected_columns)]
x_test_fs_2 = x_test.iloc[:,ni-NUM_REC:]
x_test = pd.concat([x_test_fs_1, x_test_fs_2], axis=1)

fig, axes = plt.subplots(figsize=(8/2.54, 8/2.54))

# Cross Validation of training set
print('Cross-Validation on training set with Stratified KFold...')
best_model = skf_crossval(x_train,y_train,axes)

import pickle
with open('TML_model.pkl', 'wb') as f:
     pickle.dump(best_model, f)