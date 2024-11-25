import random
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import warnings
from joblib import Parallel, delayed
import os

from tsfresh.feature_extraction import extract_features, MinimalFCParameters, ComprehensiveFCParameters, EfficientFCParameters
from sklearn.metrics import roc_auc_score

from pyod.models.iforest import IForest
from pyod.models.lof import LocalOutlierFactor
from pyod.models.pca import PCA

from utils_fonctions import read_dataset, sliding_window, transformation_label, pivot4tsfresh, add_json_file


def main_function (name_path):
    """
    Main function that takes the path to a time series as input and adds the results
    to the .json file of the associated dataset.

    name_path : name of the path used to access the series studied.
    """
    # Removing warnings.
    warnings.simplefilter("ignore", UserWarning)
    # Creation of a dictionary of detection algorithms used.
    dict_methods = {"IF" : IForest(random_state=123), 
                   "PCA" : PCA(random_state=123, standardization=False, n_components=0.75),
                   "LOF" : LocalOutlierFactor()}
    # List of window sizes tested.
    w_size = [32,64,128,256]
    # Initialisation of the dictionary containing the AUCs of a time series associated with the specified path.
    dict_auc = {"TS" : np.zeros((len(w_size), len(dict_methods)), dtype=int).tolist(), "FE" :np.zeros((len(w_size), len(dict_methods)), dtype=int).tolist() } 
    # Reading data.
    data_ts = read_dataset(name_path) 

    # Application of pre-processing and algorithms to a time series concerned for each window size.
    for id_ws, ws in enumerate(w_size):
       
        # Application of windowing to transform the ‘data_ts’ series into a table.
        data_wo_fe = sliding_window(data_ts.value, ws)
        # Application du fenêtrage sur les labels de la série 'data_ts' permettant d'agréger les labels de chaque sous-séquence obtenue
        true_labels_transf = transformation_label(data_ts.label, ws)

        # Application of standardisation (if desired).
        
        # Tsfresh requires a specific data format, so the ‘data_wo_fe’ table must be rotated.
        data_pivot = pivot4tsfresh(data_wo_fe)
        # Feature extraction (MinimalFCParameters = 10 features et EfficientFCParameters = 777 features).
        # Specify the name of the processors to be used with n_jobs.
        FE = extract_features(data_pivot, default_fc_parameters= EfficientFCParameters(), 
            column_id="TSname", column_sort="time", column_kind=None, column_value=None,n_jobs = 42)
        # Suppression de la table pivotée pour gagner en espace mémoire
        del data_pivot 
       
        # Display of the number of columns in the ‘FE’ table containing the characteristics extracted from the subsequences and the size of the window.
        print("\n",  "     before drop", name_path, FE.shape, "ws =", ws)
        # Replacement of -infinite and +infinite values by NA.
        FE.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Deleting columns with missing values.
        FE.dropna(axis='columns', inplace = True)
        # Display of the number of columns in the ‘FE’ table after deleting columns containing missing values.
        print("       after drop", name_path, FE.shape,  "ws =", ws, "\n")
    
        # Application of anomaly detection methods.
        for (id_method, (name_method, method)) in enumerate(dict_methods.items()): 

            # Application of the algorithm to temporal data.
            method_TS = method
            method_TS.fit(data_wo_fe)
            if name_method != "LOF" :
                anomaly_scores_TS = method_TS.decision_function(data_wo_fe)
            else : 
                anomaly_scores_TS = -method_TS.negative_outlier_factor_
            # Calculation of the AUC (on time series data).
            data_auc = round(roc_auc_score(true_labels_transf.label, anomaly_scores_TS),3)
           
            # Applying the algorithm to data containing features.
            method_FE = method
            method_FE.fit(FE)
            if name_method != "LOF": 
                anomaly_scores_FE = method_FE.decision_function(FE)
            else : 
                anomaly_scores_FE = -method_FE.negative_outlier_factor_
            # Calculation of AUC (on features).   
            fe_auc = round(roc_auc_score(true_labels_transf.label, anomaly_scores_FE),3)
            
            # Addition of the two AUCs calculated to the dictionary grouping the results of the series associated with the specified path.
            dict_auc["TS"][id_ws][id_method] = data_auc
            dict_auc["FE"][id_ws][id_method] = fe_auc
            
    # Add the dictionary containing the AUCs of the series associated with the specified path to the results of the other series in the dataset.
    add_json_file("results/efficient_no_norm_auc.json", dict_auc, name_path)
    

# Application of the previous function to the directory named ‘Data’ containing time series.
dir_name = "Data"
Parallel(n_jobs=1)(delayed(main_function)(os.path.join(dir_name, file)) for file in os.listdir(dir_name))

