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

"""
Fonction principale prenant en entrée le chemin vers une série temporelle et ajoutant les résultats 
dans le fichier .json du jeu de données associé.

name_path : nom du chemin permettant d'accéder à la série étudiée.
"""
def main_function (name_path):
    # Suppression des warnings
    warnings.simplefilter("ignore", UserWarning)
    # Création du dictionnaire des algorithmes de détection utilisés
    dict_methods = {"IF" : IForest(random_state=123), 
                   "PCA" : PCA(random_state=123, standardization=False, n_components=0.75),
                   "LOF" : LocalOutlierFactor()}
    # Liste des tailles de fenêtres testées
    w_size = [32,64,128,256]
    # Initialisation du dictionnaire contenant les AUC de la série associée au chemin spécifié
    dict_auc = {"TS" : np.zeros((len(w_size), len(dict_methods)), dtype=int).tolist(), "FE" :np.zeros((len(w_size), len(dict_methods)), dtype=int).tolist() } 
    # Lecture des données 
    data_ts = read_dataset(name_path) 

    # Application des prétraitements et des algorithmes sur la série concernée pour chaque taille de fenêtre 
    for id_ws, ws in enumerate(w_size):
       
        # Application du fenêtrage permettant de transformer la série 'data_ts' en table
        data_wo_fe = sliding_window(data_ts.value, ws)
        # Application du fenêtrage sur les labels de la série 'data_ts' permettant d'agréger les labels de chaque sous-séquence obtenue
        true_labels_transf = transformation_label(data_ts.label, ws)

        # Application d'une normalisation (si cela est souhaité)
        
        # Tsfresh nécessite un format spécifique des données, il est donc nécessaire de pivoter la table 'data_wo_fe'
        data_pivot = pivot4tsfresh(data_wo_fe)
        # Extraction des caractéristiques (MinimalFCParameters = 10 features et EfficientFCParameters = 777 features)
        # Spécifier le nom de processeurs à utiliser avec n_jobs
        FE = extract_features(data_pivot, default_fc_parameters= EfficientFCParameters(), 
            column_id="TSname", column_sort="time", column_kind=None, column_value=None,n_jobs = 42)
        # Suppression de la table pivotée pour gagner en espace mémoire
        del data_pivot 
       
       # Affichage du nombre de colonnes de la table 'FE' contenant les caractéristiques extraites des sous-séquences et de la taille de la fenêtre
        print("\n",  "     before drop", name_path, FE.shape, "ws =", ws)
        # Remplacement des valeurs -infini et +infini par NA 
        FE.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Suppression des colonnes contenant des valeurs manquantes
        FE.dropna(axis='columns', inplace = True)
        # Affichage du nombre de colonnes de la table 'FE' après suppression des colonnes contenant des valeurs manquantes et de la taille de la fenêtre 
        print("       after drop", name_path, FE.shape,  "ws =", ws, "\n")
    
        # Application des méthodes de détection d'anomalies 
        for (id_method, (name_method, method)) in enumerate(dict_methods.items()): 

            # Application de l'algorithme sur les données temporelles 
            method_TS = method
            method_TS.fit(data_wo_fe)
            if name_method != "LOF" :
                anomaly_scores_TS = method_TS.decision_function(data_wo_fe)
            else : 
                anomaly_scores_TS = -method_TS.negative_outlier_factor_
            # Calcul de l'AUC (sur les données temporelles)
            data_auc = round(roc_auc_score(true_labels_transf.label, anomaly_scores_TS),3)
           
            # Application de l'algorithme sur les données contenant les features  
            method_FE = method
            method_FE.fit(FE)
            if name_method != "LOF": 
                anomaly_scores_FE = method_FE.decision_function(FE)
            else : 
                anomaly_scores_FE = -method_FE.negative_outlier_factor_
            # Calcul de l'AUC (sur les features)    
            fe_auc = round(roc_auc_score(true_labels_transf.label, anomaly_scores_FE),3)
            
            # Ajout des deux AUC calculés au dictionnaire regroupant les résultats de la série associée au chemin spécifié
            dict_auc["TS"][id_ws][id_method] = data_auc
            dict_auc["FE"][id_ws][id_method] = fe_auc
            
    # Ajout du dictionnaire des AUC de la série associée au chemin spécifié aux résultats des autres séries du jeu de données
    add_json_file("results/efficient_no_norm_auc.json", dict_auc, name_path)
    

# Application de la fonction précédente au répertoire nommé 'Data' contenant les séries temporelles 
dir_name = "Data"
Parallel(n_jobs=1)(delayed(main_function)(os.path.join(dir_name, file)) for file in os.listdir(dir_name))

