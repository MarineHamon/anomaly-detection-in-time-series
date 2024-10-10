import pandas as pd
import os
import json


def read_dataset(file):
    """
    Lecture de la série temporelle et des labels associés dans un pandas Dataframe.

    file : nom du chemin permettant d'accéder à la série étudiée.
    """
    df = pd.read_csv(file, header=0) 
    return df


def sliding_window(TS, size):
    """
    Fenêtrage de la série temporelle.

    TS : une série temporelle sous forme de pandas.Series,
    size : longueur de la fenêtre.
    """

    # Séquence de valeurs allant de 0 jusqu'au nombre de sous-séquences créées moins une, soit len(TS) - size.
    rows = range(len(TS) - size + 1)
    # Création d'un pandas Dataframe vide, qui contiendra les sous-séquences de la série temporelle. 
    # Les lignes sont numérotées à l'aide de la séquence 'rows' et 
    # les colonnes sont nommées 'value0' à 'value{size-1}' où size représente la taille du fenêtrage.
    df_windows = pd.DataFrame(index = [f"w{i:06d}" for i in rows], 
                              columns = [f"value{i}" for i in range(size)]) 
    # Remplissage du tableau avec les sous-séquences extraites de la série temporelle.
    for i in rows :
        df_windows.iloc[i,:] = TS[(i) : (i + size)]

    return df_windows


def transformation_label(labels, size):
    """
    Fenêtrage des labels de la série temporelle étudiée.

    labels : labels de la série temporelle, sous forme de pandas.Series,
    size : longueur de la fenêtre.
    """

    # Définition du nombre de sous-séquences créées par fenêtrage.
    new_size_labels = len(labels) - size + 1
    # Initialisation à 0 de la liste correspondant aux labels agrégés, de longueur 'new_size_labels'.
    new_labels = [0]*new_size_labels 

    # Boucle permettant d'agréger une sous-séquence de labels en un seul label.
    for i in range(new_size_labels) :
        # Somme des valeurs d'une sous-séquence de labels.
        sum_window = labels[(i) : (i + size)].sum()
        # Si la somme est nulle, nous conservons cette valeur. Cela signifie qu'il n'y a pas d'anomalies dans la sous-séquence concernée.
        # Sinon, elle est remise à 1, signifiant qu'au moins un point de données de la sous-séquence concernée est une anomalie.
        if sum_window > 0:
            new_labels[i] = 1
    # Transformation de la liste en pandas Dataframe.
    new_labels_df = pd.DataFrame(new_labels)
    # Renommage de la colonne contenant l'agrégation des labels.
    new_labels_df.columns = ["label"]
    return new_labels_df




def pivot4tsfresh(data):
    """
    Pivotage de la table de données comportant les sous-séquences de longueur 'size' afin que la fonction 'extract_features' soit applicable.

    data : tableau contenant, en lignes, les sous-séquences issues du fenêtrage de la série temporelle.
    """

    # Ajout de la colonne 'TSname', correspondant aux noms des sous-séquences extraites, au tableau obtenu par fenêtrage.
    new_data = data.reset_index().rename(columns={'index':'TSname'})
    # Pivotage du tableau de telle sorte que la nouvelle table contienne deux variables comme index : 
    #   - 'TSname' : noms des sous-séquences, 
    #   - 'time' : timestamps des sous-séquences, allant de 0 à size-1, 
    # et une colonne 'value' contenant la valeur de la sous-séquence au nom 'TSname' et au temps 'time'.
    new_data = pd.wide_to_long(new_data, stubnames='value', i='TSname', j='time').sort_values(['TSname', 'time'])
    # Redéfinition des index du tableau (et donc ajout des variables 'TSname' et 'time' comme colonnes).
    new_data.reset_index(inplace=True)
    # Les valeurs des données sont converties en float.
    new_data['value'] = new_data['value'].astype(float)
    return new_data




def add_json_file(file_name, nv_dict, ts_name):
    """
    Fonction permettant de joindre les résultats provenant de plusieurs séries temporelles dans un fochier .json.

    file_name : nom du fichier .json dans lequel les résultats sont conservés,
    nv_dict : dictionnaire dont les clés sont 'TS' et 'FE' correspondant à l'approche employée et les valeurs sont des 
    listes de liste de la forme [[auc_size1_method1, auc_size1_method2], [auc_size2_method1, auc_size2_method2], ...],
    ts_name : nom de la série temporelle.
    """
    # Vérification de l'existence du fichier.
    if os.path.exists(file_name):
        # Si le fichier existe, les données existantes sont récupérées.
        with open(file_name, 'r', encoding='utf-8') as f1:
            dict_json = json.load(f1)
    else:
        # Si le fichier n'existe pas, une nouvelle liste est initialisée.
        dict_json = {}

    # Ajout du nouveau dictionnaire à la liste.
    dict_json[ts_name] = nv_dict
    
    # Mise à jour du fichier avec les nouvelles données.
    with open(file_name, 'w', encoding='utf-8') as f2:
        json.dump(dict_json, f2, ensure_ascii=False, indent=4)



