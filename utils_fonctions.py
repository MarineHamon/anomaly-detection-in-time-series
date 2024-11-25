import pandas as pd
import os
import json


def read_dataset(file):
    """
    Reading the time series and associated labels in a pandas Dataframe.

    file : name of the path used to access the series studied.
    """
    df = pd.read_csv(file, header=0) 
    return df


def sliding_window(TS, size):
    """
    Time series windowing.

    TS : a time series in the form of pandas.Series.
    size : window length.
    """

    # Sequence of values from 0 to the number of subsequences created minus one, i.e. len(TS) - size.
    rows = range(len(TS) - size + 1)
    # Creation of an empty pandas Dataframe, which will contain the subsequences of the time series. 
    # Lines are numbered using the sequence ‘rows’ and 
    # the columns are named ‘value0’ to ‘value{size-1}’ where size represents the window size.
    df_windows = pd.DataFrame(index = [f"w{i:06d}" for i in rows], 
                              columns = [f"value{i}" for i in range(size)]) 
    # Fill in the table with the subsequences extracted from the time series.
    for i in rows :
        df_windows.iloc[i,:] = TS[(i) : (i + size)]

    return df_windows


def transformation_label(labels, size):
    """
    Windowing of labels of the time series studied.

    labels : time series labels, in the form of pandas.Series.
    size : window length.
    """

    # Definition of the number of subsequences created by windowing.
    new_size_labels = len(labels) - size + 1
    # Initialise the list of aggregated labels to 0, of length ‘new_size_labels’.
    new_labels = [0]*new_size_labels 

    # Loop used to aggregate a subsequence of labels into a single label.
    for i in range(new_size_labels) :
        # Sum of the values of a label subsequence.
        sum_window = labels[(i) : (i + size)].sum()
        # If the sum is zero, we retain this value. This means that there are no anomalies in the subsequence concerned.
        # Otherwise, it is reset to 1, meaning that at least one data point in the sub-sequence concerned is an anomaly.
        if sum_window > 0:
            new_labels[i] = 1
    # Transformation of the list into pandas Dataframe.
    new_labels_df = pd.DataFrame(new_labels)
    # Renames the column containing the label aggregation.
    new_labels_df.columns = ["label"]
    return new_labels_df




def pivot4tsfresh(data):
    """
    Pivot the data table containing subsequences of length 'size' so that the 'extract_features' function can be applied.

    data : table containing, in rows, the subsequences resulting from the windowing of the time series.
    """

    # Addition of the ‘TSname’ column, corresponding to the names of the extracted subsequences, to the table obtained by windowing.
    new_data = data.reset_index().rename(columns={'index':'TSname'})
    # Rotate the table so that the new table contains two variables as indexes: 
    #   - 'TSname': names of the subsequences, 
    #   - 'time': subsequence timestamps, ranging from 0 to size-1, 
    # and a 'value' column containing the value of the subsequence with name 'TSname' and time 'time'.
    new_data = pd.wide_to_long(new_data, stubnames='value', i='TSname', j='time').sort_values(['TSname', 'time'])
    # Redefinition of the array indexes (and therefore addition of the variables ‘TSname’ and ‘time’ as columns).
    new_data.reset_index(inplace=True)
    # Data values are converted to float.
    new_data['value'] = new_data['value'].astype(float)
    return new_data




def add_json_file(file_name, new_dict, ts_name):
    """
    Function for combining results from several time series in a .json file.

    file_name : name of the .json file in which the results are stored.
    new_dict : dictionary whose keys are 'TS' and 'FE' corresponding to the approach used and whose values are list of lists 
               of the form [[auc_size1_method1, auc_size1_method2], [auc_size2_method1, auc_size2_method2], ...].
    ts_name : name of the time series.
    """
    # Check that the file exists.
    if os.path.exists(file_name):
        # If the file exists, the existing data is retrieved.
        with open(file_name, 'r', encoding='utf-8') as f1:
            dict_json = json.load(f1)
    else:
        # If the file does not exist, a new list is initialized.
        dict_json = {}

    # Add the new dictionary to the list.
    dict_json[ts_name] = new_dict
    
    # Updating the file with the new data.
    with open(file_name, 'w', encoding='utf-8') as f2:
        json.dump(dict_json, f2, ensure_ascii=False, indent=4)



