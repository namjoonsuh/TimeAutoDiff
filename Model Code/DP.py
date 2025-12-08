import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import tqdm
import tqdm.notebook
import gc
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import process_edited as pce
from datetime import date
from sklearn.preprocessing import FunctionTransformer

################################################################################################################
def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

# cyclical encoding function
def cyclical_encode(df, year_period=3, month_period=12, day_period=365, hour_period=24):
    # Assuming df datetime follows the following format: 'YYYY-MM-DD HH:MM:SS' with column name 'date'
    res = df.copy()
    res.date = pd.to_datetime(res.date)
    res.set_index('date', inplace=True)
    time = res.index

    # If not using any period then set to False
    if year_period is not None:
        res['year_sin'] = sin_transformer(year_period).fit_transform(time.year)
        res['year_cos'] = cos_transformer(year_period).fit_transform(time.year)

    if month_period is not None:
        res['month_sin'] = sin_transformer(month_period).fit_transform(time.month)
        res['month_cos'] = cos_transformer(month_period).fit_transform(time.month)

    if day_period is not None:
        res['day_sin'] = sin_transformer(day_period).fit_transform(time.day_of_year)
        res['day_cos'] = cos_transformer(day_period).fit_transform(time.day_of_year)
    
    if hour_period is not None:
        res['hour_sin'] = sin_transformer(hour_period).fit_transform(time.hour)
        res['hour_cos'] = cos_transformer(hour_period).fit_transform(time.hour)
    
    return res

################################################################################################################
def partition_multi_seq(real_df, threshold, column_to_partition):    
    
    # column_to_partition
    real_df1 = real_df.drop('date', axis=1)
    parser = pce.DataFrameParser().fit(real_df1, threshold)
    processed_data = torch.from_numpy(parser.transform()).unsqueeze(0)
    column_name = parser._column_order
    column_index = column_name.index(column_to_partition)

    # Partition multi-sequence data
    unique_values = np.unique(processed_data[:, :, column_index])

    partitioned_tensors = torch.zeros(len(unique_values), int(len(processed_data[0,:,:])/len(unique_values)), processed_data.shape[2])

    # Partition the tensor based on unique values in the specified column
    i = 0
    for value in unique_values:
        mask = processed_data[:, :, column_index] == value
        partitioned_tensors[i, :, :] = processed_data[mask]
        i = i + 1
    
    # Partition the multi-sequence data's date information
    df2 = cyclical_encode(real_df); 
    partitioned_tensors_ts = torch.zeros(len(unique_values), int(len(processed_data[0,:,:])/len(unique_values)), 8)
    time_info = torch.tensor(df2.iloc[:,-8:].values).unsqueeze(0)
    
    i = 0
    for value in unique_values:
        mask = processed_data[:, :, column_index] == value
        partitioned_tensors_ts[i, :, :] = time_info[mask]
        i = i + 1
    
    # Remove the column at column_index_to_remove
    partitioned_tensors = torch.cat((partitioned_tensors[:, :, :column_index], 
                                   partitioned_tensors[:, :, column_index+1:]), dim=2)

    return (partitioned_tensors, partitioned_tensors_ts)

################################################################################################################
def splitData(real_df, seq_len, threshold):
    """Load and preprocess real-world datasets.
    Args:
      - data_name: Numpy array with the values from a a Dataset
      - seq_len: sequence length
    Returns:
      - data: preprocessed data.
    """
    # Flip the data to make chronological data
    # Normalize the data
    parser = pce.DataFrameParser().fit(real_df, threshold)
    data = parser.transform()

    # Convert the data to a PyTorch tensor with float32 type
    ori_data = torch.tensor(data.astype('float32'))

    batch_size = len(ori_data) - seq_len

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, batch_size):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    #idx = np.random.permutation(len(temp_data))
    #data = []
    #for i in range(len(temp_data)):
    #    data.append(temp_data[idx[i]])

    data = torch.tensor(temp_data)
    
    return data

################################################################################################################
def splitTimeData(real_df, seq_len):
    """Load and preprocess real-world datasets.
    Args:
      - data_name: Numpy array with the values from a a Dataset
      - seq_len: sequence length
    Returns:
      - data: preprocessed data.
    """
    # Flip the data to make chronological data
    # Normalize the data
    df2 = cyclical_encode(real_df); tlen = df2.shape[1]
    time_info = torch.tensor(df2.iloc[:,-8:].values).numpy()
      
    batch_size = len(time_info) - seq_len

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(time_info) - seq_len):
        _x = time_info[i:i + seq_len]
        temp_data.append(_x)

    data = torch.tensor(temp_data)
    
    return data

