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
import random
from pygrinder import mcar, seq_missing, block_missing

device = 'cuda' if torch.cuda.is_available() else 'cpu'

""
def create_missingness(X, rate, pattern, sub_seq_len, block_len, block_width):
    """Create missingness in the data.
    Parameters
    ----------
    X:
        The input data.

    rate:
        The missing rate.

    pattern:
        The missing pattern to apply to the dataset.
        Must be one of ['point', 'subseq', 'block'].

    Returns
    -------
    """
    supported_missing_pattern = ["point", "subseq", "block"]

    assert 0 < rate < 1, "rate must be in [0, 1)"
    assert (
        pattern.lower() in supported_missing_pattern
    ), f"pattern must be one of {supported_missing_pattern}, but got {pattern}"

    if pattern == "point":
        return torch.isnan(mcar(X, rate)).to(torch.int) 
    elif pattern == "subseq":
        return torch.isnan(seq_missing(X, rate, sub_seq_len)).to(torch.int) 
    elif pattern == "block":
        return torch.isnan(block_missing(X, factor=rate, block_len=block_len, block_width=block_width)).to(torch.int) 
    else:
        raise ValueError(f"Unknown missingness pattern: {pattern}")


""
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

""
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

    #cond = partitioned_tensors[:, :, column_index]

    # Remove the column at column_index_to_remove
    partitioned_tensors = torch.cat((partitioned_tensors[:, :, :column_index], 
                                   partitioned_tensors[:, :, column_index+1:]), dim=2)
    
    return (partitioned_tensors, partitioned_tensors_ts)

""
def sliding_window(ori_data, seq_len, stride, threshold):
    
    # Preprocess the dataset
    temp_data = []; i = 0
    # Cut data by sequence length
    while i <= (len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        i+=stride
        temp_data.append(_x)

    data = torch.tensor(temp_data, dtype=torch.float)
    
    return data

""
def splitData(real_df, seq_len, stride, threshold):
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
    ori_data = torch.tensor(data.astype('float32'), dtype=torch.float32).numpy()

    batch_size = len(data) - seq_len

    # Preprocess the dataset
    temp_data = []; i = 0
    # Cut data by sequence length
    while i <= (len(real_df) - seq_len):
        _x = ori_data[i:i + seq_len]
        i+=stride
        temp_data.append(_x)

    data = torch.tensor(temp_data, dtype=torch.float)
    
    return data

""
def splitTimeData(real_df, seq_len):
    """Load and preprocess real-world datasets.
    Args:
      - data_name: Numpy array with the values from a Dataset
      - seq_len: sequence length
    Returns:
      - data: preprocessed data.
    """
    # Flip the data to make chronological data
    # Normalize the data
    df2 = cyclical_encode(real_df); tlen = df2.shape[1]
    time_info = torch.tensor(df2.iloc[:,-8:].values).numpy()
    real_ts = torch.arange(0, len(real_df)).numpy()

    batch_size = len(time_info) - seq_len

    # Preprocess the dataset
    temp_data = []; real_ts_data = []; i = 0

    # Cut data by sequence length
    while i <= (len(real_df) - seq_len):
        _x = time_info[i:i + seq_len]
        _ts = real_ts[i:i + seq_len]
        i+=1
        temp_data.append(_x)
        real_ts_data.append(_ts)

    data = torch.tensor(temp_data, dtype=torch.float).to(device)
    real_ts = torch.tensor(real_ts_data, dtype=torch.float).to(device)

    return (data, real_ts)

""
def train_test_split(length):    
    training_idx = random.sample(range(length), int(0.8*length))
    random_numbers_set = set(training_idx)

    # Find remaining numbers
    all_numbers_set = set(range(length))
    test_idx = list(all_numbers_set - random_numbers_set)

    return(training_idx, test_idx)

""
def single_split_train_test(real_df, seq_len, stride):
    real_df1 = real_df.drop('date', axis=1)
    real_df2= real_df
    
    parser = pce.DataFrameParser().fit(real_df1, 1)
    
    threshold = 1; device = 'cuda'
    processed_data = splitData(real_df1, seq_len, stride, threshold)  
    time_info, real_ts = splitTimeData(real_df2, seq_len)      
    train_idx, test_idx = train_test_split(len(processed_data))
    
    return (processed_data, time_info, real_ts, train_idx, test_idx)

""
def multi_split_train_test(real_df, column_to_partition):
    processed_data, cond, time_info = partition_multi_seq(real_df, 1, column_to_partition)    
    return (processed_data, cond, time_info)

###############################################################################################################
# ## For Unconditional Generation
def unconditional_dataprovider(real_df, seq_len, stride):
    processed_data, time_info, real_ts, train_idx, test_idx = single_split_train_test(real_df, seq_len, stride)
    response_train, response_test = processed_data[train_idx,:,:], processed_data[test_idx,:,:]
    cond_train, cond_test = torch.zeros_like(response_train), torch.zeros_like(response_test)
    target_mask_train, target_mask_test = torch.ones_like(response_train), torch.ones_like(response_test)
    time_info_train, time_info_test = time_info[train_idx,:,:], time_info[test_idx,:,:]

    return (target_mask_train, target_mask_test, response_train, response_test, cond_train, cond_test, time_info_train, time_info_test, real_ts_train, real_ts_test)

### For Forecasting problem
def forecasting_dataprovider(real_df, seq_len, stride, timewindow):
    processed_data, time_info, real_ts, train_idx, test_idx = single_split_train_test(real_df, seq_len, stride)
    shape = processed_data.shape
    cond, response, mask = torch.zeros_like(processed_data), torch.zeros_like(processed_data), torch.zeros_like(processed_data)
    target_time_info = torch.zeros_like(time_info)

    cond[:, :timewindow, :] = processed_data[:, :timewindow, :]
    response[:, timewindow:, :] = processed_data[:, timewindow:, :]
    mask[:, timewindow:, :] = torch.ones(shape[0], shape[1]-timewindow, shape[2])
    target_time_info[:, timewindow:, :] = time_info[:, timewindow:, :]

    response_train, response_test = response[train_idx,:,:], response[test_idx,:,:]
    cond_train, cond_test = cond[train_idx,:,:], cond[test_idx,:,:]
    target_mask_train, target_mask_test = mask[train_idx,:,:], mask[test_idx,:,:]    
    time_info_train, time_info_test = target_time_info[train_idx,:,:], target_time_info[test_idx,:,:]
    real_ts_train, real_ts_test = real_ts[train_idx,:], real_ts[test_idx,:]

    return (target_mask_train, target_mask_test, response_train, response_test, cond_train, cond_test, time_info_train, time_info_test, real_ts_train, real_ts_test)

### For Imputation problem
def imputation_dataprovider(real_df, seq_len, stride, rate, pattern, sub_seq_len, block_len, block_width):
    processed_data, time_info, real_ts, train_idx, test_idx = single_split_train_test(real_df, seq_len, stride)

    # mode : 'separate' or 'concurrent'
    batch_size, seq_len, feat_dim = processed_data.shape
    missing_mask = create_missingness(processed_data, rate, pattern, sub_seq_len, block_len, block_width)

    response = processed_data*missing_mask   # Missing Data
    cond = processed_data*(1-missing_mask)   # Observed Data

    response_train, response_test = response[train_idx,:,:], response[test_idx,:,:]
    cond_train, cond_test = cond[train_idx,:,:], cond[test_idx,:,:]
    target_mask_train, target_mask_test = missing_mask[train_idx,:,:], missing_mask[test_idx,:,:]
    time_info_train, time_info_test = time_info[train_idx,:,:], time_info[test_idx,:,:]

    return (target_mask_train, target_mask_test, response_train, response_test, cond_train, cond_test, time_info_train, time_info_test, train_idx, test_idx)

### For metadata conditional generation
def timevaryingcon_dataprovider(real_df, seq_len, stride, column_list, target_column):
    processed_data, time_info, real_ts, train_idx, test_idx = single_split_train_test(real_df, seq_len, stride)

    B, T, F = processed_data.shape
    target_mask = torch.zeros_like(processed_data)

    indices = [column_list.index(col) for col in target_column]
    target_mask[:,:,indices] = 1

    response = processed_data * target_mask
    cond = processed_data * (1-target_mask)

    response_train, response_test = response[train_idx,:,:], response[test_idx,:,:]
    cond_train, cond_test = cond[train_idx,:,:], cond[test_idx,:,:]
    target_mask_train, target_mask_test = target_mask[train_idx,:,:], target_mask[test_idx,:,:]    
    time_info_train, time_info_test = time_info[train_idx,:,:], time_info[test_idx,:,:]
    real_ts_train, real_ts_test = real_ts[train_idx,:], real_ts[test_idx,:]

    return (target_mask_train, target_mask_test, response_train, response_test, cond_train, cond_test, time_info_train, time_info_test, real_ts_train, real_ts_test)

    
