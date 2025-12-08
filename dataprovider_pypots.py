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
import random
from pygrinder import mcar, seq_missing, block_missing
from sklearn.preprocessing import FunctionTransformer

# --- Helper functions for time embedding ---

def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

def cyclical_encode(df, year_period=3, month_period=12, day_period=365, hour_period=24):
    """
    Encodes datetime features into cyclical sin/cos representations.
    """
    # Assuming df datetime follows the following format: 'YYYY-MM-DD HH:MM:SS' with column name 'date'
    res = df.copy()
    # Ensure 'date' is in datetime format
    res['date'] = pd.to_datetime(res['date'])
    res.set_index('date', inplace=True)
    time = res.index

    # Create a new DataFrame to hold the cyclical features
    time_features = pd.DataFrame(index=time)

    # If not using any period then set to None
    if year_period is not None:
        time_features['year_sin'] = sin_transformer(year_period).fit_transform(time.year.values.reshape(-1, 1))
        time_features['year_cos'] = cos_transformer(year_period).fit_transform(time.year.values.reshape(-1, 1))

    if month_period is not None:
        time_features['month_sin'] = sin_transformer(month_period).fit_transform(time.month.values.reshape(-1, 1))
        time_features['month_cos'] = cos_transformer(month_period).fit_transform(time.month.values.reshape(-1, 1))

    if day_period is not None:
        time_features['day_sin'] = sin_transformer(day_period).fit_transform(time.day_of_year.values.reshape(-1, 1))
        time_features['day_cos'] = cos_transformer(day_period).fit_transform(time.day_of_year.values.reshape(-1, 1))

    if hour_period is not None:
        time_features['hour_sin'] = sin_transformer(hour_period).fit_transform(time.hour.values.reshape(-1, 1))
        time_features['hour_cos'] = cos_transformer(hour_period).fit_transform(time.hour.values.reshape(-1, 1))

    return torch.tensor(time_features.values, dtype=torch.float32)

def train_val_test_split_data(tensor, train_ratio=0.7, val_ratio=0.15, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    num_rows = tensor.shape[0]
    indices = torch.randperm(num_rows)

    train_end = int(train_ratio * num_rows)
    val_end = train_end + int(val_ratio * num_rows)

    train = tensor[indices[:train_end]]
    val = tensor[indices[train_end:val_end]]
    test = tensor[indices[val_end:]]

    return train, val, test

def sliding_window(ori_data, seq_len, stride):
    
    # Preprocess the dataset
    temp_data = []; i = 0
    # Cut data by sequence length
    while i <= (len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        i+=stride
        temp_data.append(_x)

    data = torch.stack(temp_data)
    
    return data

def infer_conditional_dtypes_with_names(
    parser,
    target_column,
    encoded_data,
):
    """
    parser: DataFrameParser (fit & transform까지 끝난 상태)
    target_column: response(타겟)으로 쓰는 컬럼 이름 리스트
    encoded_data: parser.transform() 결과 (shape: (N, F))
                  -> 각 컬럼이 전부 숫자 인코딩된 행렬

    return: dict
      - cond_cont_idx:  conditional matrix에서 continuous로 볼 feature index 리스트
      - cond_cat_idx:   conditional matrix에서 categorical/binary로 볼 feature index 리스트
      - cond_cont_cols: 위 index에 해당하는 column name 리스트
      - cond_cat_cols:  위 index에 해당하는 column name 리스트
      - cat_num_classes_idx:  {feature_index: num_classes}
      - cat_num_classes_name: {column_name: num_classes}
    """
    # 0. 인코딩 이후 컬럼 순서 (binary + categorical + numerical)
    col_order = parser.column_name()   # list[str]
    target_set = set(target_column)

    # 1. parser가 기록해둔 타입별 column name
    bin_cols = parser.binary_columns
    cat_cols = parser.categorical_columns
    num_cols = parser.numerical_columns

    # 2. "조건부" 쪽에 들어가는 column name들만 추리기
    #    (target_column은 cond에서 제외)
    cat_like_cols = bin_cols + cat_cols
    cond_cat_cols = [c for c in cat_like_cols if c not in target_set]
    cond_cont_cols = [c for c in num_cols if c not in target_set]

    # 3. column name -> encoded feature index 매핑
    cond_cat_idx = [col_order.index(c) for c in cond_cat_cols]
    cond_cont_idx = [col_order.index(c) for c in cond_cont_cols]

    # 4. categorical feature별 클래스 개수 (encoded_data에서 max+1)
    cat_num_classes_idx = {}
    cat_num_classes_name = {}
    for c in cond_cat_cols:
        idx = col_order.index(c)
        # NaN이 만약 있다면 np.nanmax로 바꾸면 됨
        num_classes = int(np.nanmax(encoded_data[:, idx])) + 1
        cat_num_classes_idx[idx] = num_classes
        cat_num_classes_name[c] = num_classes

    info = {
        "cond_cont_idx": cond_cont_idx,
        "cond_cat_idx": cond_cat_idx,
        "cond_cont_cols": cond_cont_cols,
        "cond_cat_cols": cond_cat_cols,
        "cat_num_classes_idx": cat_num_classes_idx,
        "cat_num_classes_name": cat_num_classes_name,
    }
    return info

import torch

def add_continuous_metadata_noise_subset(
    cond: torch.Tensor,
    cont_idx: list[int],
    level: float,
) -> torch.Tensor:
    """
    cond: (B, T, F)  # TVMCG에서 나온 cond_{train/val/test}
    cont_idx: 연속형 metadata column index 리스트 (예: [3, 5, 6])
    level: noise 비율 (0.0, 0.1, 0.2, 0.4)

    return: cond_noisy (same shape, same device)
    """
    if level <= 0 or len(cont_idx) == 0:
        return cond

    device = cond.device
    cont_idx_tensor = torch.tensor(cont_idx, dtype=torch.long, device=device)

    # (B, T, F_cont)
    x_cont = cond[:, :, cont_idx_tensor]

    # feature별 std (batch, time 전체 기준)
    std = x_cont.std(dim=(0, 1), keepdim=True)  # (1, 1, F_cont)
    std = std + 1e-8

    noise = torch.randn_like(x_cont) * (level * std)
    x_noisy = x_cont + noise

    cond_noisy = cond.clone()
    cond_noisy[:, :, cont_idx_tensor] = x_noisy
    return cond_noisy

def flip_categorical_metadata_subset(
    cond: torch.Tensor,
    cat_idx: list[int],
    p: float,
    num_classes_dict: dict[int, int],
) -> torch.Tensor:
    """
    cond: (B, T, F)
    cat_idx: 범주형 metadata column index 리스트 (예: [2, 4])
    p: flip 확률 (0.0, 0.1, 0.2, 0.4)
    num_classes_dict: {column_index: num_classes} 매핑
    """
    if p <= 0 or len(cat_idx) == 0:
        return cond

    device = cond.device
    cond_flipped = cond.clone()

    for col in cat_idx:
        num_classes = num_classes_dict[col]

        # ★ 클래수 수가 0 또는 1이면 flip 할 게 없으니 그냥 스킵
        if num_classes <= 1:
            # 디버그용으로 보고 싶으면:
            # print(f"[flip_categorical] Skip col {col}: num_classes={num_classes}")
            continue

        # (B, T) 현재 column의 값
        x_col = cond[:, :, col].long()  # 정수형으로 캐스팅

        rand = torch.rand_like(x_col.float(), device=device)
        flip_mask = rand < p  # True면 flip

        # 0 ~ num_classes-2에서 뽑고, 원래 값 이상이면 +1 해서 원래 class 건너뛰기
        random_base = torch.randint(
            low=0,
            high=num_classes - 1,  # num_classes >= 2라서 high >= 1 보장
            size=x_col.shape,
            device=device,
        )

        new_classes = random_base + (random_base >= x_col).long()

        x_new = x_col.clone()
        x_new[flip_mask] = new_classes[flip_mask]

        cond_flipped[:, :, col] = x_new.float()  # cond는 float32라면 float로 다시 캐스팅

    return cond_flipped

def multi_sliding_window(ori_data, seq_len, stride, threshold):
    
    # Preprocess the dataset
    temp_data = []; i = 0
    # Cut data by sequence length
    while i <= (len(ori_data[0]) - seq_len):
        _x = ori_data[:,i:i + seq_len,:]
        i+=stride
        temp_data.append(_x)

    data = torch.cat(temp_data, dim=0)
    
    return data

def create_missingness_pypots(X, rate, pattern, sub_seq_len, block_len, block_width):
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
        return mcar(X, rate)
    elif pattern == "subseq":
        return seq_missing(X, rate, sub_seq_len)
    elif pattern == "block":
        return block_missing(X, factor=rate, block_len=block_len, block_width=block_width)
    else:
        raise ValueError(f"Unknown missingness pattern: {pattern}")

def UnconGen(real_df, train_ratio, val_ratio, test_ratio, seq_len, stride):
    """
    Processes the dataframe, creates time embeddings, splits the data, 
    creates missingness, and prepares the data for imputation models.
    All outputs in the returned dictionary are PyTorch Tensors.
    """
    # --- 1. Create Time Embeddings ---
    time_info = cyclical_encode(real_df)

    # --- 2. Process Feature Data ---
    real_df1 = real_df.drop(['date'], axis=1)
    
    parser = pce.DataFrameParser().fit(real_df1, threshold=1)
    data = parser.transform()
    ori_data = torch.tensor(data.astype('float32'), dtype=torch.float32)

    # --- 3. Split Data and Time Info ---
    idx_train, idx_val, idx_test = make_split_indices(ori_data.shape[0], train_ratio, val_ratio, test_ratio)
    
    train_set_X, train_set_time = ori_data[idx_train], time_info[idx_train]
    val_set_X, val_set_time = ori_data[idx_val], time_info[idx_val]
    test_set_X, test_set_time = ori_data[idx_test], time_info[idx_test]    

    # --- 4. Apply Sliding Window to both Features and Time ---
    train_X = sliding_window(train_set_X, seq_len, stride)
    val_X = sliding_window(val_set_X, seq_len, stride)
    test_X = sliding_window(test_set_X, seq_len, stride)
    
    time_info_train = sliding_window(train_set_time, seq_len, stride)
    time_info_val = sliding_window(val_set_time, seq_len, stride)
    time_info_test = sliding_window(test_set_time, seq_len, stride)

    # Assemble the processed data into a dictionary, keeping everything as tensors
    processed_dataset = {
        "n_steps": seq_len,
        "n_features": train_X.shape[-1],
        "train_X_ori": train_X,
        "val_X_ori": val_X,
        "test_X_ori": test_X,
        "time_info_train": time_info_train,
        "time_info_val": time_info_val,
        "time_info_test": time_info_test
    }

    target_mask_train = torch.ones_like(train_X)
    target_mask_val = torch.ones_like(val_X)
    target_mask_test = torch.ones_like(test_X)

    response_train = train_X * target_mask_train
    cond_train = train_X * (1 - target_mask_train)
    response_val = val_X * target_mask_val
    cond_val = val_X * (1 - target_mask_val)
    response_test = test_X * target_mask_test
    cond_test = test_X * (1 - target_mask_test)

    processed_dataset.update({
        "parser": parser,
        "train_X": train_X,
        "val_X": val_X,
        "test_X": test_X,
        "target_mask_train": target_mask_train,
        "target_mask_val": target_mask_val,
        "target_mask_test": target_mask_test,
        "response_train": response_train,
        "response_val": response_val,
        "response_test": response_test,
        "cond_train": cond_train,
        "cond_val": cond_val,
        "cond_test": cond_test,
    })
        
    return processed_dataset

def ImpPypots(real_df, train_ratio, val_ratio, test_ratio, seq_len, stride, rate, pattern, sub_seq_len, block_len, block_width):
    """
    Processes the dataframe, creates time embeddings, splits the data, 
    creates missingness, and prepares the data for imputation models.
    All outputs in the returned dictionary are PyTorch Tensors.
    """
    # --- 1. Create Time Embeddings ---
    time_info = cyclical_encode(real_df)

    # --- 2. Process Feature Data ---
    real_df1 = real_df.drop(['date'], axis=1)
    
    parser = pce.DataFrameParser().fit(real_df1, threshold=1)
    data = parser.transform()
    ori_data = torch.tensor(data.astype('float32'), dtype=torch.float32)

    # --- 3. Split Data and Time Info ---
    n_samples = len(ori_data)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    train_set_X, train_set_time = ori_data[:train_end], time_info[:train_end]
    val_set_X, val_set_time = ori_data[train_end:val_end], time_info[train_end:val_end]
    test_set_X, test_set_time = ori_data[val_end:], time_info[val_end:]

    # --- 4. Apply Sliding Window to both Features and Time ---
    train_X_ori = sliding_window(train_set_X, seq_len, stride)
    val_X_ori = sliding_window(val_set_X, seq_len, stride)
    test_X_ori = sliding_window(test_set_X, seq_len, stride)
    
    time_info_train = sliding_window(train_set_time, seq_len, stride)
    time_info_val = sliding_window(val_set_time, seq_len, stride)
    time_info_test = sliding_window(test_set_time, seq_len, stride)

    # Assemble the processed data into a dictionary, keeping everything as tensors
    processed_dataset = {
        "n_steps": seq_len,
        "n_features": train_X_ori.shape[-1],
        "train_X_ori": train_X_ori,
        "val_X_ori": val_X_ori,
        "test_X_ori": test_X_ori,
        "time_info_train": time_info_train,
        "time_info_val": time_info_val,
        "time_info_test": time_info_test
    }

    if rate > 0:
        train_X = create_missingness_pypots(train_X_ori, rate, pattern, sub_seq_len, block_len, block_width)
        val_X = create_missingness_pypots(val_X_ori, rate, pattern, sub_seq_len, block_len, block_width)
        test_X = create_missingness_pypots(test_X_ori, rate, pattern, sub_seq_len, block_len, block_width)

        target_mask_train = torch.isnan(train_X).int()
        target_mask_val = torch.isnan(val_X).int()
        target_mask_test = torch.isnan(test_X).int()

        response_train = train_X_ori * target_mask_train
        cond_train = train_X_ori * (1 - target_mask_train)
        response_val = val_X_ori * target_mask_val
        cond_val = val_X_ori * (1 - target_mask_val)
        response_test = test_X_ori * target_mask_test
        cond_test = test_X_ori * (1 - target_mask_test)

        processed_dataset.update({
            "parser": parser,
            "train_X": train_X,
            "val_X": val_X,
            "test_X": test_X,
            "target_mask_train": target_mask_train,
            "target_mask_val": target_mask_val,
            "target_mask_test": target_mask_test,
            "response_train": response_train,
            "response_val": response_val,
            "response_test": response_test,
            "cond_train": cond_train,
            "cond_val": cond_val,
            "cond_test": cond_test,
        })
    else:
        print("Warning: Rate is 0, no missing values are artificially added.")
        processed_dataset["train_X"] = processed_dataset["train_X_ori"]
        processed_dataset["val_X"] = processed_dataset["val_X_ori"]
        processed_dataset["test_X"] = processed_dataset["test_X_ori"]

    return processed_dataset


def nan_after_timewindow(x: torch.Tensor, timewindow: int) -> torch.Tensor:    
    y = x.clone()
    y[:, timewindow:, :] = torch.nan
    return y

import torch
import numpy as np
from typing import Tuple

def make_split_indices(
    B: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    rng = np.random.default_rng(1234)
    perm = rng.permutation(B)

    n_train = int(B * train_frac)
    n_val   = int(B * val_frac)
    # Any remainder (from rounding) goes into the test split
    n_test  = B - n_train - n_val

    idx_train = perm[:n_train]
    idx_val   = perm[n_train : n_train + n_val]
    idx_test  = perm[n_train + n_val :]

    return idx_train, idx_val, idx_test

def ForePypots(real_df, train_ratio, val_ratio, test_ratio, seq_len, stride, timewindow):
    """
    Processes the dataframe, creates time embeddings, splits the data, 
    creates missingness, and prepares the data for imputation models.
    All outputs in the returned dictionary are PyTorch Tensors.
    """
    # --- 1. Create Time Embeddings ---
    time_info = cyclical_encode(real_df)

    # --- 2. Process Feature Data ---
    real_df1 = real_df.drop(['date'], axis=1)
    
    parser = pce.DataFrameParser().fit(real_df1, threshold=1)
    data = parser.transform()
    ori_data = torch.tensor(data.astype('float32'), dtype=torch.float32)
    
    # --- 3. Split Data and Time Info --- 
    ori_data = sliding_window(ori_data, seq_len, stride)   # [B, T, D]
    time_info = sliding_window(time_info, seq_len, stride)   # [B, T, D]

    idx_train, idx_val, idx_test = make_split_indices(ori_data.shape[0], train_ratio, val_ratio, test_ratio)
    
    train_X_ori, time_info_train = ori_data[idx_train], time_info[idx_train]
    val_X_ori, time_info_val = ori_data[idx_val], time_info[idx_val]
    test_X_ori, time_info_test = ori_data[idx_test], time_info[idx_test]

    # Assemble the processed data into a dictionary, keeping everything as tensors
    processed_dataset = {
        "n_steps": seq_len,
        "n_features": train_X_ori.shape[-1],
        "train_X_ori": train_X_ori,
        "val_X_ori": val_X_ori,
        "test_X_ori": test_X_ori,
        "time_info_train": time_info_train,
        "time_info_val": time_info_val,
        "time_info_test": time_info_test
    }

    train_X = nan_after_timewindow(train_X_ori, timewindow)
    val_X = nan_after_timewindow(val_X_ori, timewindow)
    test_X = nan_after_timewindow(test_X_ori, timewindow)

    target_mask_train = torch.isnan(train_X).int()
    target_mask_val = torch.isnan(val_X).int()
    target_mask_test = torch.isnan(test_X).int()

    response_train = train_X_ori * target_mask_train
    cond_train = train_X_ori * (1 - target_mask_train)
    response_val = val_X_ori * target_mask_val
    cond_val = val_X_ori * (1 - target_mask_val)
    response_test = test_X_ori * target_mask_test
    cond_test = test_X_ori * (1 - target_mask_test)

    processed_dataset.update({
        "parser": parser,
        "train_X": train_X,
        "val_X": val_X,
        "test_X": test_X,
        "target_mask_train": target_mask_train,
        "target_mask_val": target_mask_val,
        "target_mask_test": target_mask_test,
        "response_train": response_train,
        "response_val": response_val,
        "response_test": response_test,
        "cond_train": cond_train,
        "cond_val": cond_val,
        "cond_test": cond_test,
    })
    
    return processed_dataset

def TVMCG(real_df, train_ratio, val_ratio, test_ratio,
          seq_len, stride, column_list, target_column):
    """
    Processes the dataframe, creates time embeddings, splits the data, 
    creates missingness, and prepares the data for imputation models.
    All outputs in the returned dictionary are PyTorch Tensors.
    """
    # --- 1. Create Time Embeddings ---
    time_info = cyclical_encode(real_df)

    # --- 2. Process Feature Data ---
    real_df1 = real_df.drop(['date'], axis=1)
    
    parser = pce.DataFrameParser().fit(real_df1, threshold=1)
    data = parser.transform()               # (N, F) numpy array
    data = data.astype('float32')
    ori_data = torch.tensor(data, dtype=torch.float32)

    # >>> 여기서 conditional dtype + column name 자동 추출 <<<
    cond_info = infer_conditional_dtypes_with_names(
        parser=parser,
        target_column=target_column,
        encoded_data=data,
    )

    # --- 3. Split Data and Time Info ---
    idx_train, idx_val, idx_test = make_split_indices(
        ori_data.shape[0], train_ratio, val_ratio, test_ratio
    )
    
    train_set_X, train_set_time = ori_data[idx_train], time_info[idx_train]
    val_set_X,   val_set_time   = ori_data[idx_val],  time_info[idx_val]
    test_set_X,  test_set_time  = ori_data[idx_test], time_info[idx_test]    

    # --- 4. Sliding window ---
    train_X = sliding_window(train_set_X, seq_len, stride)
    val_X   = sliding_window(val_set_X,   seq_len, stride)
    test_X  = sliding_window(test_set_X,  seq_len, stride)
    
    time_info_train = sliding_window(train_set_time, seq_len, stride)
    time_info_val   = sliding_window(val_set_time,   seq_len, stride)
    time_info_test  = sliding_window(test_set_time,  seq_len, stride)

    processed_dataset = {
        "n_steps": seq_len,
        "n_features": train_X.shape[-1],
        "train_X_ori": train_X,
        "val_X_ori":   val_X,
        "test_X_ori":  test_X,
        "time_info_train": time_info_train,
        "time_info_val":   time_info_val,
        "time_info_test":  time_info_test,
    }

    # --- 5. Target column index 계산 ---
    # parser.column_name()을 기준으로 index 매핑
    column_list = parser.column_name()
    target_idx = [column_list.index(col) for col in target_column]

    # --- 6. Target mask 생성 ---
    target_mask_train = torch.zeros_like(train_X)
    target_mask_train[:, :, target_idx] = 1

    target_mask_val = torch.zeros_like(val_X)
    target_mask_val[:, :, target_idx] = 1

    target_mask_test = torch.zeros_like(test_X)
    target_mask_test[:, :, target_idx] = 1

    # --- 7. Response / Conditional 분리 ---
    response_train = train_X * target_mask_train
    cond_train     = train_X * (1 - target_mask_train)
    response_val   = val_X   * target_mask_val
    cond_val       = val_X   * (1 - target_mask_val)
    response_test  = test_X  * target_mask_test
    cond_test      = test_X  * (1 - target_mask_test)

    # --- 8. 결과 딕셔너리 업데이트 ---
    processed_dataset.update({
        "train_X": train_X,
        "val_X":   val_X,
        "test_X":  test_X,
        "target_mask_train": target_mask_train,
        "target_mask_val":   target_mask_val,
        "target_mask_test":  target_mask_test,
        "response_train": response_train,
        "response_val":   response_val,
        "response_test":  response_test,
        "cond_train": cond_train,
        "cond_val":   cond_val,
        "cond_test":  cond_test,

        # --- 메타데이터 dtype + 이름 정보 ---
        "parser": parser,
        "column_list": column_list,

        "cond_cont_idx":  cond_info["cond_cont_idx"],
        "cond_cat_idx":   cond_info["cond_cat_idx"],
        "cond_cont_cols": cond_info["cond_cont_cols"],
        "cond_cat_cols":  cond_info["cond_cat_cols"],
        "cat_num_classes_idx":   cond_info["cat_num_classes_idx"],
        "cat_num_classes_name":  cond_info["cat_num_classes_name"],

        # --- 새로 추가된 target 관련 정보 ---
        "target_column": target_column,
        "target_idx": target_idx,
    })
        
    return processed_dataset

    
from dataprovider import partition_multi_seq

def MultiImpPypots(real_df, train_ratio, val_ratio, test_ratio, seq_len, stride, rate, pattern, sub_seq_len, block_len, block_width, column_to_partition):
    """
    Processes the dataframe, creates time embeddings, splits the data, 
    creates missingness, and prepares the data for imputation models.
    All outputs in the returned dictionary are PyTorch Tensors.
    """
    
    # --- 1. Process Feature Data ---
    ori_data, time_info = partition_multi_seq(real_df, threshold=1, column_to_partition=column_to_partition) 

    # --- 2. Split Data and Time Info ---
    n_samples = len(ori_data)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    # --- 3. Split Data and Time Info ---
    train_X_ori, time_info_train = ori_data[:train_end], time_info[:train_end]
    val_X_ori, time_info_val = ori_data[train_end:val_end], time_info[train_end:val_end]
    test_X_ori, time_info_test = ori_data[val_end:], time_info[val_end:]

    # --- 4. Apply Sliding Window to both Features and Time ---
    #train_X_ori = multi_sliding_window(train_set_X, seq_len, stride, threshold=1)
    #val_X_ori = multi_sliding_window(val_set_X, seq_len, stride, threshold=1)
    #test_X_ori = multi_sliding_window(test_set_X, seq_len, stride, threshold=1)
    
    #time_info_train = multi_sliding_window(train_set_time, seq_len, stride, threshold=1)
    #time_info_val = multi_sliding_window(val_set_time, seq_len, stride, threshold=1)
    #time_info_test = multi_sliding_window(test_set_time, seq_len, stride, threshold=1)

    # Assemble the processed data into a dictionary, keeping everything as tensors
    processed_dataset = {
        "n_steps": seq_len,
        "n_features": train_X_ori.shape[-1],
        "train_X_ori": train_X_ori,
        "val_X_ori": val_X_ori,
        "test_X_ori": test_X_ori,
        "time_info_train": time_info_train,
        "time_info_val": time_info_val,
        "time_info_test": time_info_test
    }

    if rate > 0:
        train_X = create_missingness_pypots(train_X_ori, rate, pattern, sub_seq_len, block_len, block_width)
        val_X = create_missingness_pypots(val_X_ori, rate, pattern, sub_seq_len, block_len, block_width)
        test_X = create_missingness_pypots(test_X_ori, rate, pattern, sub_seq_len, block_len, block_width)

        target_mask_train = torch.isnan(train_X).int()
        target_mask_val = torch.isnan(val_X).int()
        target_mask_test = torch.isnan(test_X).int()

        response_train = train_X_ori * target_mask_train
        cond_train = train_X_ori * (1 - target_mask_train)
        response_val = val_X_ori * target_mask_val
        cond_val = val_X_ori * (1 - target_mask_val)
        response_test = test_X_ori * target_mask_test
        cond_test = test_X_ori * (1 - target_mask_test)

        processed_dataset.update({
            "train_X": train_X,
            "val_X": val_X,
            "test_X": test_X,
            "target_mask_train": target_mask_train,
            "target_mask_val": target_mask_val,
            "target_mask_test": target_mask_test,
            "response_train": response_train,
            "response_val": response_val,
            "response_test": response_test,
            "cond_train": cond_train,
            "cond_val": cond_val,
            "cond_test": cond_test,
        })
    else:
        print("Warning: Rate is 0, no missing values are artificially added.")
        processed_dataset["train_X"] = processed_dataset["train_X_ori"]
        processed_dataset["val_X"] = processed_dataset["val_X_ori"]
        processed_dataset["test_X"] = processed_dataset["test_X_ori"]

    return processed_dataset

