# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 08:59:16 2023
@author: Namjoon Suh
"""
import pandas as pd
import numpy as np
import random
from dython.nominal import theils_u

def detect_column_types(dataframe):
    continuous_columns = []
    categorical_columns = []

    for col in dataframe.columns:
        # Calculate the ratio of unique values to the total number of rows
        n_unique = dataframe[col].nunique()

        # If the ratio is below the threshold, consider the column as categorical
        if n_unique >= 1 and n_unique <= 25 or dataframe[col].dtype == 'int64':
            categorical_columns.append(col)
        elif dataframe[col].dtype == 'object':
            categorical_columns.append(col)
        else:
            continuous_columns.append(col)

    return continuous_columns, categorical_columns

def theils_u_mat(df):
  # Compute Theil's U-statistics between each pair of columns
  cate_columns = df.shape[1]
  theils_u_mat = np.zeros((cate_columns, cate_columns))

  for i in range(cate_columns):
      for j in range(cate_columns):
          theils_u_mat[i, j] = theils_u(df.iloc[:, i], df.iloc[:, j])

  return theils_u_mat

# See the post https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def correlation_ratio(categories, measurements):
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat)+1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(0,cat_num):
            cat_measures = measurements.iloc[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
        numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
        denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
        if numerator == 0:
            eta = 0.0
        else:
            eta = numerator/denominator
        return eta

def ratio_mat(df, continuous_columns, categorical_columns):
    rat_mat = pd.DataFrame(index=continuous_columns, columns=categorical_columns)
    
    if len(categorical_columns) == 0 or len(continuous_columns) == 0:
        return np.zeros(1)
    else:
        for cat_col in categorical_columns:
            for cont_col in continuous_columns:
                rat_mat[cat_col][cont_col] = correlation_ratio(df[cat_col], df[cont_col])
    return rat_mat.values

def fillNa_cont(df):
    for col in df.columns:
        mean_values = df[col].mean()
        df[col].fillna(mean_values, inplace=True)
    return df

def fillNa_cate(df):
    for col in df.columns:
        mode_values = df[col].mode()[0]
        df[col].fillna(mode_values, inplace=True)
    return df

def compute_correlation(df, continuous_columns, categorical_columns):

    num_mat = pd.DataFrame(df.iloc[:,continuous_columns])
    cat_mat = pd.DataFrame(df.iloc[:,categorical_columns])
   
    num_mat = fillNa_cont(num_mat)
    cat_mat = fillNa_cate(cat_mat)
    
    pearson_sub_matrix = np.corrcoef(num_mat, rowvar = False)
    theils_u_matrix = theils_u_mat(cat_mat)
    correl_ratio_mat = ratio_mat(df, continuous_columns, categorical_columns)
   
    return (pearson_sub_matrix, theils_u_matrix, correl_ratio_mat)

def final_correlation(real_df, syn_df):
    
    continuous_columns, categorical_columns = detect_column_types(real_df)
    
    real_pearson, real_theils, real_ratio = compute_correlation(real_df, continuous_columns, categorical_columns)
    syn_pearson, syn_theils, syn_ratio = compute_correlation(syn_df, continuous_columns, categorical_columns)
    
    result = []
    
    result.append(np.linalg.norm(real_pearson - syn_pearson))
    result.append(np.linalg.norm(real_theils - syn_theils))
    result.append(np.linalg.norm(real_ratio - syn_ratio))
    
    return result

######################################################################################################################

#strings_set = {'abalone', 'adult', 'Bean', 'Churn_Modelling','faults', 'HTRU', 'indian_liver_patient', 
#               'insurance', 'Magic', 'nursery', 'Obesity', 'News', 'Shoppers', 'Titanic', 'wilt'}
#Model = {'TabAutoDiff'}

#for model in Model:
#    print(model)
    
#    string_idx = 0
#    Data_L2_Dist = np.zeros((len(strings_set),3))
    
#    for string in strings_set:
#        print(string)
#        real_file = f'C:/Users/Namjoon Suh/Desktop/Tabular/Real-data/{string}.csv'
#        real_df = pd.read_csv(real_file)
#        continuous_columns, categorical_columns = detect_column_types(real_df)

#        real_pearson, real_theils, real_ratio = compute_correlation(real_df, continuous_columns, categorical_columns)
            
#        L2_Dist = np.zeros( (10, 3) )
        
#        for i in range(1,11):
#            syn_file = f'C:/Users/Namjoon Suh/Desktop/Tabular/Synthetic-data/{model}/{string}/AutoDiff_{string}{i}.csv'
            
#            syn_df = pd.read_csv(syn_file)
            
#            syn_pearson, syn_theils, syn_ratio = compute_correlation(syn_df, continuous_columns, categorical_columns)

#            L2_Dist[i-1,0] = np.linalg.norm(real_pearson - syn_pearson)
#            L2_Dist[i-1,1] = np.linalg.norm(real_theils - syn_theils)
#            L2_Dist[i-1,2] = np.linalg.norm(real_ratio - syn_ratio)

#        Data_L2_Dist[string_idx,:] = np.mean(L2_Dist, axis=0)
#        string_idx += 1
        
#    Data_L2_Dist = np.nan_to_num(Data_L2_Dist, nan=0)
#    print(np.mean(Data_L2_Dist, axis=0))


