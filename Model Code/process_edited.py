import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
import pandas as pd

__all__ = ['StandardScaler', 'LabelEncoder', 'FreqLabelEncoder', 'DataFrameParser', 'SingleDataset']

class StandardScaler(object):
    def __init__(self):
        self.max = None
        self.min = None

    def fit(self, x):
        self.max, self.min = np.max(x), np.min(x)
        return self

    def transform(self, x):
        standardized = (np.array(x) - self.min) / (self.max - self.min + 1e-7)
        imputed = np.nan_to_num(standardized)
        return imputed
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def fit_invert(self, x, encoded_col):
        self.max, self.min = np.max(x), np.min(x)
        return encoded_col * (self.max - self.min + 1e-7) + self.min
    
class LabelEncoder(object):
    def __init__(self):
        self.mapping = dict()

    def __len__(self):
        return len(self.mapping)

    def fit(self, x):
        np.random.seed(200)
        self.mapping = {v: i for i, v in enumerate(set(x))}
        return self
    
    def fit_bin_int(self, x):
        np.random.seed(200)
        self.bin_int_encoder = x
        return self
    
    def fit_transform(self, x):
        np.random.seed(200)
        return np.array(list(map(self.mapping.__getitem__, x)))

    def fit_int_transform(self, x):
        np.random.seed(200)
        return np.array(x)
    
    def fit_invert(self, x, encoded_col):
        np.random.seed(200)
        self.mapping = {v: i for i, v in enumerate(set(x))}
        inverse_mapping = {v: k for k, v in self.mapping.items()}
        return np.array(list(inverse_mapping[i] for i in encoded_col))
    
class FreqLabelEncoder(object):
    ''' A composition of label encoding and frequency encoding. Not reversible. '''
    def __init__(self):
        self.freq_counts = None

    def __len__(self):
        return len(self.lbl_encoder)

    def fit(self, x):
        self.freq_counts = Counter(x)
        self.lbl_encoder = LabelEncoder().fit(self.freq_counts.values())
        return self

    def transform(self, x):
        freq_encoded = np.array(list(map(self.freq_counts.__getitem__, x)))
        lbl_encoded = self.lbl_encoder.transform(freq_encoded)
        return lbl_encoded

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class DataFrameParser(object):
    """ Transform dataframe to numpy array for modeling. Not a reversible process.
        It will reshuffle the columns according to datatype: binary->categorical->numerical
        Encoding:
            + Binary variables will be coded as 0, 1
            + Small categorical variables will be label encoded as integers.
            + Categorical variables with large cardinalities will go through count/frequency encoding before label encoding.
            + Numerical will be standardized.
        NaN handling:
            + Fill with mean for numerical. # TODO: Need handling of NaN in categorical? If present in training data is fine.
    """
    def __init__(self, max_cardinality=25):
        self.max_cardinality = max_cardinality
        self.binary_columns = list()
        self.categorical_columns = list() # variable name to mode mapping
        self._cards = list()
        self.numerical_columns = list()
        self.need_freq_encoding = set()
        self.need_int_encoding = list()
        self.need_bin_int = list()
        
    def fit(self, dataframe, threshold):
        
        self.new_dataframe = dataframe.copy()
        
        self._original_order = self.new_dataframe.columns.tolist()
        self._original_column_to_dtype = column_to_dtype = self.new_dataframe.dtypes.to_dict()

        # sort through columns in dataframe.
        for column, datatype in column_to_dtype.items():
            if datatype in ['O', '<U32']:
                cardinality = self.new_dataframe[column].nunique(dropna=False)
                if cardinality <= 2:
                    self.binary_columns.append(column)
                elif cardinality > self.max_cardinality:
                    self.numerical_columns.append(column)
                else:
                    self.categorical_columns.append(column)
            
            elif np.issubdtype(datatype, np.integer) and self.new_dataframe[column].nunique() <= 2:
                self.binary_columns.append(column)
                self.need_bin_int.append(column)

            elif np.issubdtype(datatype, np.float64) and self.new_dataframe[column].nunique() <= 2:
                self.binary_columns.append(column)
                self.need_bin_int.append(column)
                
            elif np.issubdtype(datatype, np.integer) and self.new_dataframe[column].nunique() <= 25 \
                    and self.new_dataframe[column].nunique() >= 3:
                self.categorical_columns.append(column)
                self.need_int_encoding.append(column)
            
            elif np.issubdtype(datatype, np.float64) and self.new_dataframe[column].nunique() <= 25 \
                    and self.new_dataframe[column].nunique() >= 3:
                self.categorical_columns.append(column)
                self.need_int_encoding.append(column)
            
            else:
                self.numerical_columns.append(column)   
                counts = self.new_dataframe[column].value_counts()
                repeated_entries = counts[counts > threshold * len(self.new_dataframe[column])].index.tolist()
                
                if len(repeated_entries) == 1:
                    new_column_name = 'Binary_' + column
                    self.binary_columns.append('Binary_' + column)
                    self.need_bin_int.append('Binary_' + column)
                    self.new_dataframe[new_column_name] = \
                        self.new_dataframe[column].apply(lambda x: repeated_entries.index(x) + 1 if x in repeated_entries else 0)
                    self.new_dataframe[new_column_name] = self.new_dataframe[new_column_name].astype(int)

                if len(repeated_entries) >= 2 and len(repeated_entries) <= 25:
                    new_column_name = 'Cate_' + column   
                    self.categorical_columns.append('Cate_' + column)
                    self.need_int_encoding.append('Cate_' + column)
                    self.new_dataframe[new_column_name] = \
                        self.new_dataframe[column].apply(lambda x: repeated_entries.index(x) + 1 if x in repeated_entries else 0)

                    self.new_dataframe[new_column_name] = self.new_dataframe[new_column_name].astype(int)

        self._column_order = self.binary_columns + self.categorical_columns + self.numerical_columns

        # fit encoders
        encoders = dict()
        for column in self.binary_columns:
            if self.new_dataframe[column].dtype == int:
                encoders[column] = LabelEncoder().fit_bin_int(self.new_dataframe[column].astype(int))
            else:
                encoders[column] = LabelEncoder().fit(self.new_dataframe[column].astype(str))

        for column in self.categorical_columns:
            if column in self.need_freq_encoding:
                encoders[column] = FreqLabelEncoder().fit(self.new_dataframe[column].astype(str))
            elif column in self.need_int_encoding:
                encoders[column] = LabelEncoder().fit(self.new_dataframe[column].astype(int))
            else:
                encoders[column] = LabelEncoder().fit(self.new_dataframe[column].astype(str))
            self._cards.append(len(encoders[column]))

        for column in self.numerical_columns:
            encoders[column] = StandardScaler().fit(self.new_dataframe[column].astype(float))

        self._embeds = [int(min(600, 1.6 * card ** .5)) for card in self._cards]
        self.encoders = encoders
        self.new_df = dataframe
        
        return self
    
    def transform(self):
        df = self.new_dataframe[self._column_order].copy()
        for column, encoder in self.encoders.items():
            if column in self.numerical_columns:
                df[column] = encoder.fit_transform(df[column])
            
            elif column in self.need_int_encoding:
                df[column] = encoder.fit_transform(df[column].astype(int))
            
            elif column in self.need_bin_int:
                df[column] = encoder.fit_int_transform(df[column].astype(int))
            
            else:
                df[column] = encoder.fit_transform(df[column].astype(str))
                
        df.columns = self._column_order

        return df.values

    def invert_fit(self, encoded_table):
        decoded_table = encoded_table[self._column_order].copy()
        for column, encoder in self.encoders.items():
            if column in self.numerical_columns:
                decoded_table[column] = StandardScaler().fit_invert(self.new_dataframe[column], encoded_table[column])
                
            elif column in self.binary_columns and np.issubdtype(self.new_dataframe[column].dtype, np.integer) == True: 
                decoded_table[column] = encoded_table[column].astype(int)
            
            elif column in self.need_int_encoding:
                decoded_table[column] = LabelEncoder().fit_invert(self.new_dataframe[column].astype(int), encoded_table[column])
            
            else:
                decoded_table[column] = LabelEncoder().fit_invert(self.new_dataframe[column].astype(str), encoded_table[column])
                
        return decoded_table
    
    @property
    def n_bins(self): return len(self.binary_columns)

    @property
    def n_cats(self): return len(self.categorical_columns)

    @property
    def n_nums(self): return len(self.numerical_columns)

    @property
    def cards(self): return self._cards

    @property
    def embeds(self): return self._embeds

    def datatype_info(self): return {'n_bins': self.n_bins, 'n_cats': self.n_cats, 'n_nums': self.n_nums, 'cards': self._cards}
    
    def column_name(self): return self._column_order

####################################################################################################################
device = 'cuda'

def convert_to_tensor(org_df, gen_output, threshold, data_size, seq_len):
    import torch.nn.functional as F

    def sigmoid_threshold(logits):
        sigmoid_output = torch.sigmoid(logits).to(device)
        threshold_output = torch.where(sigmoid_output > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device)).to(device)
        return threshold_output

    def softmax_with_max(predictions):
        # Applying softmax function
        probabilities = F.softmax(predictions.to(device), dim=2).to(device)
        # Getting the index of the maximum element
        max_indices = torch.argmax(probabilities.to(device), dim=2).to(device)
        return max_indices

    parser_conv = DataFrameParser().fit(org_df, threshold)
    datatype_info = parser_conv.datatype_info()

    n_bins = datatype_info['n_bins']; n_cats = datatype_info['n_cats']
    n_nums = datatype_info['n_nums']; cards = datatype_info['cards']
    
    synth_data = torch.tensor([]).to(device)
    
    if n_bins != 0:
        bin_tensor = torch.empty(data_size, seq_len, n_bins).to(device)
        for idx in range(n_bins):
            bin_tensor[:,:,idx] = sigmoid_threshold(gen_output['bins'][:,:,idx].detach()).to(torch.int64)
        synth_data = bin_tensor
        
    if len(cards) != 0:
        cat_tensor = torch.empty(data_size, seq_len, n_cats).to(device)
        for idx in range(len(cards)):
            cat_tensor[:,:,idx] = softmax_with_max(gen_output['cats'][idx].detach()).to(torch.int64).to(device)
        synth_data = torch.cat((synth_data, cat_tensor),dim=2)
        
    if n_nums != 0:
        num_tensor = torch.empty(data_size, seq_len, n_nums).to(device)
        num_tensor = gen_output['nums'].detach().to(device)
        synth_data = torch.cat((synth_data, num_tensor),dim=2).to(device)
    
    return synth_data

####################################################################################################################
def convert_to_table(org_df, synth_data, threshold):
    
    import pandas as pd
    
    B, L, K = synth_data.shape
    t_np = synth_data.cpu().reshape(B * L, K).numpy() # convert to Numpy array
    syn_df = pd.DataFrame(t_np)
    
    parser = DataFrameParser().fit(org_df, threshold)
    real_df = pd.DataFrame(parser.transform())
    
    real_df.columns = parser.column_name()
    syn_df.columns = parser.column_name()
    #syn_df = parser.invert_fit(syn_df)   
    
    # Detect pairs of columns with column names 'Cate_X' and 'X'
    bin_column_pairs = [(col.replace('Binary_', ''), col) for col in syn_df.columns if col.startswith('Binary_')]

    ####################################################################################################################
    # Replace the entries in 'X' columns with the corresponding entries in 'Bin_X' columns
    for col, bin_col in bin_column_pairs:
        counts = real_df[col].value_counts()
        repeated_entries = counts[counts > threshold * len(real_df[col])].index.tolist()

        # Create a mapping between array labels and real numbers
        label_to_number = {label: number for label, number in enumerate(repeated_entries, 1)}
        syn_df[col][syn_df[bin_col] == 1] = label_to_number.get(1, 1)

    # Detect pairs of columns with column names 'Cate_X' and 'X'
    cate_column_pairs = [(col.replace('Cate_', ''), col) for col in syn_df.columns if col.startswith('Cate_')]

    ####################################################################################################################
    # Replace the entries in 'X' columns with the corresponding entries in 'Cate_X' columns
    for col, cat_col in cate_column_pairs:
        counts = real_df[col].value_counts()
        repeated_entries = counts[counts > threshold * len(real_df[col])].index.tolist()

        # Create a mapping between array labels and real numbers
        label_to_number = {label: number for label, number in enumerate(repeated_entries, 1)}
        array_replaced = np.array([label_to_number.get(label, label) for label in syn_df[cat_col]]).astype(int)
        syn_df[col] = np.where(array_replaced != 0, array_replaced, syn_df[col].to_numpy())
    
    ## Drop the 'Cate_X' columns and 'Bin_X' columns
    syn_df = syn_df.drop(columns=[cate_col for col, cate_col in cate_column_pairs])
    syn_df = syn_df.drop(columns=[bin_col for col, bin_col in bin_column_pairs])

    orig_column = org_df.columns
    syn_df = syn_df.reindex(columns=orig_column)    
    feat_num = syn_df.shape[1]
    
    return torch.tensor(syn_df.values).reshape(B, L, feat_num)