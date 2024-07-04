# TimeAutoDiff
This is the Github repository for code implementation of the TimeAutoDiff model (https://arxiv.org/pdf/2406.16028).
Thanks for your interest in our model! 

## File descriptions in ``Model Code Folder''.
In the Model Code file, we have the following py-files.
 
 - **DP.py**: This file is for splitting the dataset. We have two data splitters: one for single-sequence data (i.e., splitData), and another for multi-sequence data (i.e., partition_multi_seq).
   - For splitData(real_df, seq_len, threshold), it splits the data with window size ''seq_len'' 

