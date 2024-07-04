# TimeAutoDiff
This is the Github repository for code implementation of the TimeAutoDiff model (https://arxiv.org/pdf/2406.16028).
Thanks for your interest in our model! 

## File descriptions in ``Model Code Folder''.
In the Model Code file, we have the following py-files.
 
 - **DP.py**: This file is for splitting the dataset. We have two data splitters: one for single-sequence data (i.e., splitData), and another for multi-sequence data (i.e., partition_multi_seq).
   - For splitData(real_df, seq_len, threshold), it splits the data with windows of size ''seq_len''. 
   - For partition_multi_seq(real_df, threshold, column_to_partition), it splits the data with respect to the labels of the entity in the ``column_to_partition'' variable. Here, we assume the lengths of the sequence from each entity are the same.
   - You can put ``threshold=1'' for both functions.
  
 - **process_edited.py**: This file pre-processes your tabular data into the tensor format. Users can refer to Section 2 of our paper for detailed descriptions of pre- and post-processing steps.
 - **timeautoencoder.py**: This file is the implementation of VAE in TimeAutoDiff. Look at Figure 2 for a schematic overview of the architecture. It also has the code for training VAE. 
 - **timediffusion.py**: This file is the implementation of the Diffusion model in TimeAutoDiff. Look at Figure 3 for a schematic overview of the architecture. It also has the code for the training & sampling process. 
 - **timediffusion_cond_label.py**: This file is the implementation of the conditional_Diffusion model in TimeAutoDiff for entity conditional generation. 
