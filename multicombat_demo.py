# Code demonstration for MultiComBat harmonization
# Written by Hannah Horng (hhorng@seas.upenn.edu)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import multicombat as multi
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np

## Specify the input and output directories
"""
Some information on the files contained in "demo_data"
    - multipoint_original.csv: 5 simulated normally distributed features with NO batch added
    - multipoint_batch.csv: batch assignments for each sample in the simulated data
    - multipoint_batch_eff.csv: values of the added batch effects for each of the batches
    - multipoint_data.csv: multipoint_original.csv with batch effects for 2 simulated batch variables 
        (each with 5 batch groups) added
        - This is essentially multipoint_original + multipoint_batch_eff (but indexed to the batch 
            assignments in multipoint_batch)
"""
data_path = "C:/Users/horng/OneDrive/Documents/GitHub/multicombat/demo_data/"
save_path = "C:/Users/horng/OneDrive/Documents/GitHub/multicombat/demo_output/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Read in data
data_df = pd.read_csv(os.path.join(data_path, 'multipoint_data.csv'))
dat = data_df.iloc[:, 1:].T

# Read in batch
batch = pd.read_csv(os.path.join(data_path, 'multipoint_batch.csv'))
covars = pd.DataFrame()
n_batchvar = batch.shape[1] - 1

for a in range(n_batchvar):
    covars['batch' + str(a)] = batch.iloc[:, a + 1]
batch_list = list(covars.columns)

le = LabelEncoder()
batch_combo = batch.iloc[:, 1:].apply(tuple, axis=1)
batch_combo_encode = le.fit_transform(batch_combo.astype(str))
covars['batch_combo'] = batch_combo_encode

# Separate into train/test
"""
The setup for this allocates 2 samples from each combination to the training data (to simulate
under-sampling) and the remaining samples from each combination to the testing data (to better
reflect the underlying distribution). 
"""
total_size = 22
batch_sort = covars.sort_values(by='batch_combo')
pull_ind = [[i, i + 1] for i in [x * total_size for x in range(int(len(batch_sort) / total_size))]]
pull_ind = [i for sublist in pull_ind for i in sublist]
train_ind = [list(batch_sort.index)[i] for i in pull_ind]
train_ind.sort()
test_ind = list(set(batch_sort.index) - set(train_ind))

pd.Series(train_ind).to_csv(save_path + 'index_train.csv') # Save the train/test indices for reproducibility
pd.Series(test_ind).to_csv(save_path + 'index_test.csv')

caseno_train = data_df.iloc[train_ind, 0]
dat_train = data_df.iloc[train_ind, 1:].reset_index(drop=True).T
covars_train = covars.iloc[train_ind, :].reset_index(drop=True)

caseno_test = data_df.iloc[test_ind, 0]
dat_test = data_df.iloc[test_ind, 1:].reset_index(drop=True).T
covars_test = covars.iloc[test_ind, :].reset_index(drop=True)

# set the reference
ref_batch = [3, 3]

# run MultiComBat (train)
output_train = multi.multiCombat(dat_train, covars_train, batch_list, ref_batch,
                                 categorical_cols=[],
                                 continuous_cols=[], mean_only=True, parametric=True)
# run MultiComBat (test)
output_test = multi.neuroCombatFromTraining(dat_test,
                                            covars_test[batch_list],
                                            output_train['estimates'])

# Save the outputs (train)
save_path_train = save_path + 'train/'
if not os.path.exists(save_path_train):
    os.makedirs(save_path_train)
output_df_train = pd.DataFrame.from_records(output_train['data'].T)
output_df_train.columns = dat_train.index
output_df_train.to_csv(save_path_train + 'features_combat.csv')

# Save the outputs (test)
save_path_test = save_path + 'test/'
if not os.path.exists(save_path_test):
    os.makedirs(save_path_test)
output_df_test = pd.DataFrame.from_records(output_test['data'].T)
output_df_test.columns = dat_test.index
output_df_test.to_csv(save_path_test+'features_combat.csv')
