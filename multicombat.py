# MultiComBat developed by Hannah Horng (hhorng@seas.upenn.edu)
# Developed from the neuroComBat package by Nick Cullen and JP Fortin
from __future__ import absolute_import, print_function
import pandas as pd
import numpy as np
import numpy.linalg as la
import math
import copy

def multiCombat(dat,
                covars,
                batch_list,
                ref_batch,
                categorical_cols=None,
                continuous_cols=None,
                eb=True,
                parametric=True,
                mean_only=False):
    """
    Run MultiComBat to remove scanner effects in multi-site imaging data

    Arguments
    ---------
    dat : a pandas data frame or numpy array
        - neuroimaging data to correct with shape = (features, samples) e.g. cortical thickness measurements, image voxels, etc

    covars : a pandas data frame w/ shape = (samples, covariates)
        - contains all batch/scanner covariates as well as additional covariates (optional) that should be preserved during harmonization.
        
    batch_list : list of strings
        - contains all batch (scanner) column names in covars (e.g. ['scanner', 'kernel'])

    categorical_cols : list of strings
        - specifies column names in covars data frame of categorical variables to be preserved during harmonization (e.g. ["sex", "disease"])

    continuous_cols : list of strings
        - indicates column names in covars data frame of continuous variables to be preserved during harmonization (e.g. ["age"])

    eb : should Empirical Bayes be performed?
        - True by default

    parametric : should parametric adjustements be performed?
        - True by default

    mean_only : should only be the mean adjusted (no scaling)?
        - False by default

    ref_batch : list
        - batch combination to be used as reference for batch adjustment.
        - should contain batch memberships as encoded in the covars dataframe
            - e.g. if 'Vendor 1' and 'Location 1' are in covars, then ref_batch = ['Vendor 1','Location 1']
        
    Returns
    -------
    A dictionary of length 3:
    - data: A numpy array with the same shape as `dat` which has now been MultiComBat-harmonized
    - estimates: A dictionary of the MultiComBat estimates used for harmonization
    - info: A dictionary of the inputs needed for MultiComBat harmonization
    """
    ##############################
    ### CLEANING UP INPUT DATA ###
    ##############################
    if not isinstance(covars, pd.DataFrame):
        raise ValueError('covars must be pandas dataframe -> try: covars = pandas.DataFrame(covars)')

    if not isinstance(categorical_cols, (list,tuple)):
        if categorical_cols is None:
            categorical_cols = []
        else:
            categorical_cols = [categorical_cols]
    if not isinstance(continuous_cols, (list,tuple)):
        if continuous_cols is None:
            continuous_cols = []
        else:
            continuous_cols = [continuous_cols]

    covar_labels = np.array(covars.columns)
    covars = np.array(covars, dtype='object') 

    if isinstance(dat, pd.DataFrame):
        dat = np.array(dat, dtype='float32')



    ##############################

    # get column indices for relevant variables
    bat_cols = [np.where(covar_labels==b_var)[0][0] for b_var in batch_list]
    cat_cols = [np.where(covar_labels==c_var)[0][0] for c_var in categorical_cols]
    num_cols = [np.where(covar_labels==n_var)[0][0] for n_var in continuous_cols]

    # convert batch col to integer
    batch_combo_freq = pd.DataFrame(pd.DataFrame(covars).groupby(bat_cols).size()).rename(columns={0:'count'})
    if ref_batch is None:
        # reference batch is required for this implementation
        # set it to the most frequent batch combination
        ref_batch=list(batch_combo_freq.idxmax())
    else:
        ref_indices = np.array([i for i, tupl in enumerate(list(map(tuple, covars[:, bat_cols]))) if tupl == tuple(ref_batch)])
        if ref_indices.shape[0]==0:
            print('[neuroCombat] batch.ref not found. Setting to most frequent batch combination.')
            ref_batch=list(batch_combo_freq.idxmax())
    
    ref_level = []
    for a in bat_cols:
        covars[:, a] = np.unique(covars[:, a],return_inverse=True)[-1]
        ref_level += [covars[int(ref_indices[0]), a]]
    ref_level = tuple(ref_level)
    
    # create dictionary that stores batch info (split)
    batch_levels_split = []
    sample_per_batch_split = []
    batch_info_split = []
    for i in range(len(bat_cols)):
        (batch_levels, sample_per_batch) = np.unique(covars[:, bat_cols[i]],return_counts=True)
        batch_levels_split += [(i, level) for level in list(batch_levels)]
        sample_per_batch_split += list(sample_per_batch)
        batch_levels = batch_levels[batch_levels != ref_level[i]]
        batch_info_split += [list(np.where(covars[:,bat_cols[i]]==idx)[0]) for idx in batch_levels] # contains batch_idxs for everything not in reference
    
    # create dictionary that stores batch info (combo)
    level_combo_freq = pd.DataFrame(pd.DataFrame(covars).groupby(bat_cols).size()).rename(columns={0:'count'})
    batch_levels_combo = list(level_combo_freq.index)
    sample_per_batch_combo = level_combo_freq.values    

    # create design matrix
    print('[neuroCombat] Creating design matrices')
    design_split = make_design_matrix_split(covars, bat_cols, cat_cols, num_cols, ref_level)
    design_combo = make_design_matrix_combo(covars, bat_cols, cat_cols, num_cols, ref_level)
    
    # create mapping from combo notation to just relative to the reference
    ref_map = {}
    if isinstance(batch_levels_combo[0], int):
        batch_levels_combo = [tuple([i]) for i in batch_levels_combo]
    for a in batch_levels_combo:
        batch_ind = [i for i, tupl in enumerate(list(map(tuple, covars[:, bat_cols]))) if tupl == a][0]
        ref_map[tuple(design_split[batch_ind,:len(batch_levels_combo)-len(bat_cols)+1])] = a

    # FOR STANDARDIZATION
    # batch tuples formatted as [(batch1...batchn), ... (batch1...batchn)]
    info_dict_combo = {
        'batch_levels_combo': batch_levels_combo,
        'ref_level': ref_level,
        'n_batch': len(batch_levels_combo),
        'n_batch_var': len(bat_cols),
        'n_sample': int(covars.shape[0]),
        'sample_per_batch': sample_per_batch_combo,
        'batch_info_combo': [[i for i, tupl in enumerate(list(map(tuple, covars[:, bat_cols]))) if tupl == idx] for idx in batch_levels_combo],
        'design_combo': design_combo,
        'design_split': design_split,
        'batch_info_split': batch_info_split,
        'ref_map': ref_map
    }
    
    # check that each combo has >1 sample
    batch_combo_sizes = [len(i) for i in info_dict_combo['batch_info_combo']]
    if 1 in batch_combo_sizes:
        raise ValueError('Batch combo contains 1 sample: unable to estimate variance')
    
   
    # standardize data across features
    print('[neuroCombat] Standardizing data across features')
    s_data, s_mean, v_pool, mod_mean = standardize_across_features(dat, design_combo, info_dict_combo)
    
    # fit L/S models and find priors
    print('[neuroCombat] Fitting L/S model and finding priors')
    LS_dict = fit_LS_model_and_find_priors(s_data, design_split, info_dict_combo, mean_only)

    # find parametric adjustments
    if eb:
        if parametric:
            print('[neuroCombat] Finding parametric adjustments')
            gamma_star, delta_star = find_parametric_adjustments(s_data, LS_dict, info_dict_combo, mean_only)
        else:
            print('[neuroCombat] Finding non-parametric adjustments')
            gamma_star, delta_star = find_non_parametric_adjustments(s_data, LS_dict, info_dict_combo, mean_only)
    else:
        print('[neuroCombat] Finding L/S adjustments without Empirical Bayes')
        gamma_star, delta_star = find_non_eb_adjustments(s_data, LS_dict, info_dict_combo)
    
    # exporting gamma_star formatted in combination
#    gamma_star_combo = np.zeros(s_data.shape)
#    gamma_star_combo = np.dot(batch_design[batch_idxs,:], gamma_star).T
    
    # adjust data
    print('[neuroCombat] Final adjustment of data')
    bayes_data, gamma_star_combo, delta_star_combo = adjust_data_final(s_data, design_split, gamma_star, delta_star, 
                                                                       s_mean, mod_mean, v_pool, info_dict_combo, dat)

    bayes_data = np.array(bayes_data)
    estimates = {'batches': info_dict_combo['batch_levels_combo'], 'var.pooled': v_pool, 'stand.mean': s_mean, 
                 'mod.mean': mod_mean, 'gamma.star': gamma_star, 'gamma.star.combo': gamma_star_combo, 
                 'delta.star': delta_star, 'delta.star.combo': delta_star_combo}
    estimates = {**LS_dict, **estimates, }

    return {
        'data': bayes_data,
        'estimates': estimates,
        'info': info_dict_combo
    }


def make_design_matrix_split(Y, bat_cols, cat_cols, num_cols, ref_level):
    """
    Return Matrix containing the following parts:
        - one-hot matrix of batch variable (full)
        - one-hot matrix for each categorical_cols (removing the first column)
        - column for each continuous_cols
    """
    def to_categorical(y, nb_classes=None):
        if not nb_classes:
            nb_classes = np.max(y)+1
        Y = np.zeros((len(y), nb_classes))
        for i in range(len(y)):
            Y[i, y[i]] = 1.
        return Y
    
    hstack_list = []

    ### batch one-hot ###
    # initialize
    batch_onehot_combo = np.ones([len(Y), 1]) # reference batch is all ones
    for a in range(len(bat_cols)):
    # convert batch column to integer in case it's string
        batch = np.unique(Y[:, bat_cols[a]],return_inverse=True)[-1]
        batch_onehot = to_categorical(batch, len(np.unique(batch)))
        batch_onehot = np.delete(batch_onehot, ref_level[a], 1) #remove the reference
        batch_onehot_combo = np.append(batch_onehot_combo, batch_onehot, axis=1)
    hstack_list.append(batch_onehot_combo)

    ### categorical one-hots ###
    for cat_col in cat_cols:
        cat = np.unique(np.array(Y[:,cat_col]),return_inverse=True)[1]
        cat_onehot = to_categorical(cat, len(np.unique(cat)))[:,1:]
        hstack_list.append(cat_onehot)

    ### numerical vectors ###
    for num_col in num_cols:
        num = np.array(Y[:,num_col],dtype='float32')
        num = num.reshape(num.shape[0],1)
        hstack_list.append(num)

    design = np.hstack(hstack_list)
    return design

def make_design_matrix_combo(Y, bat_cols, cat_cols, num_cols, ref_level):
    """
    Return Matrix containing the following parts:
        - one-hot matrix of batch variable (full)
        - one-hot matrix for each categorical_cols (removing the first column)
        - column for each continuous_cols
    """
    def to_categorical(y, nb_classes=None):
        if not nb_classes:
            nb_classes = np.max(y)+1
        Y = np.zeros((len(y), nb_classes))
        for i in range(len(y)):
            Y[i, y[i]] = 1.
        return Y
    
    hstack_list = []

    ### batch one-hot ###
    # initialize
    level_combo_freq = pd.DataFrame(pd.DataFrame(Y).groupby(bat_cols).size()).rename(columns={0:'count'})
    batch_levels = list(level_combo_freq.index)
    if isinstance(batch_levels[0], int):
        batch_levels = [tuple([i]) for i in batch_levels]
    # create mapping of combinations to numbers
    batch_map = {}
    for i in range(len(batch_levels)):
        batch_map[batch_levels[i]] = i
    # convert batch column to integer in case it's string
    batch = np.unique([batch_map.get(val) for val in list(map(tuple, Y[:, bat_cols]))], return_inverse=True)[-1]
    batch_onehot = to_categorical(batch, len(np.unique(batch)))
    batch_onehot[:,batch_map.get(ref_level)] = np.ones(batch_onehot.shape[0])
    hstack_list.append(batch_onehot)

    ### categorical one-hots ###
    for cat_col in cat_cols:
        cat = np.unique(np.array(Y[:,cat_col]),return_inverse=True)[1]
        cat_onehot = to_categorical(cat, len(np.unique(cat)))[:,1:]
        hstack_list.append(cat_onehot)

    ### numerical vectors ###
    for num_col in num_cols:
        num = np.array(Y[:,num_col],dtype='float32')
        num = num.reshape(num.shape[0],1)
        hstack_list.append(num)

    design = np.hstack(hstack_list)
    return design


def standardize_across_features(X, design, info_dict):
    n_batch = info_dict['n_batch']
    n_sample = info_dict['n_sample']
    sample_per_batch = info_dict['sample_per_batch']
    batch_info = info_dict['batch_info_combo']
    ref_level = info_dict['ref_level']
    batch_levels = info_dict['batch_levels_combo']

    def get_beta_with_nan(yy, mod):
        wh = np.isfinite(yy)
        mod = mod[wh,:]
        yy = yy[wh]
        B = np.dot(np.dot(la.inv(np.dot(mod.T, mod)), mod.T), yy.T)
        return B

    betas = []
    for i in range(X.shape[0]):
        betas.append(get_beta_with_nan(X[i,:], design))
    B_hat = np.vstack(betas).T
    
    #B_hat = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), X.T)
    ref_ind = [i for i, tupl in enumerate(list(batch_levels)) if tupl == ref_level][0]
    grand_mean = np.transpose(B_hat[ref_ind,:])
    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, n_sample)))
    #var_pooled = np.dot(((X - np.dot(design, B_hat).T)**2), np.ones((n_sample, 1)) / float(n_sample))

    X_ref = X[:,batch_info[ref_ind]]
    design_ref = design[batch_info[ref_ind],:]
    n_sample_ref = sample_per_batch[ref_ind]
    var_pooled = np.dot(((X_ref - np.dot(design_ref, B_hat).T)**2), np.ones((int(n_sample_ref), 1)) / float(n_sample_ref))

    var_pooled[var_pooled==0] = np.median(var_pooled!=0)
    
    mod_mean = 0
    if design is not None:
        tmp = copy.deepcopy(design)
        tmp[:,range(0,n_batch)] = 0
        mod_mean = np.transpose(np.dot(tmp, B_hat))
    ######### Continue here. 


    #tmp = np.array(design.copy())
    #tmp[:,:n_batch] = 0
    #stand_mean  += np.dot(tmp, B_hat).T

    s_data = ((X- stand_mean - mod_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, n_sample))))

    return s_data, stand_mean, var_pooled, mod_mean

def aprior(delta_hat):
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat,ddof=1)
    return (2 * s2 +m**2) / float(s2)

def bprior(delta_hat):
    m = delta_hat.mean()
    s2 = np.var(delta_hat,ddof=1)
    return (m*s2+m**3)/s2

def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2*n*g_hat+d_star * g_bar) / (t2*n+d_star)

def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

def convert_zeroes(x):
    x[x==0] = 1
    return x

def fit_LS_model_and_find_priors(s_data, design, info_dict, mean_only):
    n_batch = info_dict['n_batch']
    n_batch_var = info_dict['n_batch_var']
    batch_info = info_dict['batch_info_combo']
    ref_map = info_dict['ref_map']
    batch_levels = info_dict['batch_levels_combo']
    
    batch_design = design[:,:(n_batch-n_batch_var+1)]
    K = np.unique(batch_design, axis=0) # equivalent to K in  the derivation
    B = np.empty([len(K), len(s_data)])
    for i in range(len(K)):
        batch_level_idx = [j for j, tupl in enumerate(list(batch_levels)) if tupl == ref_map[tuple(K[i,:])]][0]
        batch_idxs = batch_info[batch_level_idx]
        B[i,:] = np.mean(s_data[:,batch_idxs], axis=1).T # STOPPING POINT 3/9/22
    
    # gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)
    gamma_hat = la.inv(K.T @ K) @ (K.T @ B)
    # gamma_hat = gamma_hat_sol - gamma_hat_sol[0,:]
    
    
    if mean_only:
        delta_hat_matrix = np.ones([len(gamma_hat), s_data.shape[0]])
    else:
        B_delta = np.empty([len(K), len(s_data)])
        for i in range(len(K)):
            batch_level_idx = [j for j, tupl in enumerate(list(batch_levels)) if tupl == ref_map[tuple(K[i,:])]][0]
            batch_idxs = batch_info[batch_level_idx]
            B_delta[i,:] = np.log(np.var(s_data[:,batch_idxs], axis=1, ddof=1).T)
        # need to make a correction for variance in the reference batch not equal to 1
        # K_delta = K[:len(gamma_hat),:]
        # B_delta = B_delta - B_delta[0,:]
        delta_hat_sol = la.inv(K.T @ K) @ (K.T @ B_delta)
        delta_hat_sol[1:,:] = delta_hat_sol[1:,:] + delta_hat_sol[0,:]
        delta_hat_matrix = np.exp(delta_hat_sol)
    delta_hat = [delta_hat_matrix[i, :] for i in range(delta_hat_matrix.shape[0])]
    
    delta_hat = list(map(convert_zeroes,delta_hat))
    gamma_bar = np.mean(gamma_hat, axis=1) 
    t2 = np.var(gamma_hat,axis=1, ddof=1)

    if mean_only:
        a_prior = None
        b_prior = None
    else:
        a_prior = list(map(aprior, delta_hat))
        b_prior = list(map(bprior, delta_hat))

    LS_dict = {}
    LS_dict['gamma_hat'] = gamma_hat
    LS_dict['delta_hat'] = delta_hat
    LS_dict['gamma_bar'] = gamma_bar
    LS_dict['t2'] = t2
    LS_dict['a_prior'] = a_prior
    LS_dict['b_prior'] = b_prior
    return LS_dict

#Helper function for parametric adjustements:
def it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
    n = (1 - np.isnan(sdat)).sum(axis=1)
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    change = 1
    count = 0
    while change > conv:
        g_new = postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((sdat - np.dot(g_new.reshape((g_new.shape[0], 1)), np.ones((1, sdat.shape[1])))) ** 2).sum(axis=1)
        d_new = postvar(sum2, n, a, b)

        change = max((abs(g_new - g_old) / g_old).max(), (abs(d_new - d_old) / d_old).max())
        g_old = g_new #.copy()
        d_old = d_new #.copy()
        count = count + 1
    adjust = (g_new, d_new)
    return adjust 



#Helper function for non-parametric adjustements:
def int_eprior(sdat, g_hat, d_hat):
    r = sdat.shape[0]
    gamma_star, delta_star = [], []
    for i in range(0,r,1):
        g = np.delete(g_hat,i)
        d = np.delete(d_hat,i)
        x = sdat[i,:]
        n = x.shape[0]
        j = np.repeat(1,n)
        A = np.repeat(x, g.shape[0])
        A = A.reshape(n,g.shape[0])
        A = np.transpose(A)
        B = np.repeat(g, n)
        B = B.reshape(g.shape[0],n)
        resid2 = np.square(A-B)
        sum2 = resid2.dot(j)
        LH = 1/(2*math.pi*d)**(n/2)*np.exp(-sum2/(2*d))
        LH = np.nan_to_num(LH)
        gamma_star.append(sum(g*LH)/sum(LH))
        delta_star.append(sum(d*LH)/sum(LH))
    adjust = (gamma_star, delta_star)
    return adjust


def find_parametric_adjustments(s_data, LS, info_dict, mean_only):
    batch_info  = info_dict['batch_info_split'] 

    gamma_star_list, delta_star_list = [], []
    for i, batch_idxs in enumerate(batch_info):
        if mean_only:
            gamma_star_list.append(postmean(LS['gamma_hat'][i+1], LS['gamma_bar'][i+1], 1, 1, LS['t2'][i+1]))
            delta_star_list.append(np.repeat(1, s_data.shape[0]))
        else:
            temp = it_sol(s_data[:,batch_idxs], LS['gamma_hat'][i+1],
                        LS['delta_hat'][i+1], LS['gamma_bar'][i+1], LS['t2'][i+1], 
                        LS['a_prior'][i+1], LS['b_prior'][i+1])
            gamma_star_list.append(temp[0])
            delta_star_list.append(temp[1])
    
    gamma_star = np.zeros(LS['gamma_hat'].shape)
    delta_star = np.ones(LS['gamma_hat'].shape)
    gamma_star[1:,:] = np.array(gamma_star_list)
    delta_star[1:,:] = np.array(delta_star_list)


    return gamma_star, delta_star

def find_non_parametric_adjustments(s_data, LS, info_dict, mean_only):
    batch_info  = info_dict['batch_info_split'] 
    
    gamma_star_list, delta_star_list = [], []
    for i, batch_idxs in enumerate(batch_info):
        if mean_only:
            LS['delta_hat'][i+1] = np.repeat(1, s_data.shape[0])
        temp = int_eprior(s_data[:,batch_idxs], LS['gamma_hat'][i+1],
                    LS['delta_hat'][i+1])

        gamma_star_list.append(temp[0])
        delta_star_list.append(temp[1])

    gamma_star = np.zeros(LS['gamma_hat'].shape)
    delta_star = np.ones(LS['gamma_hat'].shape)
    gamma_star[1:,:] = np.array(gamma_star_list)
    delta_star[1:,:] = np.array(delta_star_list) 

    return gamma_star, delta_star

def find_non_eb_adjustments(s_data, LS, info_dict):
    gamma_star = np.zeros(LS['gamma_hat'].shape)
    delta_star = np.ones(LS['gamma_hat'].shape)
    gamma_star[1:,:] = np.array(LS['gamma_hat'])
    delta_star[1:,:] = np.array(LS['delta_hat'])
 
    return gamma_star, delta_star

def adjust_data_final(s_data, design, gamma_star, delta_star, stand_mean, mod_mean, var_pooled, info_dict, dat):
    sample_per_batch = info_dict['sample_per_batch']
    n_batch = info_dict['n_batch']
    n_batch_var = info_dict['n_batch_var']
    n_sample = info_dict['n_sample']
    batch_info = info_dict['batch_info_combo']
    batch_levels = info_dict['batch_levels_combo']
    ref_level = info_dict['ref_level']

    batch_design = design[:,:(n_batch-n_batch_var+1)]

    bayesdata = s_data
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)
    gamma_star_combo = np.zeros((s_data.shape[0], len(batch_info)))
    delta_star_combo = np.zeros((s_data.shape[0], len(batch_info)))

    for j, batch_idxs in enumerate(batch_info):
        batch_combo = np.repeat(batch_design[batch_idxs[0],:].reshape((len(delta_star), 1)), len(bayesdata), axis=1)
        dsq_1 = np.multiply(batch_combo, delta_star) # a little more complicated because it's a product (not a sum)
        dsq_1[dsq_1 == 0] = 1
        dsq_2 = np.sqrt(np.prod(dsq_1, axis=0))
        dsq = dsq_2.reshape((len(dsq_2), 1))
        denom = np.dot(dsq, np.ones((1, int(sample_per_batch[j]))))
        numer = np.array(bayesdata[:,batch_idxs] - np.dot(batch_design[batch_idxs,:], gamma_star).T) # location effects are purely additive so dot product is fine
        gamma_star_combo[:,j] = np.dot(batch_design[batch_idxs[0],:], gamma_star).T
        delta_star_combo[:,j] = np.prod(dsq_1, axis=0)
        
        bayesdata[:,batch_idxs] = numer / denom

    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, n_sample))) + stand_mean + mod_mean
    
    ref_idx = [j for j, tupl in enumerate(list(batch_levels)) if tupl == ref_level][0]
    bayesdata[:, batch_info[ref_idx]] = dat[:,batch_info[ref_idx]]

    return bayesdata, gamma_star_combo, delta_star_combo




def neuroCombatFromTraining(dat,
                            batch,
                            estimates):
    """
    Combat harmonization with pre-trained ComBat estimates [UNDER DEVELOPMENT]

    Arguments
    ---------
    dat : a pandas data frame or numpy array for the new dataset to harmonize
        - rows must be identical to the training dataset
    
    batch : DataFrame specifying scanner/batch variables for the new dataset
        - all scanners/batches must also be present in the training dataset

    estimates : dictionary of MultiComBat estimates from a previously-harmonized dataset
        - should be in the same format as MultiCombat(...)['estimates']
        
    Returns
    -------
    A dictionary of length 2:
    - data: A numpy array with the same shape as `dat` which has now been ComBat-harmonized
    - estimates: A dictionary of the ComBat estimates used for harmonization
    """
    print("[neuroCombatFromTraining] In development ...\n")
    if np.min(batch)[0] > 0:
        batch = batch - np.min(batch)[0]
    batch = np.array(batch[['batch0', 'batch1']], dtype="str")
    new_levels = np.unique(batch, axis=0)
    old_levels = np.array(estimates['batches'], dtype="str")
    missing_levels = np.array(list(set(map(tuple, new_levels)) - set(map(tuple, old_levels))))
    if missing_levels.shape[0] != 0:
        raise ValueError("The batches " + str(missing_levels) +
                         " are not part of the training dataset")

    wh = [int(np.where((old_levels == x).all(axis=1))[0]) if x in old_levels else None for x in batch]

    

    var_pooled = estimates['var.pooled']
    stand_mean = estimates['stand.mean'][:, 0]
    mod_mean = estimates['mod.mean']
    # gamma_star = estimates['gamma.star']
    gamma_star_combo = estimates['gamma.star.combo'].T
    # delta_star = estimates['delta.star']
    delta_star_combo = estimates['delta.star.combo'].T
    n_array = dat.shape[1]   
    stand_mean = stand_mean+mod_mean.mean(axis=1)
    
    stand_mean = np.transpose([stand_mean, ]*n_array)
    bayesdata = np.subtract(dat, stand_mean)/np.sqrt(var_pooled)
    
    #gamma = np.transpose(np.repeat(gamma_star, repeats=2, axis=0))
    #delta = np.transpose(np.repeat(delta_star, repeats=2, axis=0))
    gamma = np.transpose(gamma_star_combo[wh,:]) # reshuffle by the adjustments
    delta = np.transpose(delta_star_combo[wh,:])
    bayesdata = np.subtract(bayesdata, gamma)/np.sqrt(delta)
    
    bayesdata = bayesdata*np.sqrt(var_pooled) + stand_mean
    out = {
        'data': bayesdata,
        'estimates': estimates
    }
    return out