import numpy as np
import sys
import os
from scipy.special import expit, logit, logsumexp
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import random
import copy
import itertools

from sklearn.covariance import ShrunkCovariance

# This file is a translation of utils_tch.py to use the MLX framework.

# --- NumPy-based Implementations (largely unchanged) ---

def getModalityTools(prior):
    tool_dict = {}
    if prior == "gaussian-means":
        tool_dict['init'] = np_init_gaussianmeans
        tool_dict['llr'] = np_gaussianmeans_llr
        tool_dict['minibatch'] = np_minibatch_gaussianmeans
        tool_dict['mle'] = np_mle_gaussianmeans
    elif prior == "multivariate":
        tool_dict['init'] = np_init_multivariate
        tool_dict['llr'] = np_multivariate_llr
        tool_dict['minibatch'] = np_minibatch_multivariate
        tool_dict['mle'] = np_mle_multivariate
    else:
        exit(f"unsupported prior {prior}")
    return tool_dict

def np_gaussianmeans_llr(xin, old_paras):
    old_means = old_paras[0]
    xp = np.expand_dims(xin, axis=1) - old_means  # (bs, nc, xdim)
    return -0.5 * np.sum(np.log(2 * np.pi) + xp**2, axis=2)

def np_multivariate_llr(xin, old_paras):
    old_means = old_paras[0]
    old_cov = old_paras[1]
    xdim = xin.shape[1]
    mat_inv = np.linalg.inv(old_cov)  # (NC,xd,xd)
    mat_det = np.linalg.det(old_cov)  # (Nc,)
    xdiff = np.expand_dims(xin, 1) - old_means
    kernel = np.expand_dims(xdiff, 2) @ mat_inv[None, ...] @ np.expand_dims(xdiff, 3)  # (bs,nc,1,1)
    return -0.5 * (xdim * np.log(2 * np.pi) + mat_det[None, :] + np.squeeze(kernel))

def np_init_gaussianmeans(xin, zargs, nclusters, cluster_idx):
    return [xin[cluster_idx]]

def np_init_multivariate(xin, zargs, nclusters, cluster_idx):
    para_means = xin[cluster_idx]
    xdim = xin.shape[1]
    para_cov = np.array([np.eye(xdim).astype("float32")] * nclusters)  # (nc,xdim,xdim)
    return [para_means, para_cov]

def np_minibatch_gaussianmeans(old_paras, zmask, xin, old_cnts):
    old_means = old_paras[0]
    zsum = np.sum(zmask, axis=0)
    ratio = zsum / (zsum + old_cnts)
    delta_x = np.sum(np.expand_dims(xin, 1) * zmask[..., None], axis=0) / zsum[:, None]
    new_means = ratio[:, None] * old_means + (1 - ratio)[:, None] * delta_x
    new_means = np.where((zsum == 0)[:, None], old_means, new_means)
    return [new_means]

def np_minibatch_multivariate(old_paras,zmask,xin,old_cnts):
    old_means = old_paras[0]
    old_cov = old_paras[1]
    nclusters = zmask.shape[1]
    zargs = np.argmax(zmask,axis=1)
    zsum = np.sum(zmask,axis=0)
    ratio = zsum/(zsum+old_cnts)
    #xp = np.expand_dims(xin,axis=1)*zmask - old_means
    sol = ShrunkCovariance()
    mean_shrunk =[]
    cov_shrunk =[]
    for ic in range(nclusters):
        sel_idx = zargs == ic
        if np.sum(sel_idx) > 0:
            cov = sol.fit(xin[sel_idx])
            mean_shrunk.append(cov.location_)
            cov_shrunk.append(cov.covariance_)
        else:
            mean_shrunk.append(old_means[ic])
            cov_shrunk.append(old_cov[ic])

    mean_shrunk = np.array(mean_shrunk)
    cov_shrunk = np.array(cov_shrunk)
    new_means = ratio[None,:]*old_means + (1-ratio)[None,:] * mean_shrunk
    new_means = np.where((zsum==0)[:,None],old_means,new_means)

    mean_diff = old_means -new_means
    sigma_diff = np.expand_dims(mean_diff,axis=2) @ np.expand_dims(mean_diff,axis=1)
    new_cov = ratio[:,None,None] * old_cov + (1-ratio)[:,None,None] * cov_shrunk \
             -((zsum+old_cnts)/(zsum+old_cnts-1))[:,None,None] * sigma_diff
    new_cov = np.where((zsum<1)[:,None,None],old_cov,new_cov)
    return [new_means, new_cov]

def np_mle_gaussianmeans(old_paras, zmask, xin):
    old_means = old_paras[0]
    zsum = np.sum(zmask, axis=0)
    new_means = np.expand_dims(xin, 1) * zmask[..., None] / (zsum[:, None])
    new_means = np.where((zsum == 0)[:, None], old_means, new_means)
    return [new_means]

def np_mle_multivariate(old_paras, zmask, xin):
    old_means = old_paras[0]
    old_cov = old_paras[1]
    new_means = np_mle_gaussianmeans([old_means], zmask, xin)[0]
    xdiff = (np.expand_dims(xin, axis=1) - new_means)
    xot = np.expand_dims(xdiff, axis=-1) @ np.expand_dims(xdiff, axis=2)
    zsum = np.sum(zmask, axis=0)
    new_cov = np.sum(xot * zmask[..., None, None], axis=0) / (zsum[:, None, None] - 1)
    new_cov = np.where((zsum > 1)[:, None, None], new_cov, old_cov)
    return [new_means, new_cov]

# --- MLX Framework-specific Functions ---

def getModalityMlx(prior):
    """Factory function for MLX-based modality tools."""
    tool_dict = {}
    if prior == 'gaussian-means':
        tool_dict['init'] = kest_init_gaussianMeans
        tool_dict['llr'] = mlx_gaussianMeanLLR
        tool_dict['minibatch'] = mlx_gaussianMeanUpdate
        tool_dict['mle'] = ML_gaussianMeanUpdate
    elif prior == "multivariate":
        tool_dict['init'] = kest_init_multivariate
        tool_dict['llr'] = mlx_K_multivariateLLR
        tool_dict['minibatch'] = mlx_multivariateUpdate
        tool_dict['mle'] = ML_multivariateUpdate
    elif prior == "gaussian":
        raise NotImplementedError("The 'gaussian' prior has not been implemented in MLX yet.")
    elif prior == "bernoulli":
        raise NotImplementedError("The 'bernoulli' prior has not been implemented in MLX yet.")
    else:
        exit(f"unsupported prior {prior}")
    return tool_dict

def print_network(net: nn.Module):
    num_params = sum(p.size for _, p in net.parameters().items())
    print(net)
    print(f'Total number of parameters: {num_params}')

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    mx.random.seed(seed)

def getSafeSaveName(savepath, basename, extension=".pkl"):
    repeat_cnt = 0
    safename = copy.copy(basename)
    while os.path.isfile(os.path.join(savepath, safename + extension)):
        repeat_cnt += 1
        safename = f"{basename}_{repeat_cnt}"
    return safename

def mlx_gaussianLLR(xin, paras):
    sum_axis = tuple(range(2, len(xin.shape) + 1))
    mean = paras[0]
    logvar = paras[1]
    xp = mx.expand_dims(xin, axis=1)
    test_llr = -0.5 * mx.sum(np.log(2 * np.pi) + logvar + mx.square(xp - mean) * mx.exp(-logvar), axis=sum_axis)
    if mx.any(test_llr > 0):
        print(test_llr)
        sys.exit()
    return test_llr

def mlx_gaussianMeanLLR(xin, paras):
    return mlx_gaussianLLR(xin, [paras[0], mx.zeros_like(paras[0])])

def mlx_K_gaussianLLR(xin, paras):
    pmeans = paras[0]
    plvars = paras[1]
    calc = -0.5 * (np.log(2 * np.pi) + plvars + (mx.exp(-plvars)) * (mx.square(mx.expand_dims(xin, axis=1) - pmeans)))
    sum_axis = tuple(range(2, len(calc.shape)))
    return mx.sum(calc, axis=sum_axis)

def mlx_K_bernoulliLLR(xin, paras):
    plogits = paras[0]
    xp = mx.expand_dims(xin.astype(mx.float32), 1)
    calc = xp * plogits - nn.functional.softplus(plogits)
    sum_axis = tuple(range(2, len(calc.shape)))
    return mx.sum(calc, axis=sum_axis)

def mlx_llr_normalization(weights, cluster_llr):
    return cluster_llr - mx.logsumexp(mx.log(weights) + cluster_llr, axis=1, keepdims=True)

def calc_GaussianKL(z_logits):
    z_mean = z_logits[0].reshape(z_logits[0].shape[0], -1)
    z_logvar = z_logits[1].reshape(z_logits[1].shape[0], -1)
    return -0.5 * mx.mean(mx.sum(z_logvar + 1 - mx.square(z_mean) - mx.exp(z_logvar), axis=1))

def calc_BernoulliKL(z_logits):
    z_llr = z_logits[0].reshape(z_logits[0].shape[0], -1)
    return mx.mean(mx.sum(np.log(2) + z_llr * mx.sigmoid(z_llr) - nn.functional.softplus(z_llr), axis=1))

def mlx_gaussianMeanUpdate(old_paras, zmask, xin, old_cnts):
    batch_size, nclusters = zmask.shape
    old_means = old_paras[0]
    xp = mx.expand_dims(xin, axis=1)
    zsum = mx.sum(zmask, axis=0)
    para_dim = [nclusters] + [1] * (len(xin.shape) - 1)
    zmask_res = zmask.astype(mx.float32).reshape(batch_size, *para_dim)
    mean_diff = mx.sum(xp * zmask_res, axis=0) / zsum.reshape(*para_dim)
    ra_res = mx.expand_dims(old_cnts / (old_cnts + zsum), -1)
    new_means = old_means * ra_res + mean_diff * (1.0 - ra_res)
    return [mx.where(mx.expand_dims(zsum == 0, -1), old_means, new_means)]

def ML_gaussianMeanUpdate(old_paras, zmask, xin):
    """
    Maximum Likelihood update for Gaussian means.
    This function is not yet implemented for MLX.
    """
    raise NotImplementedError("ML_gaussianMeanUpdate is not yet implemented for MLX.")

def kest_init_gaussianMeans(xin, zargs, nclusters, cluster_idx):
    return [xin[cluster_idx]]

def kest_init_bernoulli(xin, zargs, nclusters, cluster_idx):
    tmp_oh = nn.functional.one_hot(zargs, num_classes=nclusters)
    new_cnt = mx.sum(tmp_oh, axis=0)
    z_expand = tmp_oh.astype(mx.float32).reshape(tmp_oh.shape[0], tmp_oh.shape[1], *([1] * (len(xin.shape) - 1)))
    para_shape = (nclusters,) + tuple([1] * (len(xin.shape) - 1))
    xp = mx.expand_dims(xin.astype(mx.float32), 1)
    ber_p = mx.sum(xp * z_expand, axis=0) / new_cnt.astype(mx.float32).reshape(*para_shape)
    
    # Stability for zero counts
    ber_p = mx.where(mx.expand_dims(new_cnt == 0, -1), mx.random.uniform(low=0.0, high=1.0, shape=ber_p.shape), ber_p)
    ber_p = mx.clip(ber_p, 1e-4, 1 - 1e-4)
    return [mx.log(ber_p / (1 - ber_p))]

def mlx_K_multivariateLLR(xin, paras, **kwargs):
    eta_mu = paras[0]
    dd = eta_mu.shape[1]
    cov = paras[1]
    
    xdiff = mx.expand_dims(xin, 1) - mx.expand_dims(eta_mu, 0) # (N, K, d)
    
    matinv = kwargs.get('inv', mx.linalg.inv(cov))
    matdet = kwargs.get('det', mx.linalg.det(cov))
    
    kernel2 = -0.5 * mx.squeeze(mx.expand_dims(xdiff, 2) @ mx.expand_dims(matinv, 0) @ mx.expand_dims(xdiff, 3))
    const1 = -0.5 * dd * np.log(2 * np.pi)
    kernel1 = -0.5 * mx.log(matdet)
    
    return const1 + kernel2 + mx.expand_dims(kernel1, 0)

def kest_init_multivariate(xin, zargs, nclusters, cluster_idx):
    mu = xin[cluster_idx]
    dd = xin.shape[1]
    cov = mx.broadcast_to(mx.expand_dims(mx.eye(dd), 0), (nclusters, dd, dd)).astype(mx.float32)
    return [mu, cov]

def mlx_multivariateUpdate(old_paras, zmask, xin, old_cnts):
    batch_size, nclusters = zmask.shape
    old_means = old_paras[0]
    old_cov = old_paras[1]
    xp = mx.expand_dims(xin, 1)
    zsum = mx.sum(zmask, axis=0)
    
    para_dim_mean = (nclusters,) + tuple([1] * (len(xin.shape) - 1))
    zmask_res_mean = zmask.astype(mx.float32).reshape(batch_size, *para_dim_mean)
    mean_diff_val = mx.sum(xp * zmask_res_mean, axis=0) / zsum.reshape(*para_dim_mean)
    
    ra_res = mx.expand_dims(old_cnts / (old_cnts + zsum), -1)
    new_means = old_means * ra_res + mean_diff_val * (1.0 - ra_res)
    new_means = mx.where(mx.expand_dims(zsum == 0, -1), old_means, new_means)
    
    mean_diff_ot = mx.expand_dims(old_means - new_means, 2) @ mx.expand_dims(old_means - new_means, 1)
    x_shift = xp - old_means
    est_cov = mx.expand_dims(x_shift, 3) @ mx.expand_dims(x_shift, 2)
    
    para_dim_cov = (nclusters, 1, 1)
    zmask_res_cov = zmask.astype(mx.float32).reshape(batch_size, nclusters, 1, 1)
    cov_diff = mx.sum(est_cov * zmask_res_cov, axis=0) / (zsum - 1).reshape(*para_dim_cov)
    
    ratio1 = (old_cnts - 1) / (zsum + old_cnts - 1)
    ra_1 = mx.expand_dims(mx.expand_dims(ratio1, -1), -1)
    ra_1_cmpl = 1.0 - ra_1
    ratio_neg = (old_cnts + zsum) / (old_cnts + zsum - 1)
    ra_neg = mx.expand_dims(mx.expand_dims(ratio_neg, -1), -1)
    
    new_cov = old_cov * ra_1 + ra_1_cmpl * cov_diff - ra_neg * mean_diff_ot
    new_cov = mx.where(mx.expand_dims(mx.expand_dims(zsum > 1, -1), -1), new_cov, old_cov)
    
    return [new_means, new_cov]

def ML_multivariateUpdate(old_paras, zmask, xin):
    old_means = old_paras[0]
    old_cov = old_paras[1]
    zsum = mx.sum(zmask, axis=0)
    nclusters = len(zsum)
    batch_size = zmask.shape[0]

    para_dim_mean = (nclusters,) + tuple([1] * (len(xin.shape) - 1))
    zsh_res_mean = zmask.astype(mx.float32).reshape(batch_size, *para_dim_mean)
    xp = mx.expand_dims(xin, 1)
    new_means = mx.sum(xp * zsh_res_mean, axis=0) / zsum.reshape(*para_dim_mean)
    new_means = mx.where(mx.expand_dims(zsum == 0, -1), old_means, new_means)

    est_cov = xp - new_means
    raw_cov = mx.expand_dims(est_cov, 3) @ mx.expand_dims(est_cov, 2)

    para_dim_cov = (nclusters, 1, 1)
    zvec_res = zmask.astype(mx.float32).reshape(batch_size, nclusters, 1, 1)
    new_cov = mx.sum(raw_cov * zvec_res, axis=0) / (zsum - 1).reshape(*para_dim_cov)
    new_cov = mx.where(mx.expand_dims(mx.expand_dims(zsum > 1, -1), -1), new_cov, old_cov)
    return [new_means, new_cov]

def getSetDict(nview):
    set_dict = {"uniset": set(np.arange(nview)), "tuple_list": []}
    for widx in range(0, int(np.floor(nview / 2))):
        for item in itertools.combinations(set_dict['uniset'], widx + 1):
            cmpl_tuple = tuple(set_dict['uniset'] - set(item))
            if (item, cmpl_tuple) not in set_dict['tuple_list'] and (cmpl_tuple, item) not in set_dict['tuple_list']:
                set_dict['tuple_list'].append((item, cmpl_tuple))
    return set_dict
