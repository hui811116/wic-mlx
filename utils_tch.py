import numpy as np
import sys
import os
from scipy.special import expit, logit, logsumexp
import torch
from torch import nn
from torch.nn import functional as F
import random
import copy
import itertools

from sklearn.covariance import ShrunkCovariance

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
		exit("unsupported prior {:}".format(prior))
	return tool_dict

def getModalityTch(prior):
	tool_dict = {}
	if prior == 'gaussian-means':
		tool_dict['init'] = kest_init_gaussianMeans
		tool_dict['llr'] = tch_gaussianMeanLLR
		tool_dict['minibatch'] = tch_gaussianMeanUpdate
		tool_dict['mle'] = ML_gaussianMeanUpdate
	elif prior == "multivariate":
		tool_dict['init'] = kest_init_multivariate
		tool_dict['llr'] = tch_K_multivariateLLR
		tool_dict['minibatch'] = tch_multivariateUpdate
		tool_dict['mle'] = ML_multivariateUpdate
	elif prior == "gaussian":
		tool_dict['init']= kest_init_gaussian
		tool_dict['llr'] = tch_K_gaussianLLR
		tool_dict['minibatch'] = tch_gaussianUpdate
		tool_dict['mle'] = ML_gaussianUpdate
	elif prior == "bernoulli":
		tool_dict['init'] = kest_init_bernoulli
		tool_dict['llr'] = tch_K_bernoulliLLR
		tool_dict['minibatch'] = tch_bernoulliUpdate
		tool_dict['mle'] = ML_bernoulliUpdate
	else:
		exit("unsupported prior {:}".format(prior))
	return tool_dict

def np_gaussianmeans_llr(xin,old_paras):
	old_means = old_paras[0]
	xp = np.expand_dims(xin,axis=1) - old_means #(bs, nc, xdim)
	return -0.5 * np.sum(np.log(2*np.pi) + xp**2,axis=2)
def np_multivariate_llr(xin,old_paras):
	old_means = old_paras[0]
	old_cov = old_paras[1]
	xdim = xin.shape[1]
	mat_inv = np.linalg.inv(old_cov) # (NC,xd,xd)
	mat_det = np.linalg.det(old_cov) # (Nc,)
	xdiff = np.expand_dims(xin,1) - old_means
	#print(xdiff.shape)
	#print(mat_inv.shape)
	kernel = np.expand_dims(xdiff,2)@ mat_inv[None,...] @ np.expand_dims(xdiff,3) # (bs,nc,1,1)
	# TOTAL dim (bs, NC)
	return -0.5 * (xdim * np.log(2*np.pi) + mat_det[None,:] + np.squeeze(kernel))

def np_init_gaussianmeans(xin,zargs,nclusters,cluster_idx):
	return [xin[cluster_idx]]
def np_init_multivariate(xin,zargs,nclusters,cluster_idx):
	para_means = xin[cluster_idx]
	xdim = xin.shape[1]
	para_cov = np.array([np.eye(xdim).astype("float32")]*nclusters) #(nc,xdim,xdim)
	return [para_means, para_cov]

def np_minibatch_gaussianmeans(old_paras,zmask,xin,old_cnts):
	old_means = old_paras[0]
	zsum = np.sum(zmask,axis=0) # 
	ratio = zsum/ (zsum + old_cnts)
	delta_x = np.sum(np.expand_dims(xin,1)*zmask[...,None],axis=0) / zsum[:,None]
	new_means = ratio[:,None] * old_means + (1-ratio)[:,None] * delta_x
	new_means = np.where((zsum==0)[:,None],old_means,new_means)
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
		cov = sol.fit(xin[sel_idx])
		mean_shrunk.append(cov.location_)
		cov_shrunk.append(cov.covariance_)
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

def np_mle_gaussianmeans(old_paras,zmask,xin):
	#nclusters = zmask.shape[1]
	old_means = old_paras[0]
	zsum = np.sum(zmask,axis=0)
	new_means = np.expand_dims(xin,1)*zmask[...,None]/(zsum[:,None])
	new_means = np.where((zsum==0)[:,None],old_means,new_means)
	return [new_means]
def np_mle_multivariate(old_paras,zmask,xin):
	#nclusters = zmask.shape[1]
	#zargs = np.argmax(zmask,axis=1)
	old_means = old_paras[0]
	old_cov =old_paras[1]
	new_means = np_mle_gaussianmeans([old_means],zmask,xin)[0]
	xdiff = (np.expand_dims(xin,axis=1) - new_means)
	xot = np.expand_dims(xdiff,axis=-1) @ np.expand_dims(xdiff,axis=2)
	zsum = np.sum(zmask,axis=0)
	new_cov = np.sum(xot * zmask[...,None,None],axis=0)/(zsum[:,None,None]-1)
	new_cov = np.where((zsum>1)[:,None,None],new_cov,old_cov)
	return [new_means, new_cov]
	

### torch support
def getDevice(force_cpu):
	try:
		if force_cpu:
			device= torch.device("cpu")
			print("force using CPU")
		elif torch.backends.mps.is_available():
			device = torch.device("mps")
			print("using Apple MX chipset")
		elif torch.cuda.is_available():
			device = torch.device("cuda")
			print("using Nvidia GPU")
		else:
			device = torch.device("cpu")
			print("using CPU")
		return device
	except:
		print("MPS is not supported for this version of PyTorch")
		if torch.cuda.is_available():
			device = torch.device("cuda")
			print("using Nvidia GPU")
		else:
			device = torch.device("cpu")
			print("using CPU")
		return device
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def setup_seed(seed):
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True

def getSafeSaveName(savepath,basename,extension=".pkl"):
	repeat_cnt =0
	safename = copy.copy(basename)
	while os.path.isfile(os.path.join(savepath,safename+extension)):
		repeat_cnt += 1
		safename = "{:}_{:}".format(basename,repeat_cnt)
	# return without extension
	return safename

# numpy implementation
# PyTorch implementation
def tch_gaussianLLR(xin,paras):
	sum_axis = tuple(torch.arange(2,len(xin.size())+1))
	mean = paras[0]
	logvar = paras[1]
	xp = xin.unsqueeze(dim=1)
	test_llr = -0.5 * (np.log(2*np.pi)+ logvar + (xp - mean).square() * (-logvar).exp()).sum(sum_axis)
	if (test_llr>0).any():
		print(test_llr)
		sys.exit()
	
	return  -0.5 * (np.log(2*np.pi)+ logvar + (xp - mean).square() * (-logvar).exp()).sum(sum_axis)

def tch_gaussianMeanLLR(xin,paras):
	return tch_gaussianLLR(xin,[paras[0],torch.zeros(paras[0].size(),device=xin.device).float()])

def tch_bernoulliLLR(xin,paras):
	sum_axis = tuple(torch.arange(2,len(xin.size())+1))
	logits = paras[0]
	xp = xin.unsqueeze(dim=1)
	# mps support 
	zsoftplus = (1.0 + logits.exp()).log()
	# cuda
	return (logits *xp - F.softplus(logits)).sum(sum_axis)

# for K cluster copies
def tch_K_gaussianLLR(xin,paras):
	# always assume dims: xin-->(nsamp,xdim...), paras-->(nclusters,xdim...)
	pmeans = paras[0]
	plvars  = paras[1]
	calc = -0.5 * (np.log(2*np.pi)+plvars+((-plvars).exp())*((xin.unsqueeze(dim=1)-pmeans).square()))
	sum_axis = tuple(torch.arange(2,len(calc.size())))
	return calc.sum(dim=sum_axis)
def tch_K_gaussianMeanLLR(xin,paras):
	#return tch_K_gaussianLLR(xin,[paras[0],torch.zeros(paras[0].size()).float()],start_dim)
	pmeans = paras[0]
	#print([item.size() for item in paras])
	calc = -0.5 * (np.log(2*np.pi)+(xin.unsqueeze(dim=1)-pmeans).square())
	sum_axis = tuple(torch.arange(2,len(calc.size())))
	return calc.sum(dim=sum_axis)
	#tmpsum = calc.sum(dim=sum_axis)
	# half space...
	#return tmpsum - tmpsum.amax(dim=1,keepdim=True)

def tch_K_bernoulliLLR(xin,paras):
	plogits = paras[0]
	xp = xin.float().unsqueeze(1)
	zsoftplus = (1.0 + plogits.exp()).log()
	#calc = xp * plogits - F.softplus(plogits)
	calc = xp * plogits - zsoftplus
	sum_axis = tuple(torch.arange(2,len(calc.size())))
	return calc.sum(dim=sum_axis)

def tch_K_expLLR(xin,paras):
	plog_lb = paras[0]
	calc = plog_lb - xin.unsqueeze(dim=1)* plog_lb.exp()
	sum_axis= tuple(torch.arange(2,len(calc.size())))
	return calc.sum(dim=sum_axis)

def tch_K_possionLLR(xin,paras):
	plog_lb = paras[0]
	calc = xin.unsqueeze(dim=1) * plog_lb  - plog_lb.exp()
	sum_axis = tuple(torch.arange(2,len(calc.size())))
	return calc.sum(dim=sum_axis)

# helper functions

def llr_normalization(weights,cluster_llr):
	# shape of llr (batch_size,nclustsers)
	return cluster_llr - logsumexp(cluster_llr,b=weights,axis=1,keepdims=True)

def tch_llr_normalization(weights,cluster_llr):
	return cluster_llr - (weights.log()+cluster_llr).logsumexp(dim=1,keepdims=True)

# KL divergence for exponential family 
def calc_GaussianKL(z_logits):
	z_mean = z_logits[0].flatten(start_dim=1)
	z_logvar = z_logits[1].flatten(start_dim=1)
	return -0.5*( (z_logvar + 1 -(z_mean**2) -z_logvar.exp()).sum(dim=1)).mean()

def calc_BernoulliKL(z_logits):
	z_llr = z_logits[0].flatten(start_dim=1)
	# manual softplus (MPS support walk-around)
	z_softplus = (1.0 + z_llr.exp()).log()
	return (np.log(2) + z_llr * z_llr.sigmoid() - z_softplus).sum(dim=1).mean()

# center updates

# one-shot update
# mini-batch updates
def tch_gaussianUpdate(old_paras,zmask,xin,old_cnts):
	# update means first
	xp = xin.unsqueeze(dim=1)
	batch_size = zmask.size()[0]
	nclusters = zmask.size()[1]
	zsum = zmask.sum(0)
	para_dim = [nclusters]+[1] * (len(xin.size())-1)
	zmask_res = zmask.float().reshape(tuple([batch_size]+para_dim))
	#para_shape = tuple([len(old_cnts)]+[1]*(len(xin.size())-1))
	old_means = old_paras[0]
	diff_mean = (xp * zmask_res).sum(dim=0)/zsum.reshape(tuple(para_dim))
	ratio_un = (old_cnts / (old_cnts + zsum)).unsqueeze(-1)
	new_means = ratio_un * old_means + (1.0 -ratio_un) * diff_mean
	# stability
	new_means = torch.where((zsum==0).reshape(tuple(para_dim)),old_means,new_means)
	
	old_vars = old_paras[1].exp()
	mean_diff_sq = (new_means - old_means).square()
	var_diff = ((xp - old_means)*zmask_res).square().sum(dim=0)/zsum.reshape(tuple(para_dim))
	ra_1 = (old_cnts-1)/(old_cnts+zsum-1)
	ra_1_cmpl = 1.0 - ra_1
	ra_neg = (zsum+old_cnts)/(zsum+old_cnts-1)
	ra_1 = ra_1.unsqueeze(-1)
	ra_1_cmpl = ra_1_cmpl.unsqueeze(-1)
	ra_neg = ra_neg.unsqueeze(-1)
	new_vars = ra_1 * old_vars + ra_1_cmpl * var_diff - ra_neg * mean_diff_sq
	new_vars = torch.where((zsum>1).reshape(tuple(para_dim)),new_vars,old_vars)
	return [new_means, new_vars.clip(min=1e-4).log()]

def tch_gaussianMeanUpdate(old_paras,zmask,xin,old_cnts):
	# update the means
	batch_size = zmask.size()[0]
	nclusters = zmask.size()[1]
	old_means = old_paras[0]
	xp = xin.unsqueeze(dim=1)
	zsum = zmask.sum(dim=0)
	para_dim = [nclusters] + [1] * (len(xin.size())-1)
	zmask_res = zmask.float().reshape(tuple([batch_size]+para_dim))
	mean_diff = (xp * zmask_res).sum(dim=0)/ zsum.reshape(tuple(para_dim))
	# stability, for zero elements
	ra_res = (old_cnts/(old_cnts + zsum)).unsqueeze(-1)
	new_means = old_means*ra_res + mean_diff * (1.0 - ra_res)
	new_means = torch.where((zsum==0).reshape(tuple(para_dim)),old_means,new_means)
	return [new_means]

def tch_bernoulliUpdate(old_paras,zmask,xin,old_cnts):
	batch_size = zmask.size()[0]
	nclusters = zmask.size()[1]
	old_logits = old_paras[0]
	old_means = old_logits.sigmoid()
	xp = xin.unsqueeze(1)
	zsum = zmask.sum(dim=0)
	para_dim = [nclusters] + [1] * (len(xin.size())-1)
	zmask_res = zmask.float().reshape(tuple([batch_size]+para_dim))
	mean_diff = (xp * zmask_res).sum(dim=0)/ zsum.reshape(tuple(para_dim))
	ratio_un = (old_cnts/ (old_cnts+zsum)).unsqueeze(-1)
	new_means = ratio_un* old_means + (1.0 - ratio_un) * mean_diff
	# stability
	new_means = torch.where((zsum==0).reshape(tuple(para_dim)),old_means,new_means)
	new_means = new_means.clip(min=1e-4,max=1-1e-4)
	return [(new_means/(1.0-new_means)).log()]

# initialization 
######## kest modules
def kest_init_bernoulli(xin,zargs,nclusters,cluster_idx):
	tmp_oh = F.one_hot(zargs,num_classes=nclusters).to(xin.device)
	new_cnt = tmp_oh.sum(dim=0)
	z_expand = tmp_oh.float().reshape(shape=tuple(list(tmp_oh.size())+[1]*(len(xin.size())-1)))
	para_shape = tuple([nclusters]+[1]*(len(xin.size())-1))
	xp = xin.float().unsqueeze(1)
	ber_p = (xp * z_expand).sum(dim=0) / new_cnt.float().reshape(shape=para_shape)
	# stability when some new_cnt ==0
	# random ber p used
	ber_p = torch.where((new_cnt==0).reshape(para_shape),torch.rand(size=ber_p.size(),device=xin.device).to(xin.device),ber_p)
	# avoid 1,0 probability
	ber_p = ber_p.clip(min=1e-4,max=1-1e-4)
	# natural parameters
	return [(ber_p/(1-ber_p)).log()]

def kest_init_gaussianMeans(xin,zargs,nclusters,cluster_idx):
	return [xin[cluster_idx]]

def kest_init_gaussian(xin,zargs,nclusters,cluster_idx):
	means = xin[cluster_idx]
	default_vars = torch.ones(means.size(),device=xin.device).float()
	tmp_oh = F.one_hot(zargs,num_classes=nclusters).to(xin.device)
	new_cnt = tmp_oh.sum(0)
	batch_size = tmp_oh.size()[0]
	para_dim = [nclusters] + [1]*(len(xin.size())-1)
	zsh = tmp_oh.float().reshape(tuple([batch_size]+para_dim))
	xp = xin.unsqueeze(1)
	new_vars = ((xp- means)*zsh).square().sum(0)/(new_cnt-1).reshape(tuple(para_dim))
	new_vars = torch.where((new_cnt>1).reshape(tuple(para_dim)),new_vars,default_vars)
	return [means, new_vars.clip(min=1e-4).log()]

def getSetDict(nview):
	set_dict = {"uniset":set(np.arange(nview)),"tuple_list":[]}
	for widx in range(0,int(np.floor(nview/2))):
		for item in itertools.combinations(set_dict['uniset'],widx+1):
			cmpl_tuple = tuple(set_dict['uniset']-set(item)) 
			if (item,cmpl_tuple) in set_dict['tuple_list'] or (cmpl_tuple,item) in set_dict['tuple_list']:
				pass
			else:
				set_dict['tuple_list'].append((item,cmpl_tuple)) # from tuple to set
	return set_dict


def ML_gaussianUpdate(old_paras,zmask,xin):
	old_means = old_paras[0]
	old_vars = old_paras[1].exp()
	zsum = zmask.sum(dim=0) #(nclusters,)
	nclusters = len(zsum)
	batch_size = zmask.size()[0]
	# zmask: (batch,nclusters)
	para_dim = [nclusters] + [1]*(len(xin.size())-1)
	zsh_res = zmask.float().reshape(tuple([batch_size]+para_dim)) # xin =1 
	xp = xin.unsqueeze(dim=1) # (bs,1,nclusters,xdim)
	new_means = (xp * zsh_res).sum(dim=0)/zsum.reshape(tuple(para_dim))
	# stability
	new_means = torch.where((zsum==0).reshape(tuple(para_dim)),old_means,new_means)
	# get new dim 
	new_vars = ((xp - new_means)*zsh_res).square().sum(dim=0)/(zsum-1).reshape(tuple(para_dim))
	new_vars = torch.where((zsum>1).reshape(tuple(para_dim)),new_vars,old_vars)
	return [new_means, new_vars.clip(min=1e-4).log()]

def ML_gaussianMeanUpdate(old_paras,zmask,xin):
	old_means = old_paras[0]
	zsum = zmask.sum(dim=0) #(nclusters,)
	nclusters = len(zsum)
	batch_size = zmask.size()[0]
	# zmask: (batch,nclusters)
	para_dim = [nclusters] + [1]*(len(xin.size())-1)
	zsh_res = zmask.float().reshape(tuple([batch_size]+para_dim)) # xin =1 
	xp = xin.unsqueeze(dim=1) # (bs,1,nclusters,xdim)
	new_means = (xp * zsh_res).sum(dim=0)/zsum.reshape(tuple(para_dim))
	# stability
	new_means = torch.where((zsum==0).reshape(tuple(para_dim)),old_means,new_means)
	return [new_means]

def ML_bernoulliUpdate(old_paras,zmask,xin):
	old_means = old_paras[0].sigmoid()
	zsum = zmask.sum(dim=0) #(nclusters,)
	nclusters = len(zsum)
	batch_size = zmask.size()[0]
	# zmask: (batch,nclusters)
	para_dim = [nclusters] + [1]*(len(xin.size())-1)
	zsh_res = zmask.float().reshape(tuple([batch_size]+para_dim)) # xin =1 
	xp = xin.unsqueeze(dim=1) # (bs,1,nclusters,xdim)
	new_means = (xp * zsh_res).sum(dim=0)/zsum.reshape(tuple(para_dim))
	new_means = torch.where((zsum==0).reshape(tuple(para_dim)),old_means,new_means)
	new_means = new_means.clip(min=1e-4,max=1-1e-4)
	return [(new_means/(1.0 - new_means)).log()]

# require testing
def tch_K_multivariateLLR(xin,paras,**kwargs):
	# assume xin, is already vectorized
	# natural parameters
	# cov^{-1}*mu
	# 0.5 * cov^{-1}
	eta_mu = paras[0] # dim (K,d)
	dd = eta_mu.size()[1]
	cov= paras[1] # dimension, (K,d,d)...
	# xin (N,d)
	xdiff = xin.unsqueeze(1) - eta_mu.unsqueeze(0) # (N,K,d)
	#print("inv")
	if "inv" in kwargs.keys():
		matinv = kwargs['inv']
	else:
		matinv = torch.linalg.inv(cov) # NOTE: this should be cached somewhere to save time...
	#print("det")
	if "det" in kwargs.keys():
		matdet = kwargs['det']
	else:
		matdet = torch.linalg.det(cov) # NOTE: this should be cached somewhere to save time...
	kernel2 = -0.5 * xdiff.unsqueeze(2) @ matinv.unsqueeze(0) @ xdiff.unsqueeze(3) # (N,K,1)
	kernel2 = kernel2.squeeze() # (N,K)
	# ignore constants?
	const1 = - 0.5 * dd * np.log(2 * torch.pi)
	kernel1 = -0.5 * matdet.log() # (K,)
	# dimension should be (N,K)...
	return const1 + kernel2 + kernel1.unsqueeze(0) # (N,K)

def kest_init_multivariate(xin,zargs,nclusters,cluster_idx):
	mu = xin[cluster_idx] # treated as means
	dd = xin.size()[1] # 
	cov = torch.eye(dd).unsqueeze(0).repeat_interleave(nclusters,dim=0).float()
	return [mu,cov]

def tch_multivariateUpdate(old_paras,zmask,xin,old_cnts):
	# batch_update...
	# NOTE: assume vectorized input and parameters
	batch_size = zmask.size()[0]
	nclusters = zmask.size()[1]
	old_means = old_paras[0] #(K,xdim)
	old_cov =old_paras[1]
	xp = xin.unsqueeze(dim=1) # (N,K,xdim)
	zsum = zmask.sum(0)
	para_dim = [nclusters] + [1] * (len(xin.size())-1)
	zmask_res = zmask.float().reshape(tuple([batch_size]+para_dim))
	mean_diff = (xp * zmask_res).sum(dim=0)/ zsum.reshape(tuple(para_dim))
	# stability, for zero elements
	ra_res = (old_cnts/(old_cnts + zsum)).unsqueeze(-1)
	new_means = old_means*ra_res + mean_diff * (1.0 - ra_res)
	new_means = torch.where((zsum==0).reshape(tuple(para_dim)),old_means,new_means)
	# proceed to update the covariance
	mean_diff = old_means - new_means
	mean_diff_ot = mean_diff.unsqueeze(2) @ mean_diff.unsqueeze(1) # (K,xd,xd)
	x_shift = (xp - old_means) #(N,K,xdim)
	est_cov = x_shift.unsqueeze(3) @ x_shift.unsqueeze(2) # (N,K,xd,xd)
	zvec_res= zmask.float().reshape(tuple([batch_size,nclusters,1,1]))
	para_cov = [nclusters,1,1]
	cov_diff = (est_cov * zvec_res).sum(0)/(zsum-1).reshape(tuple(para_cov))
	
	ratio1 = (old_cnts - 1 )/ (zsum + old_cnts -1)
	ratio1_cmpl = 1.0 - ratio1
	ratio_neg = (old_cnts + zsum)/(old_cnts + zsum -1)
	ra_1 = ratio1.unsqueeze(-1).unsqueeze(-1)
	ra_1_cmpl = ratio1_cmpl.unsqueeze(-1).unsqueeze(-1)
	ra_neg = ratio_neg.unsqueeze(-1).unsqueeze(-1)
	new_cov = old_cov * ra_1 + ra_1_cmpl * cov_diff - ra_neg * mean_diff_ot
	# stability
	new_cov = torch.where((zsum>1).reshape(tuple(para_cov)),new_cov,old_cov)
	return [new_means, new_cov]
def ML_multivariateUpdate(old_paras,zmask,xin):
	old_means = old_paras[0]
	old_cov = old_paras[1]
	zsum = zmask.sum(dim=0) #(nclusters,)
	nclusters = len(zsum)
	batch_size = zmask.size()[0]
	# zmask: (batch,nclusters)
	para_dim = [nclusters] + [1]*(len(xin.size())-1)
	zsh_res = zmask.float().reshape(tuple([batch_size]+para_dim)) # xin =1 
	xp = xin.unsqueeze(dim=1) # (bs,1,nclusters,xdim)
	new_means = (xp * zsh_res).sum(dim=0)/zsum.reshape(tuple(para_dim))
	# stability
	new_means = torch.where((zsum==0).reshape(tuple(para_dim)),old_means,new_means)
	# (N,K,xdim)
	# covariance estimation...
	# NOTE: assume xin vectorized,...
	est_cov = xp - new_means #(N,K,xdim)
	raw_cov = est_cov.unsqueeze(3) @ est_cov.unsqueeze(2) # (N,K,xdim,xdim)
	# masking , assume xin a 1-d vector
	zvec_res = zmask.float().reshape(tuple([batch_size,nclusters,1,1]))
	para_cov = [nclusters,1,1]
	# unbiased estimate
	new_cov = (raw_cov * zvec_res).sum(dim=0)/(zsum-1).reshape(tuple(para_cov))
	# stability
	new_cov = torch.where((zsum>1).reshape(tuple(para_cov)),new_cov,old_cov)
	return [new_means,new_cov]