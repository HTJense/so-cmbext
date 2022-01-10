import os
import sys
import socmbext as sox
import argparse
import numpy as np
import camb
import copy
import tqdm
import sacc

parser = argparse.ArgumentParser(description = 'Extract the CMB spectra using SO-MFLike')
parser.add_argument('--packages_path', '-p', dest = 'packages_path', default = None, type = str, help = 'path to cobaya packages.')
parser.add_argument('--steps', '-s', dest = 'steps_end', default = 1000000, type = int, help = 'the maximum chain length.')
parser.add_argument('--output', '-o', dest = 'chain_filename', default = 'chain.dat', type = str, help = 'name for the chain file.')
parser.add_argument('--foreground', '-fg', dest = 'fg_chain_filename', default = 'foreground_chain.dat', type = str, help = 'name for the foreground chain file.')
parser.add_argument('--continue', dest = 'chain_overwrite', action = 'store_const', const = False, default = True, help = 'continue from last chain sample.')

args = parser.parse_args()

nuisance_params = {
	# Varying parameters
	"a_tSZ": 3.3,
	"a_kSZ": 1.7,
	"a_p": 6.7,
	"beta_p": 2.1,
	"a_c": 5.0,
	"beta_c": 2.1,
	"a_s": 3.1,
	
	# Fixed parameters
	"a_gtt": 0.0,
	"a_gte": 0.0,
	"a_gee": 0.0,
	"a_psee": 0.0,
	"a_pste": 0.0,
	"xi": 0.0,
	"T_d" : 9.60,
	"bandint_shift_93" : 0,
	"bandint_shift_145" : 0,
	"bandint_shift_225" : 0,
	"calT_93": 1,
	"calE_93": 1,
	"calT_145": 1,
	"calE_145": 1,
	"calT_225": 1,
	"calE_225": 1,
	"calG_all": 1,
	"alpha_93": 0,
	"alpha_145": 0,
	"alpha_225": 0,
}

steps_start = 0

if args.chain_overwrite:
	with open(args.chain_filename, 'w') as fp:
		fp.write('# MONTE CARLO MARKOV CHAIN DATA\n')
	
	with open(args.fg_chain_filename, 'w') as fp:
		fp.write('# FOREGROUND CHAIN DATA\n')
		fp.write('# weight tSZ kSZ Ap beta_p Ac beta_c As\n')
else:
	data = np.loadtxt(args.chain_filename, dtype = float)
	
	steps_start = data.shape[0]
	paramvec = data[-1, 1:8]
	nuisance_params['a_tSZ'] = paramvec[0]
	nuisance_params['a_kSZ'] = paramvec[1]
	nuisance_params['a_p'] = paramvec[2]
	nuisance_params['beta_p'] = paramvec[3]
	nuisance_params['a_c'] = paramvec[4]
	nuisance_params['beta_c'] = paramvec[5]
	nuisance_params['a_s'] = paramvec[6]

cmbext = sox.SOCMBExt({
	"packages_path" : args.packages_path,
	"input_file" : "data_sacc_00000.fits",
	"cov_Bbl_file" : "data_sacc_w_covar_and_Bbl.fits",
	"defaults" : {
		"polarizations" : [ "TT", "TE", "ET", "EE" ],
		"scales" : {
			"TT" : [30, 5000],
			"TE" : [30, 5000],
			"ET" : [30, 5000],
			"EE" : [30, 5000]
		},
		"symmetrize" : True
	},
	"data" : {
		"experiments" : {
			"LAT" : {
				"frequencies" : [ 93, 145, 225 ]
			}
		},
		"spectra" : [
			{
				"experiments" : ["LAT", "LAT"],
				"frequencies" : [93, 93],
				"polarizations" : ["TT", "TE", "EE"]
			},
			{
				"experiments" : ["LAT", "LAT"],
				"frequencies" : [93, 145],
			},
			{
				"experiments" : ["LAT", "LAT"],
				"frequencies" : [93, 225],
			},
			{
				"experiments" : ["LAT", "LAT"],
				"frequencies" : [145, 145],
				"polarizations" : ["TT", "TE", "EE"]
			},
			{
				"experiments" : ["LAT", "LAT"],
				"frequencies" : [145, 225],
			},
			{
				"experiments" : ["LAT", "LAT"],
				"frequencies" : [225, 225],
				"polarizations" : ["TT", "TE", "EE"]
			},
		]
	}
})

param_covs = np.loadtxt('covmat', dtype = float)

def calculate_prior(**params):
	priorvec = -(1e10) * np.ones((7,))
	
	if params['a_tSZ'] > 0.0 and params['a_tSZ'] < 10.0: priorvec[0] = 0.0
	if params['a_kSZ'] > 0.0 and params['a_kSZ'] < 10.0: priorvec[1] = 0.0
	if params['a_p'] > 0.0 and params['a_p'] < 10.0: priorvec[2] = 0.0
	if params['beta_p'] > 1.8 and params['beta_p'] < 2.2: priorvec[3] = 0.0
	if params['a_c'] > 0.0 and params['a_c'] < 50.0: priorvec[4] = 0.0
	if params['beta_c'] > 1.8 and params['beta_c'] < 2.2: priorvec[5] = 0.0
	if params['a_s'] > 0.0 and params['a_s'] < 10.0: priorvec[6] = 0.0
	
	return np.nansum(priorvec)

def takestep(**params):
	gn = np.random.normal(size = (7,))
	step = param_covs @ gn
	
	newparvec = copy.deepcopy(params)
	
	newparvec['a_tSZ'] = params['a_tSZ'] + step[0]
	newparvec['a_kSZ'] = params['a_kSZ'] + step[1]
	newparvec['a_p'] = params['a_p'] + step[2]
	newparvec['beta_p'] = params['beta_p'] + step[3]
	newparvec['a_c'] = params['a_c'] + step[4]
	newparvec['beta_c'] = newparvec['beta_p']
	newparvec['a_s'] = params['a_s'] + step[6]
	
	return newparvec

def vectorize(**params):
	vector = np.zeros((7,))
	
	vector[0] = params['a_tSZ']
	vector[1] = params['a_kSZ']
	vector[2] = params['a_p']
	vector[3] = params['beta_p']
	vector[4] = params['a_c']
	vector[5] = params['beta_c']
	vector[6] = params['a_s']
	
	return vector

oldlike = 0.0
newlike = 0.0

nuisance_params = copy.deepcopy(takestep(**nuisance_params))

like = [ ]
weight = 0

fp_chain = open(args.chain_filename, 'a')
fp_chain_fg = open(args.fg_chain_filename, 'a')

for step in tqdm.tqdm(range(steps_start, args.steps_end)):
	spec = cmbext.extract_cmb_spectra(**nuisance_params)
	
	oldlike = cmbext.loglike(spec, **nuisance_params)
	oldprior = calculate_prior(**nuisance_params)
	oldpost = oldlike + oldprior
	
	newparams = takestep(**nuisance_params)
	
	newlike = cmbext.loglike(spec, **newparams)
	newprior = calculate_prior(**newparams)
	newpost = newlike + newprior
	
	weight += 1
	
	if newpost > oldpost or np.exp(newpost - oldpost) > np.random.random():
		oldpost = newpost
		
		fp_chain_fg.write('{0:5d}\t'.format(weight))
		write_data = [ *vectorize(**nuisance_params) ]
		write_line = '\t'.join([ '{:.5e}'.format(x) for x in write_data ]) + '\n'
		fp_chain_fg.write(write_line)
		fp_chain_fg.flush()
		
		nuisance_params = copy.deepcopy(newparams)
		
		weight = 0
	
	like.append(oldpost)
	
	fp_chain.write('{0:5d}\t'.format(step))
	write_data = [ -2.0 * oldpost, *vectorize(**nuisance_params), *spec['tt'], *spec['te'], *spec['ee'] ]
	write_line = '\t'.join([ '{:.5e}'.format(x) for x in write_data ]) + '\n'
	fp_chain.write(write_line)
	fp_chain.flush()
	
	del newparams

fp_chain_fg.write('{0:5d}\t'.format(weight))
write_data = [ *vectorize(**nuisance_params) ]
write_line = '\t'.join([ '{:.5e}'.format(x) for x in write_data ]) + '\n'
fp_chain_fg.write(write_line)
fp_chain_fg.flush()

fp_chain.close()
fp_chain_fg.close()

# Write all data to a sacc file.
# It's still a bit hard-coded, should be optimized in the future in case people want to run multi-experiment extraction codes.
raw_data = np.loadtxt(args.chain_filename)
s = sacc.Sacc()

for exp in cmbext.data['experiments'].keys():
	s.add_tracer("Misc", "{}_cmb_s0".format(exp))
	s.add_tracer("Misc", "{}_cmb_s2".format(exp))
	
	freqs = cmbext.data['experiments'][exp]['frequencies']
	f = freqs[ len(freqs) // 2 ]
	i0 = 0
	data = raw_data[:,9:]
	
	# Put the tt spectra in the file.
	for spec in cmbext.spec_meta:
		if spec['pol'] == 'tt' and spec['nu1'] == f and spec['nu2'] == f:
			win = spec['bpw']
			ws = win.weight.shape[-1]
			
			ells = spec['leff']
			c_ells = np.nanmean(data[:,i0:i0+ws], axis = 0)
			i0 += ws
			
			for ell, c_ell in zip(ells, c_ells):
				s.add_data_point("cl_00", ("{}_cmb_s0".format(exp), "{}_cmb_s0".format(exp)), c_ell, ell = ell, window = win)
			
			break
	
	# Put the te spectra in the file.
	for spec in cmbext.spec_meta:
		if spec['pol'] == 'te' and spec['nu1'] == f and spec['nu2'] == f:
			win = spec['bpw']
			ws = win.weight.shape[-1]
			
			ells = spec['leff']
			c_ells = np.nanmean(data[:,i0:i0+ws], axis = 0)
			i0 += ws
			
			for ell, c_ell in zip(ells, c_ells):
				s.add_data_point("cl_0e", ("{}_cmb_s0".format(exp), "{}_cmb_s2".format(exp)), c_ell, ell = ell, window = win)
			
			break
	
	# Put the ee spectra in the file.
	for spec in cmbext.spec_meta:
		if spec['pol'] == 'ee' and spec['nu1'] == f and spec['nu2'] == f:
			win = spec['bpw']
			ws = win.weight.shape[-1]
			
			ells = spec['leff']
			c_ells = np.nanmean(data[:,i0:i0+ws], axis = 0)
			i0 += ws
			
			for ell, c_ell in zip(ells, c_ells):
				s.add_data_point("cl_ee", ("{}_cmb_s2".format(exp), "{}_cmb_s2".format(exp)), c_ell, ell = ell, window = win)
			
			break
	
	cov = np.cov(data[:,:].T)
	s.add_covariance(cov)
	s.save_fits('sacc_cmbext.fits', overwrite = True)
