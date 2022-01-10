from getdist import plots, MCSamples
import argparse
import numpy as np

parser = argparse.ArgumentParser(description = 'Plot a foreground chain.')
parser.add_argument('chain_path', type = str)
args = parser.parse_args()

data = np.loadtxt(args.chain_path)
weights = data[:,0]
samples = data[:,1:]
param_names = [ 'a_tSZ', 'a_kSZ', 'a_p', 'beta_p', 'a_c', 'beta_c', 'a_s' ]
param_labels = [ r'A_\mathrm{tSZ}', r'A_\mathrm{kSZ}', r'A_p', r'\beta_{p,c}', r'A_c', r'\beta_c', r'A_s' ]
param_toplot = [ 'a_tSZ', 'a_kSZ', 'a_p', 'beta_p', 'a_c', 'a_s' ]

mcmc = MCSamples(samples = samples, weights = weights, names = param_names, labels = param_labels)
print("Plotting a total of {:d} samples.".format(int(np.nansum(weights))))

g = plots.get_subplot_plotter()
g.triangle_plot([ mcmc ], param_toplot, filled = True, line_args = [ {'lw' : 2, 'color' : 'dodgerblue'} ], legend_labels = [ 'cmbext' ])

g.subplots[0,0].set_xlim(3.1, 3.5)
g.subplots[1,1].set_xlim(0.0, 2.5)
g.subplots[2,2].set_xlim(6.6, 7.0)
g.subplots[3,3].set_xlim(2.1, 2.14)
g.subplots[4,4].set_xlim(4.9, 5.3)
g.subplots[5,5].set_xlim(3.05, 3.15)

g.export('plot.pdf')
