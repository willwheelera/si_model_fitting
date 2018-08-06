import numpy as np
import json
import sys
import fit_si
import matplotlib.pyplot as plt

"""
PURPOSE
It seems like the formula for computing parameter derivatives for DMC, 
which samples the derivative from a different position than the mixed distribution,
is slightly different from the implementation, which samples it at the same position.

We haven't proven that the two are mathematically different;
if they are, it probably scales with the error in the trial function (squared?).

To see if there is a noticeable difference, we compute the parameter derivatives 
along a path and use numerical integration to compare to the energy and density values.
"""

def integrate_derivs(derivs, paramvals):
  means = (derivs[1:] + derivs[:-1])/2
  dx = np.diff(paramvals)
  integrand = means*dx
  deltas = np.cumsum(integrand)
  deltas = np.concatenate(([0],deltas))
  return deltas

def plot_estimate(paramvals, vals, errs, derivs, ax=None):
  deltas = integrate_derivs(derivs, paramvals)
  cum_errs = np.cumsum(errs**2)**.5
  v = ax.errorbar(paramvals, vals-vals[0], yerr=cum_errs, ls='', marker='o')
  d, = ax.plot(paramvals, deltas)
  print(d)
  ax.legend([v,d],['values','deriv estimate'])
  ax.set_xlabel('Parameter value')

def plot_est_diff(paramvals, vals, errs, derivs, ax=None):
  deltas = integrate_derivs(derivs, paramvals)
  cum_errs = np.cumsum(errs**2)**.5
  v = ax.errorbar(paramvals, vals-vals[0]-deltas, yerr=cum_errs, ls='', marker='o')
  d, = ax.plot(paramvals, np.zeros(np.shape(paramvals)))
  print(d)
  ax.legend([v],['difference'])
  ax.set_xlabel('Parameter value')

def make_plots(paramvals, df): 
  en, enerror, dm, dmerror = fit_si.combine_dfs(df)
  # first half are vals, second half are derivs (assuming one parameter)
  enval = en[0]
  enerr = enerror[0]
  dmval = dm[0]
  dmerr = dmerror[0]
  ender = 2*en[1]
  dmder = 2*dm[1]

  fig, axs = plt.subplots(2,4, figsize=(10,6))
  plot_estimate(paramvals, enval, enerr, ender, axs[0,0])
  plot_est_diff(paramvals, enval, enerr, ender, axs[1,0])
  axs[0,0].set_title('Energy')
  plot_estimate(paramvals, dmval[:,1,1], dmerr[:,1,1], dmder[:,1,1], axs[0,1])
  plot_est_diff(paramvals, dmval[:,1,1], dmerr[:,1,1], dmder[:,1,1], axs[1,1])
  axs[0,1].set_title('DM (16,16)')
  plot_estimate(paramvals, dmval[:,2,2], dmerr[:,2,2], dmder[:,2,2], axs[0,2])
  plot_est_diff(paramvals, dmval[:,2,2], dmerr[:,2,2], dmder[:,2,2], axs[1,2])
  axs[0,2].set_title('DM (17,17)')

  offdval = dmval[:,2,1] + dmval[:,1,2]
  offderr = np.sqrt(dmerr[:,2,1]**2 + dmerr[:,1,2]**2)
  offdder = dmder[:,2,1] + dmder[:,1,2]
  plot_estimate(paramvals, offdval, offderr, offdder, axs[0,3])
  plot_est_diff(paramvals, offdval, offderr, offdder, axs[1,3])
  axs[0,3].set_title('DM (17,16) + (16,17)')

  plt.tight_layout()
  plt.show()


if __name__=='__main__':
  if len(sys.argv)>1:
    fnames = sys.argv[1:]
  else:
    print('need json filenames'); quit()

  nsamples = len(fnames)
  paramvals = np.linspace(0,1,nsamples) # this is how the parameters were generated for this data
  df = fit_si.read_data(fnames)

  make_plots(paramvals, df)










