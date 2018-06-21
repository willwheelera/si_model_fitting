import numpy as np
import json
import sys
# Given the QMC data, fit on-site energies to Si (Gamma point) orbitals

"""
Model:
E_tot = \sum_{orb} E_orb n_orb
n_orb = (nup_orb + ndn_orb) is the electron density in orbital orb
"""

def read_data(fnames):
  E= []
  Eerr = []
  Ederiv= []
  Ederiverr = []
  dm = [[],[]]
  dmerr = [[],[]]
  dmderiv = [[],[]]
  dmderiverr = [[],[]]

  for fname in fnames:
    with open(fname,'r') as f:
      data = json.load(f)
    dm[0].append(   np.array(data['properties']['derivative_dm']['tbdm']['obdm']['up']))
    dm[1].append(   np.array(data['properties']['derivative_dm']['tbdm']['obdm']['down']))
    dmerr[0].append(np.array(data['properties']['derivative_dm']['tbdm']['obdm']['up_err']) )
    dmerr[1].append(np.array(data['properties']['derivative_dm']['tbdm']['obdm']['down_err']) )
    for dprdm in data['properties']['derivative_dm']['dprdm']:
      dmderiv[0].append(    np.array(dprdm['tbdm']['obdm']['up']))
      dmderiv[1].append(    np.array(dprdm['tbdm']['obdm']['down']))
      dmderiverr[0].append( np.array(dprdm['tbdm']['obdm']['up_err']) )
      dmderiverr[1].append( np.array(dprdm['tbdm']['obdm']['down_err']) )
    E.extend(    data['properties']['total_energy']['value'] )
    Eerr.extend( data['properties']['total_energy']['error'] )
    Ederiv.extend(    data['properties']['derivative_dm']['dpenergy']['vals'] )
    Ederiverr.extend( data['properties']['derivative_dm']['dpenergy']['err'] )

  dm = np.stack([np.stack(d) for d in dm])
  dmerr = np.stack([np.stack(d) for d in dmerr])
  dmderiv = np.stack([np.stack(d) for d in dmderiv])
  dmderiverr = np.stack([np.stack(d) for d in dmderiverr])
  return dm, dmerr, np.array(E), np.array(Eerr), dmderiv, dmderiverr, np.array(Ederiv), np.array(Ederiverr)

## PLOT DESCRIPTOR DATA 
import matplotlib.pyplot as plt

def plot_obdm(dm, cutoff=None, lower_cutoff=None):
  dm = dm.sum(axis=0)
  if cutoff is not None:
    dm = dm[:,:cutoff,:cutoff]
  else: 
    cutoff=dm.shape[-1]
  if lower_cutoff is not None:
    dm = dm[:,lower_cutoff:,lower_cutoff:]
  else:
    lower_cutoff=0
  n = len(dm)
  print(dm.shape)
  kwargs = dict(
    vmin = np.amin(dm),
    vmax = np.amax(dm))
  fig, axs = plt.subplots(1,n, figsize=(10,3))
  for i in range(n):
    axs[i].imshow(dm[i], extent=[lower_cutoff,cutoff,cutoff,lower_cutoff], **kwargs)
    axs[i].set_title('1.0 {0}'.format(i/5))
  axs[0].set_ylabel(r'$n_\uparrow+n_\downarrow$ density')
  plt.tight_layout()

def plot_obdm_all(dm, cutoff=None):
  if cutoff is not None:
    dm = dm[:,:,:cutoff,:cutoff]
  dm = np.concatenate((dm, [dm.sum(axis=0)]), axis=0)
  n = len(dm[0])
  print(dm.shape)
  kwargs = dict(
    vmin = np.amin(dm),
    vmax = np.amax(dm))
  fig, axs = plt.subplots(3,n, figsize=(12,4))
  for i in range(n):
    for s in [0,1,2]:
      axs[s,i].imshow(dm[s,i], **kwargs)
    axs[0,i].set_title('1.0 {0}'.format(i/5))
  for i in range(3):
    axs[i,0].set_ylabel(['up','down','sum'][i])
  plt.tight_layout()

def plot_occupations_all(dm, dmerr, cutoff=None):
  if cutoff is not None:
    dm = dm[:,:,:cutoff,:cutoff]
    dmerr = dmerr[:,:,:cutoff,:cutoff]
  else:
    cutoff = dm.shape[-1]
  n = len(dm[0])
  fig, axs = plt.subplots(1,3, sharey=True, figsize=(8,3))
  #plt.axes([0.1,0.15,.75,.8])
  for i in range(dm.shape[-1]):
    for s in [0,1]:
      axs[s].errorbar(np.arange(n)/5, dm[s,:,i,i], yerr=dmerr[s,:,i,i])
    axs[2].errorbar(np.arange(n)/5, dm[:,:,i,i].sum(axis=0), 
                  yerr=np.linalg.norm(dmerr[:,:,i,i],axis=0))
  for i in range(3):
    axs[i].set_ylabel(['up','down','sum'][i])
  axs[2].legend(np.arange(cutoff), loc='center left', bbox_to_anchor=(1, 0.5))
  plt.tight_layout()

def plot_occupations(dm, dmerr, cutoff=None):
  dm = dm.sum(axis=0)
  dmerr = np.linalg.norm(dmerr, axis=0)
  print(dm.shape, dmerr.shape)
  if cutoff is not None:
    dm = dm[:,:cutoff,:cutoff]
  else:
    cutoff = dm.shape[-1]
  n = len(dm)
  plt.figure(figsize=(7,4))
  plt.axes([0.1,0.15,.75,.8])
  for i in range(dm.shape[-1]):
    plt.errorbar(np.arange(n)/5, dm[:,i,i], yerr=dmerr[:,i,i])
  plt.xlabel('weight ratio c2/c1')
  plt.ylabel('occupation')
  plt.legend(np.arange(cutoff)+1, loc='center left', bbox_to_anchor=(1, 0.5))

def plot_energy(E, Eerr, titlestr=''):
  n=len(E)
  plt.errorbar(np.arange(n)/5, E, yerr=Eerr)
  plt.xlabel('weight ratio c2/c1')
  plt.ylabel('Energy (Ha)')
  plt.title(titlestr)

def make_descriptor_matrix(dm_spin, orb=-1, with_t=False):
  dm = np.sum(dm_spin,axis=0)
  const = np.ones(len(dm))
  descriptors = [const]
  descriptors.append(dm[:,orb,orb])
  if with_t:
    descriptors.append(dm[:,orb,orb-1]+dm[:,orb,orb-1])
  return np.stack(descriptors).T

def plot_fit_occ(D, dens_err, E, Eerr):
  p, res, rank, sing = np.linalg.lstsq(D,E)
  print('rank of descriptor matrix',rank)
  print('params',p)
  plt.plot(D[:,1], np.dot(D,p))
  plt.errorbar(D[:,1], E, yerr=Eerr, xerr=dens_err, ls='', marker='o')
  plt.title('$\epsilon_0={0}$, $\epsilon_1={1}$, Res. {2}'.format(*p,res))
  plt.xlabel('Occupation of CBM orbital')
  plt.ylabel('Energy')
  plt.legend(['model','data'])
  plt.tight_layout()
  
def plot_fit_deriv(D, dens_err, E, Eerr, nparams=0):
  p, res, rank, sing = np.linalg.lstsq(D,E)
  n = int(len(E)/(nparams+1))
  print('rank of descriptor matrix',rank)
  print('params',p)
  if nparams>0:
    plt.subplot(121)
  plt.plot(D[:n,1], np.dot(D,p)[:n])
  plt.errorbar(D[:n,1], E[:n], yerr=Eerr[:n], xerr=dens_err[:n], ls='', marker='o')
  plt.xlabel('Occupation of CBM orbital')
  plt.ylabel('Energy')
  
  if nparams>0:
    plt.subplot(122)
    plt.plot(D[n:,1], np.dot(D,p)[n:])
    plt.errorbar(D[n:,1], E[n:], yerr=Eerr[n:], xerr=dens_err[n:], ls='-', marker='o')
    plt.xlabel('CBM occupation derivative ')
    plt.ylabel('Energy derivative')
    plt.legend(['model','data'])
    plt.subplot(121)
  
  plt.title('$\epsilon_0={0}$ \n $\epsilon_1={1}$ \n Res. {2}'.format(*p,res))
  plt.legend(['model','data'])
  plt.tight_layout()
  

if __name__=='__main__':
  if len(sys.argv)>1:
    fnames = sys.argv[1:] 
  else:
    print('give json file names of results')
  dm, dmerr, E, Eerr, dmderiv, dmderiverr, Ederiv, Ederiverr = read_data(fnames)
  # Plot data 
  qmcstr = fnames[0].split('.')[1]
  print('qmcstr',qmcstr)
  deriv = False
  derivstr = '_deriv' if deriv else ''
  
  if deriv: plot_energy(Ederiv, Ederiverr, titlestr=qmcstr+derivstr)
  else: plot_energy(E, Eerr, titlestr=qmcstr)
  plt.savefig('{0}{1}_energy.png'.format(qmcstr,derivstr))
  plt.show()
  if deriv: plot_occupations_all(dmderiv, dmderiverr )
  else: plot_occupations_all(dm, dmerr )
  plt.savefig('{0}{1}_occupations.png'.format(qmcstr, derivstr))
  plt.show()
  plot_obdm_all(dmderiv if deriv else dm)
  plt.savefig('{0}{1}_obdm.png'.format(qmcstr,derivstr))
  plt.show()
  
  quit()

  # Fit model
  
  orb = 2
  D = make_descriptor_matrix(dm, orb=orb)
  Dderiv = make_descriptor_matrix(dmderiv, orb=orb)
  Dderiv[:,0] = 0
  dens_err = np.linalg.norm(dmerr[:,:,orb,orb], axis=0)
  dderiv_err = np.linalg.norm(dmderiverr[:,:,orb,orb], axis=0)
  
  D = np.concatenate([D, Dderiv], axis=0)
  derr = np.concatenate((dens_err, dderiv_err), axis=0)
  E = np.concatenate((E,Ederiv))
  Eerr = np.concatenate((Eerr,Ederiverr))
  plot_fit_deriv(D, derr, E, Eerr, nparams=1)
  plt.savefig('{0}_deriv_2orb_model.png'.format(qmcstr))
  
  plt.show()








