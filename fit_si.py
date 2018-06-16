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
  dm = [[],[]]
  dmerr = [[],[]]

  for fname in fnames:
    with open(fname,'r') as f:
      data = json.load(f)
    updm = np.array(data['properties']['derivative_dm']['tbdm']['obdm']['up'])
    dndm = np.array(data['properties']['derivative_dm']['tbdm']['obdm']['down'])
    dm[0].append(updm)
    dm[1].append(dndm)
    dmerr[0].append( np.array(data['properties']['derivative_dm']['tbdm']['obdm']['up_err']) )
    dmerr[1].append( np.array(data['properties']['derivative_dm']['tbdm']['obdm']['down_err']) )
    E.append( data['properties']['total_energy']['value'] )
    Eerr.append( data['properties']['total_energy']['error'] )

  dm = np.stack([np.stack(d) for d in dm])
  dmerr = np.stack([np.stack(d) for d in dmerr])
  return dm, dmerr, np.array(E), np.array(Eerr)

## PLOT DESCRIPTOR DATA 
import matplotlib.pyplot as plt

def plot_obdm(dm, cutoff=None, lower_cutoff=None):
  if cutoff is not None:
    dm = dm[:,:,:cutoff,:cutoff]
  else: 
    cutoff=dm.shape[2]
  if lower_cutoff is not None:
    dm = dm[:,:,lower_cutoff:,lower_cutoff:]
  else:
    lower_cutoff=0
  n = len(dm[0])
  print(dm.shape)
  fig, axs = plt.subplots(1,n, figsize=(10,3))
  for i in range(n):
    axs[i].imshow(dm[0][i]+dm[1][i], extent=[lower_cutoff,cutoff,cutoff,lower_cutoff])
    axs[i].set_title('1.0 {0}'.format(i*3/5))
  axs[0].set_ylabel(r'$n_\uparrow+n_\downarrow$ density')
  plt.tight_layout()
  plt.savefig('vmc_obdm.png')
  plt.show()

def plot_obdm_all(dm, cutoff=None):
  if cutoff is not None:
    dm = dm[:,:,:cutoff,:cutoff]
  n = len(dm[0])
  print(dm.shape)
  fig, axs = plt.subplots(3,n)
  for i in range(n):
    for s in [0,1]:
      axs[s,i].imshow(dm[s,i])
    axs[2,i].imshow(dm[0][i]+dm[1][i])
    axs[0,i].set_title('1.0 {0}'.format(i/5))
  for i in range(3):
    axs[i,0].set_ylabel(['up','down','sum'][i])
  plt.show()

def plot_occupations(dm, cutoff=None):
  if cutoff is not None:
    dm = dm[:,:,:cutoff,:cutoff]
  else:
    cutoff = dm.shape[2]
  n = len(dm[0])
  plt.figure(figsize=(7,4))
  plt.axes([0.1,0.15,.75,.8])
  plt.plot(np.arange(n)/5, np.einsum('ijj->ij',dm[0]))
  plt.xlabel('weight ratio c2/c1')
  plt.ylabel('occupation')
  plt.legend(np.arange(cutoff)+1, loc='center left', bbox_to_anchor=(1, 0.5))
  plt.savefig('vmc_occupations.png')
  plt.show()

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
  plt.title('Res. {0}'.format(res))
  plt.xlabel('Occupation of CBM orbital')
  plt.ylabel('Energy')
  plt.legend(['model','data'])
  plt.savefig('vmc_2orb_model.png')
  plt.show()
  

if __name__=='__main__':
  if len(sys.argv)>1:
    fnames = sys.argv[1:] 
  else:
    print('give json file names of results')
  dm, dmerr, E, Eerr = read_data(fnames)
  # Plot data 
  #plot_occupations(dm, cutoff=20)
  #plot_obdm(dm, cutoff=20, lower_cutoff=12)
  #quit()

  # Fit model
  D = make_descriptor_matrix(dm, orb=16)
  Dt = make_descriptor_matrix(dm, orb=16, with_t=True)
  dens_err = np.linalg.norm(dmerr[:,:,16,16], axis=0)
  plot_fit_occ(D, dens_err, E, Eerr)

















