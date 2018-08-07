import numpy as np
import json
import sys
import pandas as pd
import statsmodels.api as sm
# Given the QMC data, fit on-site energies to Si (Gamma point) orbitals

"""
Model:
E_tot = \sum_{orb} E_orb n_orb
n_orb = (nup_orb + ndn_orb) is the electron density in orbital orb
"""

ha2eV = 27.2114

def read_data(fnames):
  dfs = []
  for fname in fnames:
    with open(fname,'r') as f:
      data = json.load(f)
    dfs.append(extract(data))
  return pd.DataFrame(dfs)

def combine_dfs(df, spins=False):
  keys = df.keys()
  comb_df = {}
  for k in keys:
    arr = np.stack(df[k].values)
    if k.find('deriv')>=0:
      print('combine_df', k)
      tinds = np.arange(len(arr.shape))
      tinds[[0,1]] = [1,0]
      arr = arr.transpose(tinds)
    comb_df[k] = arr
  en = np.concatenate(([comb_df['en']], comb_df['enderiv']), axis=0)
  enerr = np.concatenate(([comb_df['enerr']], comb_df['enderiverr']), axis=0)
  dm_sp = np.concatenate(([comb_df['dm']], comb_df['dmderiv']), axis=0)
  dmerr_sp = np.concatenate(([comb_df['dmerr']], comb_df['dmderiverr']), axis=0)
  dm = dm_sp.sum(axis=2)
  dmerr = dmerr_sp.sum(axis=2)
  if spins:
    dm = np.concatenate((dm_sp, dm[:,:,np.newaxis]), axis=2)
    dmerr = np.concatenate((dmerr_sp, dmerr[:,:,np.newaxis]), axis=2)
  # dimensions: deriv, sample, (spin), (orb,orb)
  return en, enerr, dm, dmerr

def extract(data):
  deriv_dm = data['properties']['derivative_dm']
  nmo = len(deriv_dm['tbdm']['obdm']['up'])
  # Values
  en =    data['properties']['total_energy']['value'][0]
  enerr = data['properties']['total_energy']['error'][0]
  dm, dmerr = np.zeros((2,nmo,nmo)), np.zeros((2,nmo,nmo))
  for s, spin in enumerate(['up','down']):
    dm[s]    = np.array(deriv_dm['tbdm']['obdm'][spin])
    dmerr[s] = np.array(deriv_dm['tbdm']['obdm'][spin+'_err']) 
  
  # Derivatives
  dpwf =    np.array( deriv_dm['dpwf']['vals'] )
  dpwferr = np.array( deriv_dm['dpwf']['err'] )
  # energy
  dpen =    np.array( deriv_dm['dpenergy']['vals'] )
  dpenerr = np.array( deriv_dm['dpenergy']['err'] )
  enderiv = dpen - en*dpwf
  enderiverr = np.zeros(enderiv.shape) # TODO
  # density matrix
  nparam = len(dpwf)
  dpdm = np.zeros((nparam,2,nmo,nmo))
  dpdmerr = np.zeros((nparam,2,nmo,nmo))
  for p,dprdm in enumerate(deriv_dm['dprdm']):
    for s, spin in enumerate(['up','down']):
      dpdm[p][s] =    np.array(dprdm['tbdm']['obdm'][spin])
      dpdmerr[p][s] = np.array(dprdm['tbdm']['obdm'][spin+'_err'])
  dmderiv = dpdm - np.einsum('skl,p->pskl',dm,dpwf)
  dmderiverr = np.zeros(dmderiv.shape) # TODO

  output = dict(en=en, enerr=enerr, dm=dm, dmerr=dmerr, 
                enderiv=enderiv, enderiverr=enderiverr, 
                dmderiv=dmderiv, dmderiverr=dmderiverr,
                dpwf=dpwf, dpen=dpen, dpdm=dpdm,
                dpwferr=dpwferr, dpenerr=dpenerr, dpdmerr=dpdmerr)
  return output

## PLOT DESCRIPTOR DATA 
import matplotlib.pyplot as plt

def plot_obdm(dm, deriv=0, lowest_orb=1, **kwargs_):
  highest_orb = lowest_orb+dm.shape[-1]
  n = len(dm)
  v = np.amax(np.abs(dm))
  kwargs = dict( vmin=-v if deriv else 0, vmax=v, cmap='PRGn' if deriv else 'viridis')
  kwargs.update(**kwargs_)
  c=5
  if n<5:
    c=n
  r=int(np.ceil(n/c))
  fig, ax = plt.subplots(r,c, figsize=(2*c,3*r))
  axs = ax.ravel()
  for i in range(n):
    axs[i].imshow(dm[i], extent=[lowest_orb,highest_orb,highest_orb,lowest_orb], **kwargs)
  axs[0].set_ylabel(r'$n_\uparrow+n_\downarrow$ density')
  plt.tight_layout()

def plot_obdm_all(dm, deriv=0):
  n = len(dm)
  v = np.amax(np.abs(dm))
  kwargs = dict( vmin=-v if deriv else 0, vmax=v, cmap='PRGn' if deriv else 'plasma')
  fig, axs = plt.subplots(3,n, figsize=(12,4))
  for i in range(n):
    for s in [0,1,2]:
      axs[s,i].imshow(dm[i,s], **kwargs)
    axs[0,i].set_title('1.0 {0}'.format(i/5))
  for i in range(3):
    axs[i,0].set_ylabel(['up','down','sum'][i])
  plt.tight_layout()

def plot_occupations_all(dm, dmerr):
  n = len(dm)
  norbs = dm.shape[-1]
  fig, axs = plt.subplots(1,3, sharey=True, figsize=(8,3))
  #plt.axes([0.1,0.15,.75,.8])
  for i in range(dm.shape[-1]):
    for s in [0,1,2]:
      axs[s].errorbar(np.arange(n), dm[:,s,i,i], yerr=dmerr[:,s,i,i])
  for i in range(3):
    axs[i].set_ylabel(['up','down','sum'][i])
  axs[2].legend(np.arange(norbs), loc='center left', bbox_to_anchor=(1, 0.5))
  plt.tight_layout()

def plot_occupations(dm, dmerr, lowest_orb=1):
  n = len(dm)
  norbs = dm.shape[-1]
  plt.figure(figsize=(4,3))
  plt.axes([0.1,0.15,.75,.8])
  for i in range(dm.shape[-1]):
    plt.errorbar(np.arange(n), dm[:,i,i], yerr=dmerr[:,i,i])
  plt.xlabel('sample no.')
  plt.ylabel('occupation')
  plt.legend(np.arange(norbs)+lowest_orb, loc='center left', bbox_to_anchor=(1, 0.5))
  plt.tight_layout()

def plot_virt_sum(E, Eerr, dm, dmerr, nvalence):
  dens = np.einsum('ijj->i',dm[:,nvalence:,nvalence:])
  ddens = np.linalg.norm(np.einsum('ijj->ij',dmerr[:,nvalence:,nvalence:]), axis=1)
  n = len(dm)
  plt.figure(figsize=(4,3))
  plt.axes([0.1,0.15,.75,.8])
  plt.errorbar(np.arange(n), (E-np.mean(E))/np.std(E), yerr=Eerr)
  plt.errorbar(np.arange(n), (dens-np.mean(dens))/np.std(dens), yerr=ddens)
  plt.xlabel('sample no.')
  plt.ylabel('occupation')
  plt.legend(['energy fluc','CB occ fluc'], loc='center left', bbox_to_anchor=(1, 0.5))
  plt.tight_layout()
  
def plot_descriptors(dm, dmerr, lowest_orb=1):
  n = len(dm)
  norb = dm.shape[-1]
  
  plt.figure(figsize=(4,3))
  plt.axes([0.1,0.15,.75,.8])
  for i in range(n):
    plt.errorbar(np.arange(norb)+lowest_orb, np.diag(dm[i]), 
                  yerr=np.diag(dmerr[i]), marker='o', ls='')
  plt.xlabel('orbital')
  plt.ylabel('occupation')
  plt.legend(np.arange(n), loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')

def descriptor_corr_mat(en, dm, lowest_orb=1):
  n = len(dm)
  norb = dm.shape[-1]
  en[0] -= en[0].mean(axis=0)
  dens = np.einsum('...kk->...k', dm)
  dens[0] -= dens[0].mean(axis=0)
  en = en.reshape((-1,1))
  dens = dens.reshape((-1,norb))
  descriptors = np.block([en, dens]).T
  corrmat = np.corrcoef(descriptors)
  v = np.amax(np.abs(corrmat))
  plt.figure(figsize=(4,3))
  im = plt.imshow(corrmat, vmin=-v, vmax=v, aspect='equal', cmap='PRGn')
  plt.colorbar(im)
  plt.title('Descriptor correlation') 
 
  # fix labels
  locs, labels = plt.xticks()
  labels[1] = 'E'
  for i,l in enumerate(locs):
    if i<=1: continue
    labels[i] = 'orb%i'%(l-1+lowest_orb)
  plt.xticks(locs[1:-1], labels[1:-1], rotation=20)
  plt.yticks(locs[1:-1], labels[1:-1], rotation=0)
  plt.tight_layout()

def plot_energy(E, Eerr, deriv=0, titlestr=''):
  n=len(E)
  plt.figure(figsize=(4,3))
  plt.errorbar(np.arange(n), E, yerr=Eerr)
  plt.xlabel('sample no.')
  if deriv>0: plt.ylabel('Energy derivative (Ha)')
  else: plt.ylabel('Energy (Ha)')
  plt.title(titlestr)
  plt.tight_layout()

def make_descriptor_matrix(E, dm, nsamples=10, orbs=(-1,), orb_degen=None):
  """
  E: length nsamples*(nderivs+1) array. first nsamples are E, next nsamples are dE/dc_1, etc for all parameter derivatives.
  dm: nsamples*(nderivs+1) x norbs array. rows are as with E, first nsamples are occupations, then derivatives.
  nsamples: number of samples (since len(E) and len(dm) may also include paramater derivatives)
  orbs: list of orbitals to use in fit
  orb_degen: list of lists of degenerate orbitals to use in fit
  """
  print(E.shape, dm.shape)
  norb = dm.shape[-1]
  #dm = dm.transpose((1,0,2,3))
  dens = np.einsum('...kk->k...',dm).reshape((norb, -1))
  const = np.zeros(dens.shape[-1])
  const[:nsamples] = 1
  descriptors = [const]
  if orb_degen is None:
    for orb in orbs:
      descriptors.append(dens[orb])
  else:
    for os in orb_degen:
      descriptors.append(np.sum(dens[np.array(os)],axis=0))
      print(descriptors[-1].shape)
  D = np.stack(descriptors).T
  return D, E.reshape(-1)

def plot_fit(D, E, Eerr, nsamples=10 ):
  #p, res, rank, sing = np.linalg.lstsq(D,E)
  model = sm.OLS(E,D)
  result = model.fit()
  p = result.params
  perr = result.bse
  nparams = int(len(E)/(nsamples))-1
  n = nsamples
  #print('rank of descriptor matrix',rank)
  print('$R^2$',result.rsquared)
  #print('residuals', result.resid)
  print('AIC/BIC', result.aic, result.bic)
  print('model params',p)
  if nparams>0:
    plt.figure(figsize=(6,3))
    plt.subplot(121)
  else:
    plt.figure(figsize=(3,3))
  #plt.plot(D[:n,-1], np.dot(D,p)[:n])
  #plt.errorbar(D[:n,-1], E[:n], yerr=Eerr[:n], ls='', marker='o')
  #plt.xlabel('Occupation of CBM orbital')
  #plt.legend(['model','data'])
  Emin, Emax = np.amin(E[:n]), np.amax(E[:n])
  plt.plot((Emin,Emax),(Emin,Emax), ls='-')
  plt.errorbar(np.dot(D,p)[:n], E[:n], yerr=Eerr[:n], xerr=np.dot(D**2,perr**2)[:n]**0.5, 
                ls='', marker='o',markersize=3)
  plt.xlabel('Model estimate')
  plt.ylabel('Energy (Ha)')
  plt.xticks(rotation=10)
  
  if nparams>0:
    plt.subplot(122)
    #plt.plot(D[n:,1], np.dot(D,p)[n:])
    dEmin, dEmax = np.amin(E[n:]), np.amax(E[n:])
    plt.plot((dEmin,dEmax),(dEmin,dEmax), ls='-')
    plt.errorbar(np.dot(D,p)[n:], E[n:], yerr=Eerr[n:], xerr=np.dot(D**2,perr**2)[n:]**0.5
, ls='', marker='o', markersize=3)
    plt.xlabel('Model estimate')
    plt.ylabel('Energy derivative')
    #plt.legend(['model','data'])
    plt.subplot(121)
  
  #plt.title('Rank {0} \n Res. {1}'.format(rank,res))
  #plt.legend(['model','data'])
  plt.tight_layout()
 
def plot_fit_params(D, E, lowest_orb=1, use_eV=True, mfenergy=None, orb_degen=None, zero_orbs=None):
  #p, res, rank, sing = np.linalg.lstsq(D,E)
  model = sm.OLS(E,D)
  result = model.fit()
  p = result.params
  perr = result.bse
  if orb_degen is not None:
    assert len(p)==len(orb_degen)+1, "nparams is not equal to number of lists of degenerate orbitals"
    newp = [[p[0]]]
    newperr = [[p[0]]]
    if zero_orbs is not None:
      newp.append(np.zeros(len(zero_orbs)))
      newperr.append(np.zeros(len(zero_orbs)))
    newp.extend( [p[i+1]*np.ones(len(orb_degen[i])) for i in range(len(p)-1)])
    newperr.extend( [perr[i+1]*np.ones(len(orb_degen[i])) for i in range(len(p)-1)])
    p = np.concatenate(newp)
    perr = np.concatenate(newperr)
  #print('fit params (eV)\n',np.round(np.array((p,perr)).T*ha2eV,3))
  print('fit params (eV)\n',np.round(np.array(p)*ha2eV,3))
  print('fit errors (eV)\n',np.round(np.array(perr)*ha2eV,3))

  #for y in p[1:]:
  #  plt.axhline(y=y, ls='--')
  if use_eV:
    p=p*ha2eV
    perr=perr*ha2eV
  orbs = np.arange(len(p)-1)+lowest_orb
  plt.figure(figsize=(3.5,3))
  pline = plt.errorbar(orbs, p[1:], perr[1:])
  if mfenergy is not None:
    mfe = mfenergy[orbs-1]
    if use_eV:  
      mfe = mfe*ha2eV 
    mfe = mfe - mfe[2] + p[2]
    mfline, = plt.plot(orbs, mfe)
    plt.legend([pline,mfline],['model parameters','DFT energies'])
    print('DFT gap', np.amin(mfe[3:])-np.amax(mfe[:2]))
  print('Model gap', np.amin(p[4:])-np.amax(p[:3]))
  plt.ylabel('Energy (Ha)')
  if use_eV:
    plt.ylabel('Energy (eV)')
  plt.xlabel('orbital no.')
  plt.title('Model band energies')
  plt.tight_layout()

def print_en_list(E):
  ind = np.arange(len(E))+1
  ar = np.zeros((len(E),2))
  ar[:,0] = ind
  ar[:,1] = E
  print(ar)

def get_si_energies(chkfile):
  from pyscf import pbc, lib 
  from pyscf.pbc.dft import KRKS
  mol = pbc.gto.cell.loads(lib.chkfile.load(chkfile, 'mol'))
  mf = KRKS(mol)
  mf.__dict__.update(lib.chkfile.load(chkfile,'scf'))
  return mf.mo_energy

if __name__=='__main__':
  if len(sys.argv)>1:
    fnames = sys.argv[1:] 
    try:
      import re
      nums = [int( re.sub("[^0-9]", "", f)) for f in fnames]
      inds = np.argsort(nums)
      fnames = [fnames[i] for i in inds]
    except:
      print('No numbers in filenames; can\'t sort')
  else:
    print('give json file names of results')
  df = read_data(fnames)
  n=nsamples = len(fnames)

  qmcstr = fnames[0].split('.')[1] # vmc or dmc
  deriv = 0
  derivstr = '_deriv%i'%deriv if deriv else ''
  label = '_symmdeg'
  print(qmcstr+derivstr)

  en, enerr, dm, dmerr = combine_dfs(df, spins=False)
  
  print('trace', np.trace(dm[0], axis1=1, axis2=2))
  
  if True: ## Plot data 
    # Plot energy
    plot_energy(en[deriv], enerr[deriv], deriv=deriv, titlestr=qmcstr+derivstr)
    plt.savefig('{0}{1}{2}_energy.pdf'.format(qmcstr,derivstr,label), bbox_inches='tight')
    plt.show()

    # Plot occupations
    plot_virt_sum(en[deriv], enerr[deriv], dm[deriv], dmerr[deriv], nvalence=4)
    plt.show()
    
    plot_occupations(dm[deriv], dmerr[deriv], lowest_orb=13)
    plt.savefig('{0}{1}{2}_occupations.pdf'.format(qmcstr, derivstr,label), bbox_inches='tight')
    plot_descriptors(dm[deriv], dmerr[deriv], lowest_orb=13)
    plt.savefig('{0}{1}{2}_descriptors.pdf'.format(qmcstr, derivstr,label), bbox_inches='tight')
    plt.show()
    
    # Show OBDM
    plot_obdm(dm[deriv], deriv=deriv, lowest_orb=13, cmap='nipy_spectral')
    plt.savefig('{0}{1}{2}_obdm.pdf'.format(qmcstr,derivstr,label), bbox_inches='tight')
    plt.show()
 
    # Show covariance matrix 
    print(en.shape, dm.shape)
    descriptor_corr_mat(en[[deriv]], dm[[deriv]], lowest_orb=13)
    plt.savefig('{0}{1}{2}_corrmat.pdf'.format(qmcstr,derivstr,label), bbox_inches='tight')
    plt.show()

    descriptor_corr_mat(en, dm, lowest_orb=13)
    plt.savefig('{0}{1}{2}_corrmat.pdf'.format(qmcstr,'_allderivs'+0*derivstr,label), bbox_inches='tight')
    plt.show()
    quit()

  ## Fit model
  norbs = dm.shape[-1]
  orbs = np.arange(1,norbs-2) # which MOs to use for fitting model, indexed by which orbs were computed in QMC; 1 skips the first one (no change in the descriptor), norbs-2 skips the last one (no change in descriptor) and the second to last (swapping a DOF for the constant shift) 
  print(orbs)
  # some orbitals are supposed to be degenerate; let's include that
  orb_degen = [[4,5,6,7,8,9],[10,11,12]]
  zero_orbs = [1,2,3]

  D, E = make_descriptor_matrix(en, dm, nsamples=len(fnames), orbs=orbs, orb_degen=orb_degen)
  Eerr = enerr.T.reshape(-1)
 
  derivstr = '_allderivs'
  deriv=1
  if deriv==0:
    derivstr='' 
    D = D[:n]
    E = E[:n]
    Eerr = Eerr[:n]
  
  #derr = np.linalg.norm(np.stack(df['dmerr'].values)[:,:,orb,orb], axis=1)
  plot_fit(D, E, Eerr, nsamples=nsamples)
  plt.savefig('{0}{1}{2}_model.pdf'.format(qmcstr,derivstr,label), bbox_inches='tight')
  plt.show()

  dft_energy = get_si_energies('../conv_sipyscf/si_conv_k888/scf.py.chkfile')
  plot_fit_params(D, E, lowest_orb=14, mfenergy=dft_energy[0], orb_degen=orb_degen, zero_orbs=zero_orbs)
  plt.savefig('{0}{1}{2}_model_bands.pdf'.format(qmcstr,derivstr,label), bbox_inches='tight')
  plt.show()






