import numpy as np
import json
import sys
import pandas as pd
import statsmodels.api as sm
import time
# Given the QMC data, fit on-site energies to Si (Gamma point) orbitals

"""
Model:
E_tot = \sum_{orb} E_orb n_orb
n_orb = (nup_orb + ndn_orb) is the electron density in orbital orb
"""

ha2eV = 27.2114

def read_gosling_data(fnames):
  dfs = []
  for fname in fnames:
    with open(fname,'r') as f:
      data = json.load(f)
    dfs.append(extract_gosling(data))
  return pd.DataFrame(dfs)

def read_bootstrap(fnames, ext='hdf'):
  if ext is None:
    ext = fnames[0].split('.')[-1]
  if ext=='hdf':
    df = pd.concat([pd.read_hdf(fname) for fname in fnames], axis=1)
  elif ext=='json':
    df = pd.concat([pd.read_json(fname) for fname in fnames], axis=1)
  elif ext=='pickle':
    df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=1)
  return df
  
def read_bootstrap_json(fnames):
  df = pd.concat([pd.read_json(fname) for fname in fnames], axis=1)
  return df
  
def extract_bootstrap_df(bootdf, diag_only=True): # from table
  start = time.time()
  print('extract_bootstrap_df:','starting extract timer', time.time()-start)
  index = bootdf.loc['val'].index
  nfiles = len(bootdf.keys())
  #nstates = index.str.contains('normalization').sum()
  #states = [n.split('_')[1] for n in index[index.str.contains('normalization')]]
  print(index)
  nparams = index.str.contains('dpenergy').sum()

  newlabels = index[index.str.match('^energy|^.bdm')].values
  if diag_only:
    def isdiag(dmlabel):
      words = dmlabel.split('_')
      return words[0]!='obdm' or words[-1]==words[-2]
    newlabels = [s for s in newlabels if isdiag(s)]
  print('extract_bootstrap_df:','descriptors', len(newlabels))
  obdm_strs = [s for s in newlabels if s.startswith('obdm')]
  tbdm_strs = [s for s in newlabels if s.startswith('tbdm')]

  print('extract_bootstrap_df:','defined terms', time.time()-start)

  dfs={}
  for val in ['val','err']:
    df = bootdf.loc[val]  
    newd = {s:[] for s in newlabels}
    #newd['energy']  = df.filter(like='energy',axis=0).values.ravel()
    for f in df.columns:
      newd['energy'].append( df[f]['energy'] )
    for p in np.arange(nparams):
      for f in df.columns:
        newd['energy'].append( df[f]['dpenergy_{0}'.format(p)] )
    print('extract_bootstrap_df:',val)
    print('extract_bootstrap_df:','collected energy', len(newd['energy']), time.time()-start)
    for i,s in enumerate(obdm_strs):
      words = s.split('_')
      restr = '{0}_{1}.*_{2}_{3}$'.format(*words)
      plabel = 'dp{0}_{1}_{{0}}_{2}_{3}'.format(*words)
      tdf = df.filter(regex=restr,axis=0) 
      for f in tdf.columns:
        newd[s].append( tdf[f][s] )
      for p in np.arange(nparams):
        for f in tdf.columns:
          newd[s].append( tdf[f][plabel.format(p)] )
      df.drop(tdf.index.values, axis=0, inplace=True)
      # TODO copy for tbdm  
      #if i%nstates>=0: print('collecting', s, val, len(newd[s]), time.time()-start)
    newd['param'] = []
    newd['filename'] = []
    for p in range(nparams+1):
      for f in df.columns:
        newd['param'].append(p)
        newd['filename'].append(f)
    #newd['param'] = list(np.repeat(range(nparams+1),nfiles))
    #print('params', len(newd['param']))
    #newd['filename'] = list(np.tile(bootdf.keys().values,nparams+1))
    #print('filenames', len(newd['filename']))
    print('extract_bootstrap_df:','Converting to DataFrame...', time.time()-start)
    dfs[val] = pd.DataFrame(newd)#.set_index(['param','filename'])
  return dfs #pd.concat([dfs['val'],dfs['err']], keys=dfs.keys(), axis=0)

#def extract_bootstrap_hdf(df): 
#  nfiles = len(set(df['filename']))
#  nparams = df['deriv'].str.contains('dpenergy').sum()
#
#  vals = np.zeros((nparams+1, nfiles))
#  errs = np.zeros((nparams+1, nfiles))
#  vals[0] = df[df['deriv']=='energy']['value']
#  errs[0] = df[df['deriv']=='energy']['err']
#  
#  vals[1:] = df[df['deriv'].str.contains('dpenergy')]['value']
#  errs[1:] = df[df['deriv'].str.contains('dpenergy')]['err']
#  d['energyval'] = vals.copy()
#  d['energyerr'] = errs.copy()
#
#  for term in df['deriv']:
#    if term[1:4]=='bdm':
#      words = term.split('_') # 0: obdm, 1: up, 2: state_i, 3: state_j
#  en = df[df['deriv']=='energy']['value']
#  enerr = df[df['deriv']=='energy']['err']
#  ender = df[df['deriv'].str.contains('dpenergy')]['value']
#  endererr = df[df['deriv'].str.contains('dpenergy')]['err']

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

def extract_gosling(data):
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

def extrap_mixed_estimator(vmcdf, dmcdf):
  print('Not implemented yet')


## PLOT DESCRIPTOR DATA 
import matplotlib.pyplot as plt

def my_imshow(M, ax=None):
  mval = np.amax(np.abs(M))
  im = ax.imshow(M.real, vmax=mval, vmin=-mval, cmap='PRGn')
  plt.colorbar(im,fraction=0.046, pad=0.04)
  #plt.colorbar(im, ax=ax)

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

def old_plot_occupations(dm, dmerr, orbs=[]):
  n = len(dm)
  norbs = dm.shape[-1]
  assert len(orbs)==norbs, "len(orbs)={0}, norbs={1}".format(len(orbs),norbs)
  plt.figure(figsize=(4,3))
  plt.axes([0.1,0.15,.75,.8])
  for i in range(dm.shape[-1]):
    plt.errorbar(np.arange(n), dm[:,i,i], yerr=dmerr[:,i,i])
  plt.xlabel('sample no.')
  plt.ylabel('occupation')
  plt.legend(orbs, loc='center left', bbox_to_anchor=(1, 0.5))
  plt.tight_layout()

def plot_occupations(df):
  cols = df['val'].columns
  obdms = cols[cols.str.match('^obdm')]
  p0 = df['val']['param']==0
  pn0 = df['val']['param']!=0

  for p, par in enumerate((p0,pn0)):
    vals = df['val'][par][obdms]
    errs = df['err'][par][obdms]
    plt.subplot(2,1,p+1)
    for i in range(vals.shape[0]):
      plt.errorbar(np.arange(len(obdms))+i*1e-4, vals.values[i], yerr=errs.values[i])
      plt.xticks(np.arange(len(obdms)), obdms, rotation=30)

  #vals = df['val'][pn0][obdms]
  #errs = df['err'][pn0][obdms]
  #plt.subplot(212)
  #for i in range(vals.shape[0]):
  #  plt.errorbar(np.arange(len(obdms))+i*1e-4, vals.values[i], yerr=errs.values[i])
  #  plt.xticks(range(len(obdms)), obdms, rotation=30)
  
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

def df_corr_mat(df, param=None):
  val = df['val'].copy()
  err = df['err'].copy()
  cols = val.columns[val.columns.str.match('energy|obdm')]
  p0inds = val['param']==0
  meanval = val.loc[p0inds,'energy'].values.mean()
  nsamples = p0inds.sum()
  if param is not None:
    if param=='deriv' or param=='derivs':
      val = val[val['param']!=0]
      err = err[err['param']!=0]
    else:
      val = val[val['param']==param]
      err = err[err['param']==param]
  else:
    val.loc[p0inds,'energy'] -= meanval
  corrs = { 'val': val[cols].corr(),
            'err': err[cols].corr() }
  plt.figure(figsize=(8,3))
  for i,v in enumerate(('val','err')):
    my_imshow(corrs[v].values, plt.subplot(1,2,i+1))
    plt.title(v+str(param))
  
    # fix labels
    locs, labels = plt.xticks()
    labels[1:-1] = cols.values
    print('corr plot: locs', locs, 'cols', len(cols))
    locs = locs # don't know why the coords are offset this way
    #for j,l in enumerate(locs):
    #  if j<=1: continue
    #  labels[j] = cols.values[int(l/2)-1]
    #plt.xticks(locs[1:-1], labels[1:-1], rotation=20)
    plt.yticks(locs[1:-1], labels[1:-1], rotation=0)
    plt.xticks([], [])
    #plt.yticks(locs, labels, rotation=0)
  plt.tight_layout()
  
  if param is None:
    val.loc[p0inds,'energy'] += meanval
    

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
  dens = np.einsum('...kk->k...',dm).reshape((norb, -1)) # has shape norbs x nsamples*(nderivs+1)
  const = np.zeros(dens.shape[-1])
  const[:nsamples] = 1
  descriptors = [const]
  if orb_degen is None:
    for orb in orbs:
      descriptors.append(dens[orb])
  else:
    for os in orb_degen: # os is one set of degenerate orbitals
      descriptors.append(np.sum(dens[np.array(os)],axis=0)) # sum over the degen orbs
      print(descriptors[-1].shape)
  D = np.stack(descriptors).T
  return D, E.reshape(-1)

def combine_spin(df, inplace=True):
  """
  df: output of extract_bootstrap_df, rows are samples*params, cols are descriptors, also has columns filename and param for identification. Rows are multiindex, with 'val' and 'err' denoting the values for each quantity and their bootstrap errors.
  """
  keys = df.columns[df.columns.str.contains('bdm_up')]
  downkeys = [k.replace('up','down') for k in keys]
  df[keys] += df[downkeys].values
  return df.drop(columns=downkeys, inplace=inplace)
    
def make_descriptor_matrix_df(df, orbs_to_fit=None, return_Derr=False):
  """
  df: output of extract_bootstrap_df, rows are samples*params, cols are descriptors, also has columns filename and param for identification. Rows are multiindex, with 'val' and 'err' denoting the values for each quantity and their bootstrap errors.
  orbs_to_fit: list of strings, columns of df that correspond to model params. one energy param should be left out (implicitly set to zero) to account for the constant shift
  """
  assert orbs_to_fit is not None, 'Need to enter orbs_to_fit, for which states to use; leave out an orbital to fix zero energy'
  valdf = df['val']
  nmodelparams = len(orbs_to_fit)+1 # add constant term
  nsamples = int((valdf['param']==0).sum()) # param 0 is without derivative
  const = np.zeros((len(valdf),1))
  const[:,0] = (valdf['param']==0)*1
  print('where param==0', np.nonzero(const[:,0])[0])
  descriptors = valdf[orbs_to_fit].values
  if len(descriptors.shape)==1:
    descriptors = descriptors[:,np.newaxis]
  D = np.concatenate([const,descriptors], axis=1)
  E = valdf['energy']
  Eerr = df['err']['energy']
  if return_Derr:
    desc_err = df['err'][orbs_to_fit].values
    Derr = np.concatenate([const*0,desc_err], axis=1)
    return D, E, Eerr, Derr
  else:
    return D, E, Eerr

def fit_model(D, E, Eerr, weights=True):
  n = int(D[:,0].sum())
  p0inds = np.nonzero(D[:,0])[0]
  Emean = np.mean(E[p0inds])
  E.loc[p0inds] -= Emean
  if weights: model = sm.WLS(E, D, weights=Eerr**(-2)) # weights arg is 1/var
  else: model = sm.OLS(E, D ) 
  result = model.fit()
  p = result.params
  p[0] += Emean
  E.loc[p0inds] += Emean
  perr = result.bse
  #print('rank of descriptor matrix',rank)
  print('tvalues\n', result.tvalues)
  print('$R^2$',result.rsquared)
  #print('residuals', result.resid)
  print('AIC/BIC', result.aic, result.bic)
  return p, perr
  
def plot_fit(D, E, Eerr, Derr=None, weights=True, noderivs=False ):
  n = int(D[:,0].sum())
  p0inds = np.nonzero(D[:,0])[0]
  if noderivs:
    D=D[p0inds]
    E=E[p0inds]
    Eerr=Eerr[p0inds]
    if Derr is not None:
      Derr=Derr[p0inds]
  nparams = int(len(E)/n)-1
  print('nsamples',n, 'nparams', nparams, 'noderivs?',noderivs)

  p, perr = fit_model(D,E,Eerr,weights)
  print('model params (eV)\n', pd.DataFrame({'p':p,'perr':perr})*ha2eV)
  if nparams>0:
    plt.figure(figsize=(6,3))
    ax1 = plt.subplot(121)
  else:
    plt.figure(figsize=(3,3))
  #plt.plot(D[:n,-1], np.dot(D,p)[:n])
  #plt.errorbar(D[:n,-1], E[:n], yerr=Eerr[:n], ls='', marker='o')
  #plt.xlabel('Occupation of CBM orbital')
  #plt.legend(['model','data'])
  Emin, Emax = np.amin(E[:n]), np.amax(E[:n])
  plt.plot((Emin,Emax),(Emin,Emax), ls='-')
  plt.errorbar(np.dot(D,p)[:n], E[:n], yerr=Eerr[:n], 
                xerr=np.dot(D**2,perr**2)[:n]**0.5, 
                ls='', marker='', markersize=0)
  plt.scatter(np.dot(D,p)[:n], E[:n])#, c=np.arange(n))
  plt.xlabel('Model estimate')
  plt.ylabel('Energy (Ha)')
  plt.xticks(rotation=10)
  
  if nparams>0:
    ax2 = plt.subplot(122)
    #plt.plot(D[n:,1], np.dot(D,p)[n:])
    dEmin, dEmax = np.amin(E[n:]), np.amax(E[n:])
    plt.plot((dEmin,dEmax),(dEmin,dEmax), ls='-')
    plt.errorbar(np.dot(D,p)[n:], E[n:], yerr=Eerr[n:], 
                  xerr=np.dot(D**2,perr**2)[n:]**0.5,
                  ls='', marker='', markersize=0)
    dots = plt.scatter(np.dot(D,p)[n:], E[n:], marker='o', c=np.arange(len(E)-n)%n) 
    cbar = plt.colorbar(dots)
    plt.xlabel('Model estimate')
    plt.ylabel('Energy derivative')
    #plt.legend(['model','data'])
    plt.sca(ax1)
  
  #plt.title('Rank {0} \n Res. {1}'.format(rank,res))
  #plt.legend(['model','data'])
  plt.tight_layout()
 
def plot_fit_params(D, E, lowest_orb=1, use_eV=True, mfenergy=None, orb_degen=None, zero_orbs=None):
  #p, res, rank, sing = np.linalg.lstsq(D,E)
  p, perr = fit_model(D,E,None,weights=False)
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

def plot_en_dens(D, E, Eerr, Derr=None, CBorbs=[], use_eV=True, weights=True, title=''): 
  if use_eV:
    E = ha2eV*E
    Eerr = ha2eV*Eerr
  D = np.stack((D[:,0],D[:,CBorbs].sum(axis=1))).T
  p, perr = fit_model(D,E,Eerr,weights=weights)
  #model = sm.OLS(E,D)
  #result = model.fit()
  #p = result.params
  #perr = result.bse
  n = nsamples = int(D[:,0].sum())
  nparams = int(len(E)/nsamples)-1

  if nparams>0:
    fig, axs = plt.subplots(1,2, figsize=(5,3))
    plt.sca(axs[0])
    #plt.subplot(121)

  if Derr is not None:
    plt.errorbar(D[:nsamples,1], E[:nsamples], yerr=Eerr[:nsamples], xerr=Derr[:nsamples,1], ls='', marker='o',ecolor='orange')
  else:
    plt.errorbar(D[:nsamples,1], E[:nsamples], yerr=Eerr[:nsamples], ls='', marker='o',ecolor='orange')
  dlims = np.array([np.amin(D[:nsamples,1]), np.amax(D[:nsamples,1])])
  plt.plot(dlims, p[0] + p[1]*dlims)
  err = (perr[0]**2 + dlims**2*perr[1]**2)**.5
  plt.plot(dlims, p[0]+p[1]*dlims+err, ls=':')
  plt.plot(dlims, p[0]+p[1]*dlims-err, ls=':')
  plt.xlabel('CB occupation')
  plt.ylabel('Energy (%s)'%('eV' if use_eV else 'Ha'))
  plt.title('{2} Gap: {0:2.2f} ({1:2.2f})'.format(p[1], perr[1],title))

  if nparams>0:
    #plt.subplot(122)
    plt.sca(axs[1])
    if Derr is not None:
      plt.errorbar(D[nsamples:,1], E[nsamples:], yerr=Eerr[nsamples:], xerr=Derr[nsamples:,1], ls='', marker='o',ecolor='orange')
    else:
      plt.errorbar(D[nsamples:,1], E[nsamples:], yerr=Eerr[nsamples:], ls='', marker='o',ecolor='orange')
    dlims = np.array([np.amin(D[nsamples:,1]), np.amax(D[nsamples:,1])])
    plt.plot(dlims, p[1]*dlims)
    plt.plot(dlims, (p[1]+perr[1])*dlims, ls=':')
    plt.plot(dlims, (p[1]-perr[1])*dlims, ls=':')
    plt.xlabel('CB occupation derivs')
    plt.ylabel('Energy derivs (%s)'%('eV' if use_eV else 'Ha'))
  plt.tight_layout() 

def plot_vals_vs_derivs(dfs):
  val = dfs['val']
  err = dfs['err']
  energyinds = val.columns[val.columns.str.match('^energy')]
  obdminds = val.columns[val.columns.str.match('^obdm_up_17_17')]
  p0inds = val['param']==0
  p1inds = val['param']!=0
  print(val.columns)
  
  neg_inds = val['obdm_up_17_17']<0
  val.loc[neg_inds, 'obdm_up_17_17'] = (1-(val.loc[neg_inds, 'obdm_up_17_17'])**2)**.5
  #val.loc[neg_inds, 'energy'] = (1-(val.loc[neg_inds, 'energy'])**2)**.5

  plt.figure(figsize=(6,3))
  plt.subplot(1,2,1)
  plt.errorbar(val.loc[p1inds,energyinds].values.T, val.loc[p0inds,energyinds].values.T,
               xerr=err.loc[p1inds,energyinds].values.T, 
               yerr=err.loc[p0inds,energyinds].values.T)
  plt.xlabel('energy derivs')
  plt.ylabel('energy eV')
  plt.subplot(1,2,2)
  plt.errorbar(val.loc[p1inds,obdminds].values.T, val.loc[p0inds,obdminds].values.T,
               xerr=err.loc[p1inds,obdminds].values.T, 
               yerr=err.loc[p0inds,obdminds].values.T)
  plt.xlabel('occupation derivs')
  plt.ylabel('occupation')

def plot_derivs_std_devs_from_zero(dfs):
  val = dfs['val']
  err = dfs['err']
  datainds = val.columns[val.columns.str.match('^energy|^obdm')]
  print(datainds.values)
  print(val['param'].values)
  print(val['filename'].values)
  p1inds = val['param']!=0
  rel_vals = val.loc[p1inds,datainds].values/err.loc[p1inds, datainds].values
  #x = np.arange(rel_vals.size)
  x = np.arange(rel_vals.shape[0])
  plt.figure(figsize=(4.5,3))
  #plt.plot(val.loc[p1inds,'param'].values)
  leg_list = []
  for e,r in enumerate(rel_vals.T):
    ## Test stuff
    h = plt.errorbar(x, val.loc[p1inds,datainds].values[:,e], yerr=err.loc[p1inds,datainds].values[:,e], 
              ls='', marker='')
    #h=plt.scatter(x, r)
    leg_list.append(h)
  #im = plt.scatter(x, rel_vals.ravel(), c=np.mod(x,rel_vals.shape[1]))
  plt.legend(leg_list, datainds.values, loc='center left', bbox_to_anchor=(1, 0.5))
  plt.ylabel('deriv')
  plt.xlabel('file no. x deriv no.')
  plt.title('Derivatives and errors')
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

def get_si_mf(chkfile, kpts=[0,0,0]):
  from pyscf import pbc, lib 
  from pyscf.pbc.dft import KRKS
  mol = pbc.gto.cell.loads(lib.chkfile.load(chkfile, 'mol'))
  kpts=mol.make_kpts(kpts)
  mf = KRKS(mol, kpts)
  mf.__dict__.update(lib.chkfile.load(chkfile,'scf'))
  return mol, mf



