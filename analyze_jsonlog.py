import numpy as np
import pandas as pd
import sys
sys.path.append('../')
import time
import json
import multiprocessing as mp

def gather_block(blockstr, cutoff_index=1):
  blockdict={ 'energy':[], 'dpenergy':[], 'dpwf':[], }
  block = json.loads(blockstr.replace("inf","0").replace("nan","0"))['properties']
  blockdict['energy'] = block['total_energy']['value'][0]
  derivs = 'derivative_dm' in block
  if derivs:
    derivdm = block['derivative_dm'][cutoff_index] # second cutoff 
    blockdict['dpenergy'] = derivdm['dpenergy']['vals']
    blockdict['dpwf'] = derivdm['dpwf']['vals']
  else:
    derivdm = block

  if not 'tbdm' in derivdm:
    return blockdict
      
  blockdict['normalization'] = derivdm['tbdm']['normalization']['value']

  has_obdm =  'obdm' in derivdm['tbdm']
  if has_obdm: 
    for s in ['up','down']:
      blockdict['obdm_%s'%s] = derivdm['tbdm']['obdm'][s]
      if derivs:
        dprdmlist = [dprdm['tbdm']['obdm'][s] for dprdm in derivdm['dprdm']]
        blockdict['dpobdm_%s'%s] = dprdmlist 

  has_tbdm = 'tbdm' in derivdm['tbdm']
  if has_tbdm:
    for s in ['upup','updown','downup','downdown']:
      blockdict['tbdm_%s'%s] = derivdm['tbdm']['tbdm'][s]
      if derivs:
        dprdmlist = [dprdm['tbdm']['tbdm'][s] for dprdm in derivdm['dprdm']]
        blockdict['dptbdm_%s'%s] = dprdmlist 

  return blockdict

def gather_json_df(jsonfn, leave_as_matrices=False, parallel=False):
  ''' 
  Args:
    jsonfn (str): name of json file to read.
  Returns:
    DataFrame: dataframe indexed by block with columns for energy, and each dpenergy and dpwf.
  '''
  start = time.time()
  print('started', jsonfn, time.time()-start)
  with open(jsonfn) as jsonf:
    blockstr_list = jsonf.read().split("<RS>")[:-1]
  first_block = json.loads(blockstr_list[0].replace("inf","0").replace("nan","0"))['properties']
  if 'derivative_dm' in first_block:
    states = first_block['derivative_dm'][0]['tbdm']['states']
  else:
    states = first_block['tbdm']['states']

  print('Loading blocks...')
  if parallel:
    with mp.Pool(4) as pool:
      print('gather pool opened', time.time()-start)
      result = pool.map_async(gather_block, blockstr_list, chunksize=20)
      for t in range(1000):
        if result.ready(): break
        print('chunks left', result._number_left, time.time()-start); sys.stdout.flush()
        result.wait(5)
    blockdicts = result.get()
  else:
    blockdicts = list(map(gather_block, blockstr_list))
  #for i,blockstr in enumerate(blockstr_list):
  #print('dict loaded from jsons', time.time()-start); sys.stdout.flush()  
  #if has_obdm:
  #  blockdict.update(obdmdict)
  #if has_tbdm:
  #  blockdict.update(tbdmdict)
  blockdf = pd.DataFrame(blockdicts)
  print('converted to DataFrame', time.time()-start); sys.stdout.flush()  
  if not leave_as_matrices:
    blockdf=flatten_matrices(blockdf, states)
  print('unpacked matrices', time.time()-start); sys.stdout.flush()  
  return blockdf

def flatten_matrices(blockdf,states):

  ''' 
  Flatten 1- and 2-rdms 
  input: 
  blockdf - dataframe with matrices 
  labels -  labels for one axis of the matrix 
  output: 
  blockdf - dataframe with flattened values 
  ''' 
  print('flattening matrices in dataframe')
  newdf=None 
  for key in list(blockdf): 
    if key in ['energy', 'states']: 
      print('skipping', key)
      continue
    flat=np.stack(blockdf[key],axis=0) 
    meshinds = np.meshgrid(*list(map(np.arange,flat.shape[1:])), indexing='ij')
    if states is not None and ('bdm' in key or key=='normalization'):
      states = np.array(states)
      if not key.startswith('dp'):
        meshinds[0] = states[meshinds[0]] # if dp, first index is parameter, not state
      for i in range(1,len(meshinds)):
        meshinds[i] = states[meshinds[i]] 
    newlabels = [key+'_'+'_'.join(map(str,i)) for i in zip(*map(np.ravel,meshinds))]
    if 'bdm' in key:
      flat = flat.reshape(flat.shape[0], -1)
    if(newdf is None): newdf=pd.DataFrame(flat,columns=newlabels) 
    else: newdf=pd.concat((newdf,pd.DataFrame(flat,columns=newlabels)),axis=1) 
    blockdf=blockdf.drop(columns=[key]) 
  blockdf=pd.concat((blockdf,newdf),axis=1) 
  return blockdf

def unpack_matrices(df, states=None):
  def unpack(vec,key,states=None):
    # expand vector of arrays into series labeled by index
    avec = np.array(vec)
    
    print('avec',key, avec.shape)
    print(avec)
    meshinds = np.meshgrid(*list(map(np.arange,avec.shape)), indexing='ij')
    if states is not None and (key.find('bdm')>0 or key=='normalization'):
      states = np.array(states)
      if not key.startswith('dp'):
        meshinds[0] = states[meshinds[0]] # if dp, first index is parameter, not state
      for i in range(1,len(meshinds)):
        meshinds[i] = states[meshinds[i]] 
    labels = list(zip(*list(map(np.ravel,meshinds))))
    dat = pd.Series(dict(zip([key+'_'+'_'.join(map(str,i)) for i in labels], avec.ravel())))
    return dat

  def lists_to_cols(blockdf, key, states=None):
    # expand columns of arrays into separate columns labeled by index and remove original cols
    expanded_cols = blockdf[key].apply(lambda x:unpack(x,key=key,states=states))
    print(expanded_cols.keys())
    return blockdf.join(expanded_cols).drop(key,axis=1)

  for key in df.keys():
    if key in ['energy', 'states']: 
      print('skipping', key)
      continue
    else:
      blockdf = lists_to_cols(df, key, states=states)
      print('unpacked', key)
  print('reshaped columns')
  return blockdf

def undo_normalizations(df): # for reblocking
  states = [int(key.split('_')[1]) for key in df.columns if key.startswith('normalization')]
  nmo = len(states)
  nparams = np.count_nonzero([c.startswith('dpenergy') for c in df.columns])
  symmetrized = np.count_nonzero([c.startswith('obdm_up') for c in df.columns])==nmo*(nmo+1)/2
  for i,si in enumerate(states):
    for j,sj in enumerate(states[i:] if symmetrized else states): 
      norm_ij = np.sqrt(df['normalization_%i'%si]*df['normalization_%i'%sj])
      for spin in ['up','down']:
        name = 'obdm_{0}_{1}_{2}'.format(spin,si,sj)
        if name not in df.columns: continue
        df[name] *= norm_ij
        for p in range(nparams):
          df['dpobdm_{0}_{1}_{2}_{3}'.format(spin,p,si,sj)] *= norm_ij

def apply_normalizations(df): # after reblocking
  states = [int(key.split('_')[1]) for key in df.columns if key.startswith('normalization')]
  nmo = len(states)
  nparams = np.count_nonzero([c.startswith('dpenergy') for c in df.columns])
  symmetrized = np.count_nonzero([c.startswith('obdm_up') for c in df.columns])==nmo*(nmo+1)/2
  for i,si in enumerate(states):
    for j,sj in enumerate(states[i:] if symmetrized else states): 
      norm_ij = np.sqrt(df['normalization_%i'%si]*df['normalization_%i'%sj])
      for spin in ['up','down']:
        name = 'obdm_{0}_{1}_{2}'.format(spin,si,sj)
        if name not in df.columns: continue 
        df[name] /= norm_ij
        for p in range(nparams):
          df['dpobdm_{0}_{1}_{2}_{3}'.format(spin,p,si,sj)] /= norm_ij

def reblock(df, n):
  start = time.time()
  newdf = df.copy()
  print('copied', time.time()-start)
  undo_normalizations(newdf)
  print('undid normalization', time.time()-start)
  for i in range(n):
    m = newdf.shape[0]
    lasteven = m - int(m%2==1)
    newdf = (newdf[:lasteven:2]+newdf[1::2].values)/2
  print('reblocked %i times'%n, time.time()-start)
  apply_normalizations(newdf)
  print('redid normalization', time.time()-start)
  return newdf

def opt_block(df):
  """
  Finds optimal block size for each variable in a dataset
  df is a dataframe where each row is a sample and each column is a calculated quantity
  reblock each column over samples to find the best block size
  Returns optimal_block, a 1D array with the optimal size for each column in df
  """
  start = time.time()
  newdf = df.copy()
  undo_normalizations(newdf)
  iblock = 0
  ndata, nvariables = tuple(df.shape[:2])
  optimal_block = np.array([float('NaN')]*nvariables)
  serr0 = df.sem(axis=0).values
  print('serr0.shape',serr0.shape, time.time()-start)
  print('ndata',ndata, time.time()-start)
  statslist = []
  while(newdf.shape[0]>1):
    n = newdf.shape[0]
    serr = newdf.sem(axis=0).values
    serrerr = serr/(2*(newdf.shape[0]-1))**.5
    statslist.append((iblock, n, serr.copy()))

    print(iblock, n, 'serr=',serr[0],'see=',serrerr[0],'time=',time.time()-start)
    lasteven = n - int(n%2==1)
    newdf = (newdf[:lasteven:2]+newdf[1::2].values)/2
    iblock += 1
  for iblock, n, serr in reversed(statslist):
    B3 = 2**(3*iblock)
    inds = np.where(B3 >= 2*ndata*(serr/serr0)**4)[0]
    optimal_block[inds] = iblock
    print(iblock, len(inds), time.time()-start) 

  return optimal_block

def test_reblock(blockdf):
  '''
  Computes optimal blocks and compares across variables, and with pyblock
  Just for testing
  '''
  import pyblock
  start = time.time()
  optimal_block = opt_block(blockdf)
  print('found optimal block', time.time()-start)
  print('nans:', np.count_nonzero(np.isnan(optimal_block)))
  optimal_block[np.isnan(optimal_block)] = -1

  opt4 = blockdf.columns[np.where(optimal_block==4)[0]]
  bigopt4 = blockdf.columns[np.where(np.logical_and(optimal_block==4,blockdf.mean(axis=0)>.05))[0]]
  print(np.where(np.isin(opt4,bigopt4))[0])
  print(bigopt4)

  datalen, block_info, covariance = pyblock.pd_utils.reblock(blockdf[bigopt4])
  opt_block = pyblock.pd_utils.optimal_block(block_info)
  pyblock.plot.plot_reblocking(block_info)
  print(opt_block)

  print('block_info')
  print(data_len)
  print(block_info)
  summary = pyblock.pd_utils.reblock_summary(block_info)
  print('summary')
  print(summary)

  plt.plot(optimal_block)
  plt.show()

def symmetrize_obdm(blockdf):
  '''
  Symmetrizes 1RDM since t_ij=t_ji, and removes duplicate from the dataframe
  For RDM derivatives, the values will not be correct until added together
  Returns new dataframe with only the unique RDM elements (lower triangular part)
  '''
  start=time.time()

  cols = blockdf.columns
  norb = np.count_nonzero([c.startswith('normalization_') for c in cols])
  nparams = np.count_nonzero([c.startswith('dpenergy') for c in cols])
  states = [int(key.split('_')[1]) for key in cols if key.startswith('normalization')]
  norb = len(states)
  print('orbs',norb,'params',nparams) 
  print(blockdf.shape)

  print('starting ij+ji combination', time.time()-start); sys.stdout.flush()

  for o1 in range(norb):
    for o2 in range(o1):
      for spin in ['up','down']:
        blockdf['obdm_{0}_{1}_{2}'.format(spin,states[o1],states[o2])] += \
            blockdf['obdm_{0}_{2}_{1}'.format(spin,states[o1],states[o2])] 
        for p in range(nparams):
          blockdf['dpobdm_{0}_{1}_{2}_{3}'.format(spin,p,states[o1],states[o2])] += \
              blockdf['dpobdm_{0}_{1}_{3}_{2}'.format(spin,p,states[o1],states[o2])] 
  blockdf.drop(['obdm_{0}_{2}_{1}'.format(spin,states[o1],states[o2]) \
      for spin in ['up','down'] for o1 in range(norb) for o2 in range(o1)], axis=1, inplace=True)
  blockdf.drop(['dpobdm_{0}_{1}_{3}_{2}'.format(spin,p,states[o1],states[o2]) \
      for spin in ['up','down'] for p in range(nparams) for o1 in range(norb) for o2 in range(o1)], axis=1, inplace=True) 
  print('symmetrized; newcols', len(blockdf.columns), time.time()-start); sys.stdout.flush()
  return blockdf

def symmetrize_by_1body_h(blockdf, h):
  h = 0.5*(h+h.T)
  unique, index = np.unique(h, return_index=True)
  for u in unique:
    inds = list(zip(*np.where(h==u)))
    i0 = inds[0]
    for spin in ['up','down']:
      for i in inds[1:]:
        blockdf['obdm_{0}_{1}_{2}'.format(spin, *i0)] += \
            blockdf['obdm_{0}_{1}_{2}'.format(spin, *i)] 
        for p in range(nparams):
          blockdf['dpobdm_{0}_{1}_{2}_{3}'.format(spin,p,*i0)] += \
              blockdf['dpobdm_{0}_{1}_{2}_{3}'.format(spin,p,*i)] 
    blockdf = blockdf.drop(['obdm_{0}_{1}_{2}'.format(spin,*i) \
        for spin in ['up','down'] for i in inds[1:]], axis=1)
    blockdf = blockdf.drop(['dpobdm_{0}_{1}_{2}_{3}'.format(spin,p,*i) \
        for spin in ['up','down'] for p in range(nparams) for i in inds[1:]], axis=1) 
  print('newcols', len(blockdf.columns), time.time()-start)
  return blockdf
  
def select_degen(df, obdm_degen, spin_degen=True):
  """
  add together descriptors before bootstrapping to get errors
  keep energy and normalization in the labels
  df: pandas dataframe with descriptors as columns and blocks as rows
  obdm_degen: list of lists of degenerate model parameters as (i,j) int tuple
  """
  cols = df.columns
  nparams = cols.str.contains('dpenergy').sum()
  keepcols = list(cols[cols.str.match('.*energy.*|normalization|dpwf')])
  newdf = df[keepcols]
  
  newcols = []
  if spin_degen:
    for odlist in obdm_degen:
      names = ['obdm_%s_%i_%i'%(s,*od) for od in odlist for s in ['up','down']]
      dpnames_i = ['dpobdm_%s_%s_%i_%i'%(s,'%i',*od) for od in odlist for s in ['up','down']]
      for name in names:  
        assert name in cols, "name {0} is not in data: \n{1}".format(name,cols)
      newcols.append((names[0], df[names].sum(axis=1)))
      for p in range(nparams):
        dpnames = [dpn%p for dpn in dpnames_i]
        newcols.append((dpnames[0], df[dpnames].sum(axis=1).values)) 
  else:
    for odlist in obdm_degen:
      for s in ['up','down']:
        names = ['obdm_%s_%i_%i'%(s,*od) for od in odlist]
        dpnames_i = ['dpobdm_%s_%s_%i_%i'%(s,'%i',*od) for od in odlist]
        for name in names:  
          assert name in cols, "name {0} is not in data: \n{1}".format(name,cols)
        newcols.append((names[0], df[names].sum(axis=1)))
        for p in range(nparams):
          dpnames = [dpn%p for dpn in dpnames_i]
          newcols.append((dpnames[0], df[dpnames].sum(axis=1).values)) 
  newdf = pd.concat([df[keepcols], pd.DataFrame(dict(newcols))], axis=1)
  return newdf

def extrapolate_bootstrap(dfdmc01, dfdmc, dfvmc, nresamples=100, obdm_degen=None):
  if obdm_degen is not None:
    dfdmc01 = select_degen(dfdmc01, obdm_degen)
    dfdmc = select_degen(dfdmc, obdm_degen)
    dfvmc = select_degen(dfvmc, obdm_degen)
  cols = dfvmc.columns
  nparams = np.count_nonzero([c.startswith('dpenergy') for c in cols])
  nsamples = len(dfvmc[cols[0]])
  names = cols[cols.str.match('^energy.*|^.bdm')].values

  rslist = []
  for i in range(nresamples):
    rsmeans = 2*dfdmc01.sample(n=len(dfdmc01), replace=True, axis=0).mean(axis=0) \
             -1*dfdmc.sample(n=len(dfdmc), replace=True, axis=0).mean(axis=0).values 
             #-dfvmc.sample(n=len(dfvmc), replace=True, axis=0).mean(axis=0).values

    for col in names:
      if col.startswith('normalization'): continue
      if col=='energy':
        words = [col,'%i']
      else:
        words = col.split('_')
        words.insert(2,'%i')
      name = '_'.join(words)
      rsmeans[['dp'+name%p for p in range(nparams)]]-=\
        rsmeans[col] *\
        rsmeans[['dpwf_%i'%p for p in range(nparams)]].values
    rslist.append(rsmeans)
  rsdf = pd.concat(rslist, axis=1).T
  return rsdf

def resample(df, names, nsamples, nparams):
  #cols = df.columns
  #nparams = np.count_nonzero([c.startswith('dpenergy') for c in cols])
  #nsamples = len(df)
  #names = [c for c in cols if (c.startswith('energy') or c.startswith('obdm') or c.startswith('tbdm'))]
  #df = df.sample(n=nsamples, replace=True)

  #undo_normalizations(df)
  rsmeans = pd.DataFrame(df.mean(axis=0)).T
  #apply_normalizations(rsmeans)

  for col in names:
    if col=='energy':
      words = [col,'%i']
    else:
      words = col.split('_')
      words.insert(2,'%i')
    name = '_'.join(words)
    rsmeans[['dp'+name%p for p in range(nparams)]]-=\
      rsmeans[col].values[np.newaxis] *\
      rsmeans[['dpwf_%i'%p for p in range(nparams)]].values
  return rsmeans

def bootstrap(df, nresamples, obdm_degen=None):
  '''
  Bootstrap to get errors on E, rdm, dE/dp, and dRDM/dp
  df is a pandas DataFrame as loaded by gather_json_df(); should be obdm symmetrized
  '''
  if obdm_degen is not None:
    df = select_degen(df, obdm_degen)
  cols = df.columns
  nparams = np.count_nonzero([c.startswith('dpenergy') for c in cols])
  nsamples = len(df[cols[0]])
  names = cols[cols.str.match('^energy.*|^.bdm')].values
  #names = [c for c in cols if (c.startswith('energy') or c.startswith('obdm') or c.startswith('tbdm'))]
  #resamples = []
  start = time.time()
  #for rs in range(nresamples):
  class myiter:
    def __iter__(self):
      self.a = 0; return self
    def __next__(self):
      if self.a < nresamples: self.a += 1; return (df.sample(n=nsamples, replace=True, axis=0), names, nsamples, nparams)
      else: raise StopIteration

  with mp.Pool() as pool:
    print('resample pool opened', time.time()-start); sys.stdout.flush()
    result = pool.starmap_async(resample, myiter(), chunksize=4)
    for t in range(10000):
      if result.ready(): break
      print('chunks left', result._number_left, time.time()-start); sys.stdout.flush()
      result.wait(5)
  resamples = result.get()
  
  print('collecting resamples', time.time()-start); sys.stdout.flush()
  rsdf = pd.concat(resamples, axis=0)
  print('resamples', rsdf.shape, time.time()-start); sys.stdout.flush()
  return rsdf 

def get_deriv_estimate(blockdf, nbootstrap_samples=100, nreblock=2, obdm_degen=None, save_reblock_file=None):
  start = time.time()
  print('get_deriv_estimate started', time.time()-start); sys.stdout.flush()
  symmetrize_obdm(blockdf) 
  print('symmetrized', time.time()-start); sys.stdout.flush()
  blockdf = reblock(blockdf, nreblock) 
  print('reblocked', time.time()-start); sys.stdout.flush()
  if save_reblock_file is not None:
    blockdf.to_hdf(save_reblock_file, 'data')
  rsdf = bootstrap(blockdf, nbootstrap_samples, obdm_degen=obdm_degen)
  print('bootstrapped', time.time()-start); sys.stdout.flush()
  return rsdf.mean(), rsdf.std()

def get_gsw(fname):
  words = fname.split('.')
  words[-1] = 'slater'
  with open('.'.join(words),'r') as f:
    filestr = f.read()
    ind = filestr.find('DETWT')
    words = filestr[ind:ind+100].split()
    assert words[0].startswith('DETWT'), "BUG: words doesn't start with 'DETWT'"
    if words[1]=='{':
      return float(words[2])
    elif words[0]=='DETWT{': 
      return float(words[1])
    else: 
      print('Not sure how to interpret words')
      print(words[:10])
      return None

def helper_select(df, s):
  words = s.split('_')
  restr = '{0}_{1}.*_{2}_{3}$'.format(*words)
  tdf = df.filter(regex=restr,axis=0)
  return (s, tdf.values.ravel())
  
def extract_bootstrap_df(bootdf, diag_only=True, parallel=False):
  """ 
  bootdf: a pandas df with column index as filenames and row multiindex - first index is 'val'/'err', second index is descriptor name
  diag_only: save time by only gathering obdm diagonals
  returns: dict ('val','err') of dfs, where columns are descriptor names (not including derivs) and rows are samples (length nfiles*nderivs)
  """
  start = time.time()
  print('starting extract timer', time.time()-start)
  index = bootdf.loc['val'].index
  nfiles = len(bootdf.keys())
  #nstates = index.str.contains('normalization').sum()
  #states = [n.split('_')[1] for n in index[index.str.contains('normalization')]]
  nparams = index.str.contains('dpenergy').sum()

  newlabels = index[index.str.match('^energy|^.bdm')].values
  if diag_only:
    def isdiag(dmlabel):
      words = dmlabel.split('_')
      return words[0]!='obdm' or words[-1]==words[-2]
    newlabels = [s for s in newlabels if isdiag(s)]
  obdm_strs = [s for s in newlabels if s.startswith('obdm')]
  tbdm_strs = [s for s in newlabels if s.startswith('tbdm')]
  print('descriptors', len(newlabels), time.time()-start)

  class myiter:
    def __init__(self, it):
      self.i = iter(it)
    def __iter__(self):
      return self
    def __next__(self):
      try: s = next(self.i); return (df, s)
      except: raise StopIteration 

  dfs={}
  for val in ['val','err']:
    df = bootdf.loc[val]  
    newd = {s:None for s in newlabels}
    newd['energy']  = df.filter(like='energy',axis=0).values.ravel()
    print(val)
    print('collected energy, len', len(newd['energy']), time.time()-start)
    if parallel:
      with mp.Pool() as pool:
        print('extract_descriptor_matrix pool opened', time.time()-start); sys.stdout.flush()
        result = pool.starmap_async(helper_select, myiter(obdm_strs), chunksize=4)
        for t in range(10000):
          if result.ready(): break
          print('chunks left', result._number_left, time.time()-start); sys.stdout.flush()
          result.wait(5)
      newd.update( dict(result.get()) )
    else:
      nstates = index.str.contains('normalization').sum()
      for i,s in enumerate(obdm_strs):
        words = s.split('_')
        restr = '{0}_{1}.*_{2}_{3}$'.format(*words)
        tdf = df.filter(regex=restr,axis=0)
        newd[s] = tdf.values.ravel()
        df.drop(tdf.index.values, axis=0, inplace=True)
        # TODO copy for tbdm  
        if i%nstates>=0: print('collecting', s, val, len(newd[s]), time.time()-start)
    newd['param'] = list(np.repeat(range(nparams+1),nfiles))
    newd['filename'] = list(np.tile(bootdf.keys().values,nparams+1))
    print('params', nparams, 'filenames', (newd['param']==0).sum(), time.time()-start)
    dfs[val] = pd.DataFrame(newd)
  return dfs

def compute_and_save(fnames, save_name='saved_data.csv'):
  """
  Bootstrap df to get mean and stderr
  """
  dfs = []
  for fname in fnames:
    gsw = get_gsw(fname)
    blockdf=gather_json_df(fname)
    mean, std = get_deriv_estimate(blockdf, nbootstrap_samples=100)
    df = pd.DataFrame({'gsw':[gsw]*len(mean),
                       'filename':[fname]*len(mean),
                       'deriv':mean.index,
                       'value':mean.values,
                       'err':std.values})
    dfs.append(df)
  df = pd.concat(dfs)
  df.to_csv(save_name)
  return df

def compute_and_save_table(fnames, save_name='table_data.csv'):
  dfs = []
  for fname in fnames:
    gsw = get_gsw(fname)
    blockdf=gather_json_df(fname)
    mean, std = get_deriv_estimate(blockdf, nbootstrap_samples=100)
    df = pd.concat([mean,std], axis=1, keys=['val','err']) 
    df['filename'] = fname
    df['gsw'] = gsw
    dfs.append(df)
  df = pd.concat(dfs)
  df.to_csv(save_name)
  return df

def plot_bootstrapped(df):
  import matplotlib.pyplot as plt
  import seaborn as sns
  inds = df['deriv'].isin(['dpenergy_%i'%p for p in range(0,20,4)]
                         +['dpobdm_up_%i_3_3'%p for p in range(0,20,4)]
                         +['dpobdm_up_%i_3_4'%p for p in range(0,20,4)])
                         #+['dpobdm_down_%i_2_8'%p for p in range(0,20,4)])
  g = sns.FacetGrid(df[inds], col='deriv', col_wrap=5, sharey=False)
  g.map(plt.errorbar, 'gsw', 'value', 'err', ls='')
  plt.show()

if __name__=='__main__':
  # *.vmc.json files from QWalk output
  #import os
  fnames = sys.argv[1:]
  fname = fnames[0]
  words = fname.split('.')
  words.insert(-1,'pandas')
  name = '.'.join(words)
  start = time.time()
  print('started: finding optimal block')
  
  ## gather
  df = gather_json_df(fname)
  print('json gathered', time.time()-start)
  #df.to_json(name)
  #words[-1] = 'hdf'
  #df.to_hdf('.'.join(words), 'data')
  
  # opt_block
  #df = pd.read_json(name)
  print('df loaded', time.time()-start)
  print(df.keys())
  opt = opt_block(df)
  print(np.amin(opt), np.amax(opt))
  print(np.unique(opt, return_counts=True))

  print('==4\n',df.columns[opt==4].values[:,np.newaxis])
  print('==5\n',df.columns[opt==5].values[:,np.newaxis])

  #savename = os.path.commonprefix(fnames)
  #if savename[-1]!='.':
  #  savename = savename+'.'
  #try:
  #  df = pd.read_csv(savename+'saved_data.csv')
  #except Exception as err:
  #  print('could not open')
  #  print(err)
  #  df = compute_and_save(fnames, savename+'saved_data.csv')
