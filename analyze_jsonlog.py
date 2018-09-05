import numpy as np
import pandas as pd
import seaborn as sns
import sys
sys.path.append('../')
import pyblock
import time
import matplotlib.pyplot as plt
import json

def gather_json_df(jsonfn):
  ''' 
  Computing covariance is as easy as gather_json('my.json').cov()

  Args:
    jsonfn (str): name of json file to read.
  Returns:
    DataFrame: dataframe indexed by block with columns for energy, and each dpenergy and dpwf.
  '''
  blockdict={
      'energy':[],
      'dpenergy':[],
      'dpwf':[],
  }
  obdmdict={
      'obdm_up':[],
      'obdm_down':[],
      'dpobdm_up':[],
      'dpobdm_down':[],
  }
  tbdmdict={
      'tbdm_upup':[],
      'tbdm_updown':[],
      'tbdm_downup':[],
      'tbdm_downdown':[],
      'dptbdm_upup':[],
      'dptbdm_updown':[],
      'dptbdm_downup':[],
      'dptbdm_downdown':[],
  }
  with open(jsonfn) as jsonf:
    for blockstr in jsonf.read().split("<RS>"):
      if '{' in blockstr:
        block = json.loads(blockstr.replace("inf","0"))['properties']
        blockdict['energy'].append(block['total_energy']['value'][0])
        blockdict['dpenergy'].append(block['derivative_dm']['dpenergy']['vals'])
        blockdict['dpwf'].append(block['derivative_dm']['dpwf']['vals'])

        has_obdm = 'obdm' in block['derivative_dm']['tbdm']
        if has_obdm: 
          for s in ['up','down']:
            obdmdict['obdm_%s'%s].append(block['derivative_dm']['tbdm']['obdm'][s])
            dprdmlist = [dprdm['tbdm']['obdm'][s] for dprdm in block['derivative_dm']['dprdm']]
            obdmdict['dpobdm_%s'%s].append(dprdmlist) 

        has_tbdm = 'tbdm' in block['derivative_dm']['tbdm']
        if has_tbdm:
          for s in ['upup','updown','downup','downdown']:
            tbdmdict['tbdm_%s'%s].append(block['derivative_dm']['tbdm']['tbdm'][s])
            dprdmlist = [dprdm['tbdm']['tbdm'][s] for dprdm in block['derivative_dm']['dprdm']]
            tbdmdict['dptbdm_%s'%s].append(dprdmlist)

  def unpack(vec,key):
    # expand vector of arrays into series labeled by index
    avec = np.array(vec)
    meshinds = np.meshgrid(*list(map(np.arange,avec.shape)))
    indices = list(zip(*list(map(np.ravel,meshinds))))
    dat = pd.Series(dict(zip([key+'_'+'_'.join(map(str,i)) for i in indices], avec.ravel())))
    return dat

  def lists_to_cols(blockdf, key):
    # expand columns of arrays into separate columns labeled by index and remove original cols
    expanded_cols = blockdf[key].apply(lambda x:unpack(x,key=key))
    return blockdf.join(expanded_cols).drop(key,axis=1)

  print('dict loaded from jsons')
  if has_obdm:
    blockdict.update(obdmdict)
  if has_tbdm:
    blockdict.update(tbdmdict)
  blockdf = pd.DataFrame(blockdict)
  for key in blockdict.keys():
    if key=='energy': continue
    blockdf = lists_to_cols(blockdf, key)
  print('reshaped columns')
  return blockdf

def reblock(df, n):
  newdf = df.copy()
  for i in range(n):
    m = newdf.shape[0]
    lasteven = m - int(m%2==1)
    newdf = (newdf[:lasteven:2]+newdf[1::2].values)/2
  return newdf

def opt_block(df):
  """
  Finds optimal block size for each variable in a dataset
  df is a dataframe where each row is a sample and each column is a calculated quantity
  reblock each column over samples to find the best block size
  Returns optimal_block, a 1D array with the optimal size for each column in df
  """
  stats = dict(
      serr = [],
      serrerr = [])
  newdf = df.copy()
  iblock = 0
  ndata = df.shape[0]
  nvariables = df.shape[1]
  optimal_block = np.array([float('NaN')]*nvariables)
  serr0 = df.sem(axis=0).values
  print('serr0.shape',serr0.shape)
  print('ndata',ndata)
  statslist = []
  while(newdf.shape[0]>1):
    serr = newdf.sem(axis=0).values
    serrerr = serr/(2*(newdf.shape[0]-1))**.5
    statslist.append((iblock, serr.copy()))

    n = newdf.shape[0]
    print(iblock, n)
    lasteven = n - int(n%2==1)
    newdf = (newdf[:lasteven:2]+newdf[1::2].values)/2
    iblock += 1
  for iblock, serr in reversed(statslist):
    B3 = 2**(3*iblock)
    inds = np.where(B3 >= 2*ndata*(serr/serr0)**4)[0]
    optimal_block[inds] = iblock
    print(iblock, len(inds)) 

  return optimal_block

def test_reblock(blockdf):
  '''
  Computes optimal blocks and compares across variables, and with pyblock
  Just for testing
  '''
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
  cols = blockdf.columns
  norb = np.count_nonzero([c.startswith('obdm_up_0') for c in cols])
  nparams = np.count_nonzero([c.startswith('dpenergy') for c in cols])
  print('orbs',norb,'params',nparams)

  print(blockdf.shape)

  start=time.time()
  print('starting ij+ji combination')

  for o1 in range(norb):
    for o2 in range(o1):
      for spin in ['up','down']:
        blockdf['obdm_{0}_{1}_{2}'.format(spin,o1,o2)] += \
            blockdf['obdm_{0}_{2}_{1}'.format(spin,o1,o2)] 
        for p in range(nparams):
          blockdf['dpobdm_{0}_{1}_{2}_{3}'.format(spin,p,o1,o2)] += \
              blockdf['dpobdm_{0}_{1}_{3}_{2}'.format(spin,p,o1,o2)] 
  blockdf = blockdf.drop(['obdm_{0}_{2}_{1}'.format(spin,o1,o2) \
      for spin in ['up','down'] for o1 in range(norb) for o2 in range(o1)], axis=1)
  blockdf = blockdf.drop(['dpobdm_{0}_{1}_{3}_{2}'.format(spin,p,o1,o2) \
      for spin in ['up','down'] for p in range(nparams) for o1 in range(norb) for o2 in range(o1)], axis=1) 
  print('newcols', len(blockdf.columns), time.time()-start)
  return blockdf

def bootstrap(df, nresamples):
  '''
  Bootstrap to get errors on E, rdm, dE/dp, and dRDM/dp
  df is a pandas DataFrame as loaded by gather_json_df(); should be obdm symmetrized
  '''
  cols = df.columns
  norb = np.count_nonzero([c.startswith('obdm_up_0') for c in cols])
  nparams = np.count_nonzero([c.startswith('dpenergy') for c in cols])
  nsamples = len(df[cols[0]])
  names = [c for c in cols if not c.startswith('dp')]
  resamples = []
  start = time.time()
  for rs in range(nresamples):
    print('resample %i'%rs, time.time()-start)
    rsmeans = df.sample(n=nsamples, replace=True).mean(axis=0)
    funclist = []
    for col in names:
      #if col.startswith('dp'): continue
      if col=='energy':
        words = [col,'%i']
      else:
        words = col.split('_')
        words.insert(2,'%i')
      name = '_'.join(words)
      rsmeans[['dp'+name%p for p in range(nparams)]]-=\
        rsmeans[col][np.newaxis] *\
        rsmeans[['dpwf_%i'%p for p in range(nparams)]].values
    resamples.append(rsmeans)
  rsdf = pd.concat(resamples, axis=1).T
  return rsdf 

def get_deriv_estimate(blockdf, nbootstrap_samples=100):
  symmetrize_obdm(blockdf) 
  blockdf = reblock(blockdf, 2) 
  rsdf = bootstrap(blockdf, nbootstrap_samples)
  return rsdf.mean(), rsdf.std()

def get_gsw(fname):
  with open(fname.split('.')[0]+'.'+fname.split('.')[1]+'.slater','r') as f:
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

def compute_and_save(fnames, save_name='saved_data.csv'):
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

if __name__=='__main__':
  # *.vmc.json files from QWalk output
  fnames = sys.argv[1:]
  try:
    df = pd.read_csv('saved_data.csv')
  except Exception as err:
    print('could not open')
    print(err)
    df = compute_and_save(fnames, 'saved_data.csv')
  inds = df['deriv'].isin(['dpenergy_%i'%p for p in range(0,20,4)]
                         +['dpobdm_up_%i_3_3'%p for p in range(0,20,4)]
                         +['dpobdm_up_%i_3_4'%p for p in range(0,20,4)])
                         #+['dpobdm_down_%i_2_8'%p for p in range(0,20,4)])
  g = sns.FacetGrid(df[inds], col='deriv', col_wrap=5, sharey=False)
  g.map(plt.errorbar, 'gsw', 'value', 'err', ls='')
  plt.show()
