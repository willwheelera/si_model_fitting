from pandas import DataFrame,Series
from json import loads
from numpy import array,einsum,diag,zeros,eye
import numpy as np

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
     # print(blockstr)
      if '{' in blockstr:
        block=loads(blockstr.replace("inf","0"))['properties']
        blockdict['energy'].append(block['total_energy']['value'][0])
        blockdict['dpenergy'].append(block['derivative_dm']['dpenergy']['vals'])
        blockdict['dpwf'].append(block['derivative_dm']['dpwf']['vals'])
        for s in ['up','down']:
          blockdict['obdm_%s'%s].append(block['derivative_dm']['tbdm']['obdm'][s])
          dprdmlist = [dprdm['tbdm']['obdm'][s] for dprdm in block['derivative_dm']['dprdm']]
          blockdict['dpobdm_%s'%s].append(dprdmlist)
        has_tbdm = 'tbdm' in block['derivative_dm']['tbdm']
        if has_tbdm:
          for s in ['upup','updown','downup','downdown']:
            tbdmdict['tbdm_%s'%s].append(block['derivative_dm']['tbdm']['tbdm'][s])
            dprdmlist = [dprdm['tbdm']['tbdm'][s] for dprdm in block['derivative_dm']['dprdm']]
            tbdmdict['dptbdm_%s'%s].append(dprdmlist)
        #  tmptbdm = {'dptbdm_upup':[],'dptbdm_updown':[],'dptbdm_downup':[],'dptbdm_downdown':[]}
        #tmpobdm = {'dpobdm_up':[],'dpobdm_down':[]}
        #for p,dprdm in enumerate(block['derivative_dm']['dprdm']):
        #  for s in ['up','down']:
        #    tmpobdm['dpobdm_%s'%s].append(dprdm['tbdm']['obdm'][s])
        #  if has_tbdm:
        #    for s in ['upup','updown','downup','downdown']:
        #      tmptbdm['dptbdm_%s'%s].append(dprdm['tbdm']['tbdm'][s])
        #for key,value in tmpobdm.items():
        #  blockdict[key].append(value)
        #if has_tbdm:
        #  for key,value in tmptbdm.items():
        #    blockdict[key].append(value)
        
  def unpack(vec,key):
    avec = np.array(vec)
    meshinds = np.meshgrid(*list(map(np.arange,avec.shape)))
    indices = list(zip(*list(map(np.ravel,meshinds))))
    dat = Series(dict(zip([key+'_'+'_'.join(map(str,i)) for i in indices], avec.ravel())))
    #indices=range(len(vec))
    #dat=Series(dict(zip([key+'%d'%i for i in indices],vec)))
    return dat

  def lists_to_cols(blockdf, key):
    expanded_cols = blockdf[key].apply(lambda x:unpack(x,key=key))
    return blockdf.join(expanded_cols).drop(key,axis=1)

  print('dict loaded from jsons')
  if has_tbdm:
    blockdict.update(tbdmdict)
  blockdf=DataFrame(blockdict)
  for key in blockdict.keys():
    if key=='energy': continue
    blockdf = lists_to_cols(blockdf, key)
  #blockdf=blockdf.join(blockdf['dpenergy'].apply(lambda x:unpack(x,key='dpenergy'))).drop('dpenergy',axis=1)
  #blockdf=blockdf.join(blockdf['dpwf'].apply(lambda x:unpack(x,key='dpwf'))).drop('dpwf',axis=1)
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
  df is a dataframe where each row is a sample and each column is a calculated quantity
  reblock each column over samples to find the best block size
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
    #stats.append(dict(serr=serr, serrerr=serrerr, iblock=iblock))
    #B3 = 2**(3*iblock)
    #inds = np.where(B3 > 2*ndata*(serr/serr0)**4)[0]
    #optimal_block[inds] = iblock

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

def gather_json(jsonfn):
  # Not using this function yet; might not work/be useful
  ''' 
  Args:
    jsonfn (str): name of json file to read.
  Returns:
    dictionary
  '''
  blockdict={
      'energy':[],
      'dpenergy':[],
      'dpwf':[],
      'obdm':{'up':[], 'down':[]},
      'dpobdm':{'up':[], 'down':[]},
      'tbdm':{'upup':[],'updown':[],'downup':[],'downdown':[]},
      'dptbdm':{'upup':[],'updown':[],'downup':[],'downdown':[]},
      }
  with open(jsonfn) as jsonf:
    for blockstr in jsonf.read().split("<RS>"):
     # print(blockstr)
      if '{' in blockstr:
        block=loads(blockstr.replace("inf","0"))['properties']
        has_tbdm = 'tbdm' in block['derivative_dm']
        blockdict['energy'].append(block['total_energy']['value'][0])
        blockdict['dpenergy'].append(block['derivative_dm']['dpenergy']['vals'])
        blockdict['dpwf'].append(block['derivative_dm']['dpwf']['vals'])
        for dm in ['obdm','tbdm']:
          for key in blockdict[dm].keys():
            blockdict[dm][key].append(block['derivative_dm'][dm][key])
            blockdict['dp'+dm][key].append( [dprdm['tbdm'][dm][key] for dprdm in block['derivative_dm']['dprdm']])
          if not has_tbdm: break
  return blockdict

