import copy
import sys
sys.path.append('../')
#from plot_tools import plot_deltas, plot_energies
#import wills_functions as will
from pyscf import pbc, lib
from pyscf.pbc.dft import KRKS
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import pickle 

ha2eV = 27.2114

def get_si_mf():
  chkfile = '/home/will/Research/model_fitting/si_model_fitting/conv_sipyscf/si_conv_k888/scf.py.chkfile'
  mol = pbc.gto.cell.loads(lib.chkfile.load(chkfile, 'mol'))
  kpts=mol.make_kpts([8, 8, 8])
  mf = KRKS(mol, kpts)
  mf.__dict__.update(lib.chkfile.load(chkfile,'scf'))
  return mf

def gen_GS(mf):
  mo_occ = mf.mo_occ.copy()
  return [(mo_occ,0,0)]

def gen_singles(mf, cutoff=.2):
  occ=mf.mo_occ[0]
  mo_en=mf.mo_energy[0]
  mo_coeff=mf.mo_coeff
  homo=np.max(mo_en[occ>0])
  occ_list=[np.where(occ >= 1e-6)[0],np.where(occ>=1+1e-6)[0]]
  virt_list=[np.where(occ < 1e-6)[0],np.where(occ < 1+1e-6)[0]]
  saved_occs=[] 

  print('len occ, virt')
  print(len(occ_list[0]),len(virt_list[0]))
  print(len(occ_list[1]),len(virt_list[1]))
  s=0 # singles excitations, only up spin
  for o in occ_list[s]:
    for v in virt_list[s]:
      delta = mo_en[v] - mo_en[o]
      if delta<cutoff:
        mo_occ=copy.deepcopy(mf.mo_occ)
        mo_occ[0][o]-=1.
        mo_occ[0][v]+=1.
        saved_occs.append( (mo_occ, 0, delta) )
  return saved_occs

def gen_doubles(mf, cutoff=.2):
  occ=mf.mo_occ[0]
  mo_en=mf.mo_energy[0]
  mo_coeff=mf.mo_coeff
  occ_list=[np.where(occ >= 1e-6)[0],np.where(occ >=1+1e-6)[0]]
  virt_list=[np.where(occ < 1e-6)[0],np.where(occ < 1+1e-6)[0]]
  saved_occs=[] 

  for o0 in occ_list[0][::-1]:
    for o1 in occ_list[1][::-1]:
      if o1>o0: continue
      for v0 in virt_list[0]:
        for v1 in virt_list[1]:
          if v1>v0: continue
          delta=mo_en[v0]+mo_en[v1]-mo_en[o1]-mo_en[o0]
          if delta<cutoff:
            mo_occ=copy.deepcopy(mf.mo_occ)
            mo_occ[0][o0]-=1.
            mo_occ[0][o1]-=1.
            mo_occ[0][v0]+=1.
            mo_occ[0][v1]+=1.
            saved_occs.append( (mo_occ, 0, delta) )
  return saved_occs

if __name__=='__main__':
  start = time.time()

  mf = get_si_mf()

  k=0
  print('occ',mf.mo_occ[k])
  inds = np.nonzero(mf.mo_occ[k])[0]
  print('inds',inds)
  print('energy',mf.mo_energy[k][:])

  homo = np.amax(inds)
  print('gap', mf.mo_energy[k][homo+1]-mf.mo_energy[k][homo])
  print('e_tot',mf.e_tot)

  cutoff = 17/ha2eV
  print('cutoff',cutoff*ha2eV)
  GS = gen_GS(mf)
  singles = gen_singles(mf,cutoff)
  #doubles = gen_doubles(mf,cutoff)
  saved_occs = GS+singles
  with open('saved_singles.pkl','wb') as f:
    pickle.dump(saved_occs, f)
  
  print('time',time.time()-start)

  plt.figure(figsize=(3,3))
  plt.plot(np.sort([so[2] for so in singles])*ha2eV)
  #plt.plot(np.sort([so[2] for so in doubles])*ha2eV, ls=':')
  plt.xlabel('determinant index')
  plt.ylabel('Orbital energy diff (eV)')
  plt.title('Est. determinant energies')
  plt.tight_layout()
  plt.show()
  
