import sys
sys.path.append('../')
#from plot_tools import plot_deltas, plot_energies
#import wills_functions as will
from pyscf import pbc, lib
from pyscf.pbc.dft import KRKS
import numpy as np
import pickle

def get_si_mf(chkfile,kmesh=(1,1,1)):
  mol = pbc.gto.cell.loads(lib.chkfile.load(chkfile, 'mol'))
  kpts=mol.make_kpts(kmesh)
  mf = KRKS(mol, kpts)
  mf.__dict__.update(lib.chkfile.load(chkfile,'scf'))
  return mf

if __name__=='__main__':
  assert len(sys.argv)>1, "need chkfile"
  mf = get_si_mf(sys.argv[1])

  print(len(mf.mo_occ), mf.mo_occ[0].shape)
  print(len(mf.mo_coeff), mf.mo_coeff[0].shape)
  #print(np.round(mf.cell.get_scaled_kpts(mf.kpts),3))
  L = mf.cell.lattice_vectors()
  G = mf.cell.reciprocal_vectors()

  k=0
  print('occ',mf.mo_occ[k])
  inds = np.nonzero(mf.mo_occ[k])[0]
  print('inds',inds)
  print('energy',mf.mo_energy[k][:16])
  homo = np.amax(inds)
  print('direct gap', 27.2114*np.amin([mf.mo_energy[k][homo+1]-mf.mo_energy[k][homo] for k in range(64)]))
  print('gap', 27.2114*(np.amin([mf.mo_energy[k][homo+1] for k in range(64)])
              -np.amax([mf.mo_energy[k][homo] for k in range(64)])))
  print('e_tot',mf.e_tot)


  print('calculate electron energy')
  #print(mf.energy_tot())

