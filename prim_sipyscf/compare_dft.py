import numpy as np
import matplotlib.pyplot as plt
import pymatgen
from pymatgen import MPRester
import sys
sys.path.append('/home/will/Research/model_fitting/si_model_fitting')
import chkinfo

hf2ev = 27.2114

if __name__=='__main__':
  mpr = MPRester(api_key='D9uwc2I45fGbd8FU')
  bs = mpr.get_bandstructure_by_material_id('mp-149', line_mode=False)
  mproj_kpts = np.array([k._ccoords for k in bs.kpoints])
  xinds = np.nonzero(np.dot(mproj_kpts,[1,0,0])/np.linalg.norm(mproj_kpts, axis=1)==-1)[0]
  key = list(bs.bands.keys())[0]
  mproj_bands = bs.bands[key]

  print(mproj_bands.shape)
  print(np.round(mproj_kpts[[0,10,90]], 3))
  xkpts = np.abs(mproj_kpts[xinds,0])
  print(xinds)
  print(np.round(xkpts,3))
  print(bs.kpoints[0].__dict__.keys())
  #plt.plot(xkpts, mproj_bands[:6,xinds].T)
  #plt.xlabel('k (along $\Gamma$-$X$)')
  #plt.ylabel('band energy (eV)')
  #plt.title('materials project')
  #plt.show()

  nbands = 6
  fig, axs = plt.subplots(2,2, sharey=True)

  axs[0,0].plot(mproj_bands[:nbands,0])
  axs[0,1].plot(mproj_bands[:nbands,90])
  axs[1,0].plot(mproj_bands[:nbands,0]*0)
  axs[1,1].plot(mproj_bands[:nbands,90]*0)

  fnames = ['si_pyscf_aug0_cutoff0.0/scf.py.chkfile',
            'si_pyscf_aug0_cutoff0.06/scf.py.chkfile',
            'si_pyscf_aug1_cutoff0.1/scf.py.chkfile',
            'si_pyscf_aug2_cutoff0.2/scf.py.chkfile']
  gap = []
  for fname in fnames:
    mf = chkinfo.get_si_mf(fname, kmesh=(4,4,4))
    #print(np.around(mf.kpts[:10],3))
    inds = np.nonzero(np.abs(mf.kpts[:,0]/np.linalg.norm(mf.kpts, axis=1))==1)[0]
    print(inds)
    print(np.round(mf.kpts[inds],3))

    bands = np.array([mf.mo_energy[k] for k in (0,40)])*hf2ev
    axs[0,0].plot(bands[0][:nbands])
    axs[0,0].set_title('$\Gamma$-point energies')
    axs[0,0].set_ylabel('Energy (eV)')
    axs[0,1].plot(bands[1][:nbands])
    axs[0,1].set_title('$X$-point energies')
    axs[1,0].plot(bands[0][:nbands]-mproj_bands[:nbands,0])
    axs[1,0].set_ylabel('Energy (eV)')
    axs[1,0].set_title('$\Gamma$ differences')
    axs[1,1].plot(bands[1][:nbands]-mproj_bands[:nbands,90])
    axs[1,1].set_title('$X$ differences')
    cutoff = float(fname.split('/')[0][20:])
    gap.append((cutoff,bands[1][0]-bands[0][0]))
  axs[0,1].legend(['mat_proj','0.0','0.06','0.1','0.2'], 
      bbox_to_anchor=(.5,.0), loc='lower left')
  plt.tight_layout()
  plt.show()

  ax = plt.axes()
  mpgap = mproj_bands[0,90]-mproj_bands[0,0]
  ax.axhline(y=mpgap)
  ax.scatter(0,mpgap)
  for g in gap:
    ax.scatter(*g)
  ax.set_title('$\Gamma$-$X$ gap')
  ax.set_xlabel('BFD basis cutoff')
  ax.set_ylabel('Energy (eV)')
  ax.legend(['','mat_proj','0.0','0.06','0.1','0.2']) 
  plt.show()


