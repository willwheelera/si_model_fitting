import autopyscf as ap
from autopyscf import PySCFPBCWriter, PySCFReader
from pyscfmanager  import PySCFManager
from autorunner import RunnerPBS, PySCFRunnerPBS
import numpy as np
import pickle

inf = float('inf')
basis_args = [ '/home/will/Python/autogenv2/BFD_Library.xml', ['Si'] ]
basis_kwargs = dict( min_exp=0.2, naug=0, alpha=3, cutoff=0,
  basis_name='vtz', nangular={"s":inf,"p":inf,"d":inf,"f":inf,"g":inf})

param_set = [dict(naug=0, cutoff=.0),
             dict(naug=0, cutoff=.06),
             dict(naug=1, cutoff=.1),
             dict(naug=2, cutoff=.2)]

bases = []
bases.append(ap.edit_xml_basis(*basis_args, **basis_kwargs))
basis_kwargs.update(cutoff=.06)
bases.append(ap.edit_xml_basis(*basis_args, **basis_kwargs))
basis_kwargs.update(naug=1, cutoff=.1)
bases.append(ap.edit_xml_basis(*basis_args, **basis_kwargs))
basis_kwargs.update(naug=2, cutoff=.2)
bases.append(ap.edit_xml_basis(*basis_args, **basis_kwargs))

jobs = []
for params in param_set:
  basis_kwargs.update(**params)
  basis = ap.edit_xml_basis(*basis_args, **basis_kwargs)
  pwriter = PySCFPBCWriter(dict(
        gmesh=[16,16,16],
        cif=open('si.cif','r').read(),
        basis=ap.format_basis(basis),
        xc='pbe,pbe',
        method='RKS',
        kpts=[4,4,4],
        remove_linear_dep=True
  ))
  pman = PySCFManager(
    name='scf',
    path='si_pyscf_aug{0}_cutoff{1}'.format(params['naug'],params['cutoff']),
    writer=pwriter,
    runner=PySCFRunnerPBS(
      queue='secondary',
      nn=1,
      np=16,
      walltime='4:00:00'
    )
  )
  jobs.append(pman)
with open('jobs.pkl','wb') as f:
  pickle.dump(jobs,f)
