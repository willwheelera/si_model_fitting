'''
Some simple tests to check that autogen runs without erroring out.

Run repeatedly to check if runner is also checking queued items correctly.
'''

#from autogenv2 import autopyscf,autorunner,crystal
#from crystal import CrystalWriter
from autopyscf import PySCFWriter,PySCFPBCWriter,dm_set_spins
from pyscfmanager import PySCFManager
#from crystalmanager import CrystalManager
from qwalkmanager import QWalkManager
from autorunner import PySCFRunnerLocal,PySCFRunnerPBS,RunnerPBS
from variance import VarianceWriter,VarianceReader
from linear import LinearWriter,LinearReader
from dmc import DMCWriter,DMCReader
from trialfunc import SlaterJastrow
import sys



def si_pyscf():
  ''' Simple tests that check PBC is working Crystal, and that QMC can be performed on the result.'''
  jobs=[]

  pwriter=PySCFPBCWriter({
      'gmesh':[16,16,16],
      'cif':open('si.cif','r').read(),
      'dft':'pbe0',
      'method':'ROKS',
      'basis':'bfd_vtz',
      'ecp':'bfd'
    })
  pman=PySCFManager(
      name='scf',
      path='test_sipyscf',
      writer=pwriter,
      runner=PySCFRunnerPBS(
          queue='secondary',
          nn=1,
          np=16,
          walltime='4:00:00'
        )
    )
  jobs.append(pman)

  return jobs


def run():
  ''' Choose which tests to run and execute `nextstep()`.'''
  jobs=[]
  jobs+=si_pyscf()

  for job in jobs:
    job.nextstep()

if __name__=='__main__':
  run()
