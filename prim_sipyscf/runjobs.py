import pickle
import sys

assert len(sys.argv)>1, "need pkl filename(s)"

for fname in sys.argv[1:]:
  with open(fname,'rb') as f:
    job = pickle.load(f)
  job.nextstep()

