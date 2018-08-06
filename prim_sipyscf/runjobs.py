import pickle

with open('jobs.pkl','rb') as f:
  jobs = pickle.load(f)

for job in jobs:
  job.nextstep()
