# MARCC Test
import numpy as np
from joblib import Parallel, delayed
import sys
import nibabel.cifti2 as ci
import pickle

from mgcpy.independence_tests.mgcx import MGCX

num_cores = 24
# num_cores = 2
num_datasets = 360
num_runs = 360**2 // 5
# num_runs = 1
index = int(sys.argv[1])
start = index * num_runs
end = (index + 1) * num_runs

def worker(s):
    # Parameters.
    M = 1

    # Load image - individual 100307.
    img = ci.load("test_fmri_100307.nii")
    data = np.array(img.get_fdata())
    n = data.shape[0]
    p = data.shape[1]

    # Run MGC(i, j)
    i = s // p
    j = s % p
    X = data[:,i].reshape(n, 1)
    Y = data[:,j].reshape(n, 1)

    mgcx = MGCX(max_lag = M)
    test_statistic, metadata = mgcx.test_statistic(X, Y)
    optimal_lag = metadata['optimal_lag']
    optimal_scale_X = metadata['optimal_scale'][0]
    optimal_scale_Y = metadata['optimal_scale'][1]

    output = (i, j, test_statistic, optimal_lag, optimal_scale_X, optimal_scale_Y)
    return(output)

# BIG DATA !!!!11!!1!
result = Parallel(n_jobs=num_cores, verbose=10)(delayed(worker)(s) for s in range(start, end))
filename = 'result_%d.pkl' % index
output_file = open(filename, 'wb')
pickle.dump(result, output_file)
output_file.close()
