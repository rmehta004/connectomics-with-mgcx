# MARCC Job
import numpy as np
from joblib import Parallel, delayed
import sys
import nibabel as nib
import pickle

from mgcpy.independence_tests.mgcx import MGCX

num_cores = 24
# num_cores = 2
num_datasets = 200
index = int(sys.argv[1])
roi = [3, 42, 78, 92, 113, 129, 151]

def worker(s):
    # Parameters.
    M = 20

    # Load image - individual 100307.
    img = nib.load("rfMRI_REST1_LR_Atlas_hp2000_clean_filt_sm6.Schaefer400_7Network.ptseries.nii")
    data = np.array(img.get_fdata())
    n = data.shape[0]

    # Run MGC(i, j)
    X = data[:,roi[index]].reshape(n, 1)
    Y = data[:,s].reshape(n, 1)

    mgcx = MGCX(max_lag = M)
    test_statistic, metadata = mgcx.test_statistic(X, Y)
    optimal_lag = metadata['optimal_lag']
    optimal_scale_X = metadata['optimal_scale'][0]
    optimal_scale_Y = metadata['optimal_scale'][1]
    p_value = mgcx.p_value(X, Y)

    output = (index, s, test_statistic, optimal_lag, optimal_scale_X, optimal_scale_Y, p_value)
    return(output)

# Parallelize.
result = Parallel(n_jobs=num_cores, verbose=10)(delayed(worker)(s) for s in range(num_datasets))
filename = 'results_hcp_roi_%d.pkl' % index
output_file = open(filename, 'wb')
pickle.dump(result, output_file)
output_file.close()
