from scipy.linalg import eigh
import numpy as np
from time import time
a = np.random.rand(3000, 3000)
a = a + a.T
t0 = time()
w, v = np.linalg.eigh(a)
print time() - t0
