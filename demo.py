import os
import time
import numpy as np
import matplotlib.pyplot as plt
from prns import invert

fdir = os.path.dirname(os.path.abspath(__file__))
ftarget = os.path.join(fdir, "prns", "target-example.txt")
target = np.loadtxt(ftarget)

# showing the target
plt.imshow(target)
plt.title("Proton radiography example profile")
plt.colorbar()
plt.show()

# do the retrieval
nsamples = 20 # 20 sample is too few for real examples. you should collect as much as you like
print("Retrieving with %d samples (approx ~5s / sample)..." % nsamples)
t0 = time.time()
all_phis, all_sources, all_targetRs, all_fs = invert(target, nsamples=nsamples, return_all=True)
print("Done with %d samples in %fs" % (nsamples, time.time() - t0))

# calculate the mean and std
mean_phis = np.mean(all_phis, axis=0)
std_phis = np.std(all_phis, axis=0)
plt.subplot(1,2,1)
plt.imshow(mean_phis)
plt.title(r"Mean $\Phi$")
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(std_phis)
plt.title(r"Std $\Phi$")
plt.colorbar()
plt.show()

if nsamples < 100:
    print("%d samples are too few. "
          "You should collect as much samples as you can "
          "to provide robust statistics." % nsamples)
