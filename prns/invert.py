import random
import numpy as np
import sunbear as sb
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from scipy.stats import multivariate_normal
from scipy.interpolate import interp2d

seed = 2018
random.seed(seed)
np.random.seed(seed)

__all__ = ["invert"]

def invert(target,
           nsamples=1000,
           # prior of the source profiles
           length_scale=(0.2, 2), lscale_log=True,
           amplitude=0.1, kernel_cls=RBF,
           # function options
           return_all=False):
    """
    Perform the statistical inversion without the source profiles.
    The length unit here is normalized so that the longest side of the `target`
    is 2 unit.
    The intensity unit is also normalized where the mean of the target equals
    to 1.

    Parameters
    ----------
    * target : 2-dimensional np.ndarray
        The proton radiography profile.
    * nsamples : int (optional)
        The number of samples of source profiles. Default: 1000.
    * length_scale : tuple of float (optional)
        The lower and upper bound of the correlation length of the source
        profiles. Default: (0.2, 2).
    * lscale_log : bool (optional)
        Flag indicating if the length_scale should be sampled in logscale.
        Default: True.
    * amplitude : float (optional)
        The amplitude of the deviation in the source profiles. It is expressed
        in a normalized unit where the mean value of the `target` is 1.
        Default: 0.1.
    * kernel_cls : scikit-learn kernel (optional)
        Class of the kernel in generating the source profiles. Default: RBF.
    * return_all : bool (optional)
        If True, then it returns a list of phis, generated sources,
        reconstructed targets, and the optimum values of the phis (useful
        in checking whether the reconstruction was successful).
        Default: False.
    """

    # normalize the target to have mean = 1
    target = target / np.mean(target)

    ndiff_samples = int(np.ceil(nsamples / 10.0))
    nsamples_same = 10 # number of samples with the same length scale
    all_sources = []
    all_phis = []
    all_targetRs = []
    all_fs = []
    for i in range(ndiff_samples):
        # sample the length_scale
        lscale = sample(length_scale, logscale=lscale_log)
        kernel = kernel_cls(lscale) * (amplitude)**2
        sources, phis, targetRs, fs = invert_with_kernel(target, kernel,
            nsamples=nsamples_same)

        if i == 0:
            all_sources = sources
            all_phis = phis
            all_targetRs = targetRs
            all_fs = fs
        else:
            all_sources = np.concatenate((all_sources, sources))
            all_phis = np.concatenate((all_phis, phis))
            all_targetRs = np.concatenate((all_targetRs, targetRs))
            all_fs = np.concatenate((all_fs, fs))

        print("%d out of %d samples collected" % (len(all_fs), nsamples))

    if return_all:
        return all_phis, all_sources, all_targetRs, all_fs
    else:
        return all_phis

def generate_sources(x1, y1, downsample, kernel, nsamples=1):
    # return np.ones(shape)
    gpr = GaussianProcessRegressor(kernel)
    shape = (len(x1), len(y1))

    # get the downsampled version
    shape2 = (shape[0]//downsample, shape[1]//downsample)
    x12 = np.linspace(np.min(x1), np.max(x1), shape2[0])
    y12 = np.linspace(np.min(y1), np.max(y1), shape2[1])
    x2, y2 = np.meshgrid(x12, y12)
    X2 = np.array([x2.ravel(), y2.ravel()]).T

    shape2_all = (nsamples, shape2[0], shape2[1])
    ss2 = gpr.sample_y(X2, random_state=seed, n_samples=nsamples).T\
             .reshape(shape2_all)

    sources = []
    for s2 in ss2:
        # interpolate
        f = interp2d(x12, y12, s2, kind="cubic")
        s = f(x1, y1)
        source = s.reshape(shape) + 1.0

        # normalize the mean
        source = source / np.mean(source)
        sources.append(source)
    return np.array(sources)

def invert_with_kernel(target, kernel, nsamples=1):
    ny, nx = target.shape
    ndownsample = 4 # downsample to speed up the gaussian process sampling
    ratio = nx * 1.0 / ny
    if ratio >= 1:
        x1 = np.linspace(-1.0, 1.0, nx)
        y1 = np.linspace(-1./ratio, 1./ratio, ny)
    else:
        x1 = np.linspace(-ratio, ratio, nx)
        y1 = np.linspace(-1, 1, ny)

    # generate the sources
    sources = generate_sources(x1, y1, ndownsample, kernel, nsamples)

    phis = []
    targetRs = []
    fs = []
    for i,source in enumerate(sources):
        # retrieve the magnetic field and reconstruct the target
        phi, f = sb.inverse(source, target, return_f=True)
        targetR = sb.forward(source, phi)

        # save them
        phis.append(phi)
        targetRs.append(targetR)
        fs.append(f)

    phis = np.asarray(phis)
    targetRs = np.asarray(targetRs)
    fs = np.asarray(fs)
    return sources, phis, targetRs, fs

def sample(bounds, logscale=False):
    if not logscale:
        return np.random.random() * (bounds[1] - bounds[0]) + bounds[0]
    else:
        blog = (np.log(bounds[0]), np.log(bounds[1]))
        return np.exp(sample(blog, False))
