'''
Module contains function for calculating the left-tail probability of the sum
of RVs based on a numerical method emplying convolutions
'''
from typing import Tuple, Callable
import numpy as np
from numba import jit
from scipy.stats import lognorm
from scipy.integrate import simpson, trapezoid

@jit(nopython=True)
def find_convolution(f, h, n):
    '''
    Optimized function calculating convolution exploting iid of RVs

    Parameters
    ----------
    f : np.ndarray
        Discretized pdf of the RVs in the sum
    h : float
        Step size of mesh
    n : int
        Number of convolutions 

    Returns
    -------
    f : np.ndarray
        Input function f n-times convoluted
    '''

    m = f.shape[0]
    f_conv = np.zeros(m)
    f_conv_prv = f.copy()
    f_conv_prv_flip = np.ascontiguousarray(np.flip(f.copy()))
    l = int(np.floor(np.log2(n))) # Numer of iterations
    # Calculate f 2**l times convoluted
    for _ in range(l):
        for i in range(m):
            f_conv[i] = h*(f_conv_prv[:i+1]@f_conv_prv_flip[-(i+1):] - f_conv_prv[0]*f_conv_prv[i])
        f_conv_prv = f_conv.copy()
        f_conv_prv_flip = np.ascontiguousarray(np.flip(f_conv.copy()))
    # If n is not a power of two, we calculate n-2**l further convolutions with f.
    # Implementation could be optimized further takeing advantage of convolutions calculated above
    if 2**l < n:
        for _ in range(n-2**l):
            for i in range(m):
                f_conv[i] = h*(f[:i+1]@f_conv_prv_flip[-(i+1):] - f_conv_prv[0]*f_conv_prv[i])
            f_conv_prv_flip = np.ascontiguousarray(np.flip(f_conv.copy()))
    return f_conv

def convolution_estimate(gamma: int,
                         m: int | None = None,
                         n: int | None = None,
                         distr_func: Callable[[np.ndarray], np.ndarray] | None = None,
                         f: np.ndarray | None = None,
                         kind: str = 'boole'
                         ) -> (Tuple[float, float] | Tuple[float, float, float, float]):
    '''
    Method to compute estimate of cdf and pdf at n*x for the sum of RVs. Note that either m, n and
    distr_func or f need to be supplied.

    If arguments m, n and distr_func is supplied we assume that we are calculating the value of the
    cdf and pdf of the sum of n RVs with distribution function defined by distr_func at the point 
    gamma = zn, using m+1 interpolation points for the numerical integrations.

    Otherwise, if f is supplied we assume that it is a n x (m+1) matrix, with row i containing the
    values of RV i in the sum for each of the m + 1 points used for calculating the convolution,
    i.e. f[i,j] = f_i(x_j), where f_i is the pdf of X_i and X_{m+1} = z*n.

    Parameters
    ----------
    gamma : int
        Value at which we want to estimate the cdf/pdf
    m : int, optional
        Number of intervals in the mesh used in the numerical integration to estimate the 
        convolution
    n : int, optional
        Number of i.i.d variables in the sum
    distr_func : Callable, optional
        Distribution function used to calculate cdf
    f : np.array, optional
        An n x (m+1) matrix with row i containing the values of RV i in the sum for each of the
        m + 1 points used for calculating the convolution
    kind : str, default = 'boole'
        The kind of integration method used to estimate the cdf from the convolution
        - 'trapezoid'
        - 'simpson'  
        - 'boole' 
        - 'all'

    Returns
    -------
    cdf : float
        Value of cdf of S_n at the point gamma = z*n calculated using method as specified by kind
        If kind is all, three values for the cdf is returned
    pdf : flaot
        Value of pdf of S_n at the point gamma = z*n    
    '''

    if kind not in ['trapezoid', 'simpson', 'boole', 'all']:
        raise Warning(f'Unsupported kind argument {kind} was passed')

    # If m, n or distr_func is not supplied, find n and m from f
    if m is None or n is None or distr_func is None:
        n, m = f.shape
        m -= 1
    # Find x-values in mesh and step size h
    x, h = np.linspace(0, gamma, m+1, retstep=True)

    # If m, n and distr_func is supplied
    if m is not None and n is not None and distr_func is not None:
        f = distr_func(x)
        f_conv = find_convolution(f, h, n)
    # Otherwise, calculate f_conv from f using a slower implementation, not optimized and not
    # explointing that we have iid RVs...
    else:
        f_conv = np.zeros(x.shape)
        f_conv_prv = f[0,:].copy()
        for j in range(1,n):
            for i in range(m+1):
                f_conv[i] = h*(f_conv_prv[:i+1]@np.flip(f[j,:i+1]))
            f_conv_prv = f_conv.copy()
    # Return cdf and pdf estiamtes
    if kind == 'trapezoid':
        return trapezoid(f_conv, x), f_conv[-1]
    elif kind == 'simpson':
        return simpson(f_conv, x), f_conv[-1]
    else:
        boole = (2*h/45)*(7*(f_conv[0] + f_conv[-1])
                          + 32*sum(f_conv[1:-1:2])
                          + 12*sum(f_conv[2:-2:4])
                          + 14*sum(f_conv[4:-4:4]))
        if kind == 'boole':
            return boole, f_conv[-1]
        elif kind == 'all':
            return trapezoid(f_conv, x), simpson(f_conv, x), boole, f_conv[-1]

if __name__ == '__main__':
    # Asmussen et al. example, using gamma = z*N
    N = 16         # Number of RVs
    M = int(1e4)   # Number of intervals in mesh
    SIGMA = 0.125  # Shape value of lognormal distribution
    zs = [0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.98]

    print(f'{"x":^5s}| {"Cdf":^12s}|{"Pdf":^12s}')
    for z in zs:
        est_cdf, est_pdf = convolution_estimate(z*N, m=M, n = N,
                                                distr_func=lambda x: lognorm.pdf(x, SIGMA),
                                                kind = 'boole')
        print(f'{z:^5.2f}|{est_cdf:^12.3E}|{est_pdf:^12.3E}')

    print(f'Varying sigma:\n{"x":^5s}| {"Cdf":^12s}|{"Pdf":^12s}')
    sigmas = 4*[0.25] + 4*[0.125] + 4*[0.072] + 4*[0.035]
    for z in zs:
        mesh = np.linspace(0, z*N, M+1)
        pdfs = np.array([lognorm.pdf(mesh, sigma) for sigma in sigmas])
        est_cdf, est_pdf = convolution_estimate(z*N, M, f = pdfs)
        print(f'{z:^5.2f}|{est_cdf:^12.3E}|{est_pdf:^12.3E}')
