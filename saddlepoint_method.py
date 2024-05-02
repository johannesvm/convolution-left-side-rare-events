'''
Module contains function for calculating the left-tail probability of the sum
of RVs based on a numerical method emplying saddlepoints
'''

import numpy as np
from scipy.stats import norm
from scipy.special import lambertw

# Approximating theta
def gamma(x:int, sigma:float):
    'Function representing gamma for theta approximation'
    return 1/2*(-1 - np.log(x) + np.sqrt((1-np.log(x))**2 + 2*sigma**2))

def theta_aprox(x:int, sigma: float):
    'Funciton approximating theta before using Newtons method to get better estimate'
    return round(gamma(x, sigma)*np.exp(gamma(x, sigma))/sigma**2, 2)

def w_k_gen(k:int, sigma:float):
    'Generator for w_k(theta) function'
    return lambda theta: lambertw(theta*sigma**2*np.exp(k*sigma**2)).real

def sigma_k_squared_gen(k:int, sigma:float):
    'Generator for sigma_k_squared(theta) function'
    return lambda theta: sigma**2/(1+w_k_gen(k, sigma)(theta))

# Derivative of Laplace transform
def laplace_a_gen(k: int, sigma: float):
    'Function generating laplace_a(theta) funciton'
    w_k = w_k_gen(k, sigma)
    sigma_k_squared = sigma_k_squared_gen(k, sigma)
    return lambda theta: (np.sqrt(sigma_k_squared(theta))
                          /sigma*np.exp(-1/(2*sigma**2)*w_k(theta)**2
                                        - 1/sigma**2*w_k(theta)
                                        + 1/2*k**2*sigma**2))

def i_k_gen(k:int, sigma:float):
    'Alternative function generationg i_k(theta)'
    w_k = w_k_gen(k, sigma)
    sigma_k_squared = sigma_k_squared_gen(k, sigma)
    def lambda_k_gen(m, theta):
        return w_k(theta)*sigma_k_squared(theta)**(m/2)/(sigma**2*np.math.factorial(m))
    return lambda theta: (1
                          - 3*lambda_k_gen(4, theta)
                          + 15/2*lambda_k_gen(3, theta)**2)

def laplace_k_gen(k:int, sigma:float):
    'Function representing the kth derivative of the laplace transform, given theta'
    return lambda theta: laplace_a_gen(k, sigma)(theta)*i_k_gen(k, sigma)(theta)

def laplace_gen(theta:float, sigma:float):
    'Funciton generating values of laplace transform, given theta and sigma'
    return laplace_k_gen(0, sigma)(theta)

def kappa_2(theta:float, sigma:float):
    'Alternative function generating doubble derivative of kappa evaluated at theta'
    laplace = laplace_gen(theta, sigma)
    laplace_1 = laplace_k_gen(1, sigma)(theta)
    laplace_2 = laplace_k_gen(2, sigma)(theta)
    return (laplace_2/laplace
            - (laplace_1/laplace)**2)

def kappa_3(theta:float, sigma:float):
    'Function generating tripple derivative of kappa evaluated at theta'    
    laplace = laplace_gen(theta, sigma)
    laplace_1 = laplace_k_gen(1, sigma)(theta)
    laplace_2 = laplace_k_gen(2, sigma)(theta)
    laplace_3 = laplace_k_gen(3, sigma)(theta)
    return -(laplace_3/laplace
             + 2*laplace_1**3/laplace**3
             - 3*laplace_1*laplace_2/laplace**2)

def kappa_4(theta:float, sigma:float):
    'Function generating quadrupple derivative of kappa evaluated at theta'    
    laplace = laplace_gen(theta, sigma)
    laplace_1 = laplace_k_gen(1, sigma)(theta)
    laplace_2 = laplace_k_gen(2, sigma)(theta)
    laplace_3 = laplace_k_gen(3, sigma)(theta)
    laplace_4 = laplace_k_gen(4, sigma)(theta)
    return (laplace_4/laplace
            - 3*laplace_2**2/laplace**2
            - 6*laplace_1**4/laplace**4
            - 4*laplace_3*laplace_1/laplace**2
            + 12*laplace_1**2*laplace_2/laplace**3)

def lambda_n_gen(n:int, theta:float, sigma:float):
    'Funciton generating value of lambda_n'
    return theta*np.sqrt(n*kappa_2(theta, sigma))

def saddlepoint_estimate(x:float, n:int, sigma:float):
    '''
    Function used to estimate left-tail probabilities for sum of n lognormal RVs

    Parameters
    ----------
    x : int
        Value used to decide the point at which we want to estimate the cdf/pdf
    n : int, optional
        Number of lognormal variables in the sum
    sigma : float, optional
        Shape parameter of lognormal distribution when assuming i.i.d lognormal variables in sum

    Returns
    -------
    cdf : float
        Value of cdf of S_n at the point gamma = z*n
    pdf : flaot
        Value of pdf of S_n at the point gamma = z*n
    theta : float
        Estimated value for theta used in calculation
    '''
    theta = theta_aprox(x, sigma)
    # Approximating theta using Newthons method
    laplace_1 = laplace_k_gen(1, sigma)
    for _ in range(4):
        theta = theta - (-laplace_1(theta)/laplace_gen(theta, sigma) + x)/kappa_2(theta, sigma)

    lambda_n = lambda_n_gen(n, theta, sigma)
    kappa_dagger = -(np.log(laplace_gen(theta, sigma)) + x*theta)
    b_0 = lambda_n*np.exp(lambda_n**2/2)*norm.cdf(-lambda_n)
    b_3 = -(lambda_n**3*b_0 - (lambda_n**3 - lambda_n)/(np.sqrt(2*np.pi)))
    b_4 = lambda_n**4*b_0 - (lambda_n**4 - lambda_n**2)/(np.sqrt(2*np.pi))
    b_6 = lambda_n**6*b_0 - (lambda_n**6 - lambda_n**4 + 3*lambda_n**2)/(np.sqrt(2*np.pi))
    zeta_3 = kappa_3(theta, sigma)/kappa_2(theta, sigma)**(3/2)
    zeta_4 = kappa_4(theta, sigma)/kappa_2(theta, sigma)**2

    saddle_pdf = ((2*np.pi*n*kappa_2(theta, sigma))**(-1/2)
                  *np.exp(-n*kappa_dagger)
                  *(1 + 1/n*(zeta_4/8 - 5*zeta_3**2/24)))
    saddle_cdf = 1/lambda_n*np.exp(-n*kappa_dagger)*(b_0
                                                     + zeta_3/(6*np.sqrt(n))*b_3
                                                     + zeta_4/(24*n)*b_4
                                                     + zeta_3**2/(72*n)*b_6)
    return saddle_cdf, saddle_pdf, theta

if __name__ == '__main__':
    # Asmussen et al. example, using gamma = z*N
    N = 16
    SIGMA = 0.125
    xs = [0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.98]
    print('{:^5s}| {:^6s}|{:^12s}|{:^12s}'.format('x', '\u03F4(x)', 'Saddle-cdf', 'Saddle-pdf'))
    for x_outer in xs:
        est_cdf, est_pdf, est_theta = saddlepoint_estimate(x_outer, N, SIGMA)
        print(f'{x_outer:^5.2f}|{est_theta:6.2f} |{est_cdf:^12.3E}|{est_pdf:^12.3E}')
