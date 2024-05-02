'''
Module running and producing plots for numerical experiments in section 4.2-4.4
'''

import json
from ast import literal_eval
from typing import List, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy, chi2, lognorm, nakagami, rice
from convolution_method import convolution_estimate
from saddlepoint_method import saddlepoint_estimate
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 14

def comparison_with_saddlepoint_method() -> None:
    '''
    Here we may compare the raw results using a table, and/or the CPU-time for different N
    '''
    n = 16
    m = int(1e4)
    sigma = 0.125
    xs = [0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.98]
    print(f'Number of random variables: {n}\nNumver of intervals in mesh: {m}\n\u03C3: {sigma}')
    print(f'{"x":^5s}|{"Conv-cdf":^12s}|{"Saddle-cdf":^12s}|{"Conv-pdf":^12s}|{"Saddle-pdf":^12s}')
    for x in xs:
        conv_cdf, conv_pdf = convolution_estimate(x*n, m, n, lambda x: lognorm.pdf(x, sigma))
        sadl_cdf, sadl_pdf, _ = saddlepoint_estimate(x, n, sigma)
        print(f'{x:^5.2f}|{conv_cdf:^12.3E}|{sadl_cdf:^12.3E}|{conv_pdf:^12.3E}|{sadl_pdf:^12.3E}')

def convergence_of_method_lognormal(m_start: int = 2**10,
                                    threshold: float = 1e-8,
                                    m_baseline: int = 2**17,
                                    kinds: List[str] | Tuple[str] = ('trapezoid',
                                                                     'simpson',
                                                                     'boole'), 
                                    ps: List[int] | Tuple[int] = (2, 4, 6)) -> None:
    '''
    Here we test the relative error of each method, relative to a baseline calculated with
    Boole for a large value of m, as a function of m for the lognormal distribution

    Parameters
    ----------
    m_start : int, default = 2**10
        Size of grid for first calculation
    threshold : float, default = 1e-8
        Relative error to reach in order to stop calculations
    m_baseline : int, default = 2**17
        Size of grid used to calculate pseudo-reference solution
    kinds : List[str] | Tuple[str], default = ('trapezoid', 'simpson', 'boole')
        Kinds argument passed to convolutoin_estimate method
    ps : List[int] | Tuple[int], default = (2, 4, 6)
        Powers of reference slope to draw over plot of relative error for the respectiv methods
        passed in kinds argument
    '''
    n = 16
    x = 0.7
    sigma = 0.125
    path = 'convergence_of_convolution_to_alpha'
    try:
        fp = open(f'./data/{path}.json', 'r', encoding='utf-8')
        data = json.load(fp)
        relative_errors = data['err']
        ms = data['ms']
        alpha = data['alpha']
    except FileNotFoundError:
        print(f"{'-'*10} Running experiments using Lognormal distribution {'-'*10}")
        m = int(m_start)
        ms = []
        relative_errors = {key:[] for key in kinds}
        rel_err = threshold + 1
        alpha, _ = convolution_estimate(x*n, int(m_baseline), n, lambda x: lognorm.pdf(x, sigma))
        while rel_err > threshold:
            rel_err = 0
            # pylint: disable=unbalanced-tuple-unpacking
            trap, simp, boole, _ = convolution_estimate(x*n, m, n, lambda x: lognorm.pdf(x, sigma),
                                                        kind='all')
            for kind, alpha_bar in zip(kinds, [trap, simp, boole]):
                rel_error_tmp = np.abs(alpha_bar-alpha)/alpha
                if len(relative_errors[kind]) < 1 or relative_errors[kind][-1] > threshold:
                    relative_errors[kind].append(rel_error_tmp)
                rel_err = np.max([rel_err, rel_error_tmp])
            print(f'm: {m}, error: {rel_err}')
            ms.append(m)
            m *= 2
        data = {'err':relative_errors, 'ms':ms, 'alpha':alpha}
        with open(f'./data/{path}.json',
                  'w',
                  encoding='utf-8') as fp:
            json.dump(data, fp)

    fig, ax = plt.subplots()
    sadl_cdf, _, _ = saddlepoint_estimate(x, n, sigma)
    rel_err_sadl = np.abs(sadl_cdf-alpha)/alpha
    ax.hlines(rel_err_sadl, ms[0], ms[-1], color='k', ls='-.', label='Saddlepoint')
    ax.hlines(threshold, ms[0], ms[-1], color='k', ls='--')
    ax.text(ms[-6], 1.25*threshold, f'Threshold: ${threshold:.0E}'.replace('-0', '^{-') + r'}$')
    for i, (kind, p) in enumerate(zip(kinds, ps)):
        y = np.array(relative_errors[kind])
        x_plot = np.array(ms[:len(y)], dtype=np.double)
        a = y[0]*x_plot[0]**(p)
        # Plot errors
        ax.plot(x_plot, y, '-o', label=kind.capitalize(),  markersize=4)
        # Plot theoretical slope
        ax.plot(x_plot, 1.75*a*x_plot**(-p), 'k:')
        # Text with slope function
        ax.text(x_plot[int(3*len(x_plot)/4)] if i != 2 else x_plot[2],
                2.5*a*x_plot[int(3*len(x_plot)/4)]**(-p) if i != 2 else 1.9*a*x_plot[2]**(-p),
                f'$C_{i+1}N^{{-{p}}}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('N')
    ax.set_ylabel('Error relative to pseudo-reference solution')
    fig.tight_layout()
    fig.savefig(f'./figures/{path}.pdf')

def convergence_of_method_lognormal_varying_sigma(m_start: int = 2**10,
                                                  threshold: float = 1e-8,
                                                  m_baseline: int = 2**17,
                                                  kinds: List[str] = ('trapezoid',
                                                                      'simpson',
                                                                      'boole'), 
                                                  ps: List[int] = (2, 4, 6)) -> None:
    '''
    Here we test the relative error of each method, relative to some baseline calculated with
    Boole for some large value of m, as a function of m when having lognormals with different sigma

    Parameters
    ----------
    m_start : int, default = 2**10
        Size of grid for first calculation
    threshold : float, default = 1e-8
        Relative error to reach in order to stop calculations
    m_baseline : int, default = 2**17
        Size of grid used to calculate pseudo-reference solution
    kinds : List[str] | Tuple[str], default = ('trapezoid', 'simpson', 'boole')
        Kinds argument passed to convolutoin_estimate method
    ps : List[int] | Tuple[int], default = (2, 4, 6)
        Powers of reference slope to draw over plot of relative error for the respectiv methods
        passed in kinds argument
    '''
    n = 16
    x = 0.7
    sigmas = 4*[0.25] + 4*[0.125] + 4*[0.072] + 4*[0.035]
    path = 'convergence_of_convolution_to_alpha_vary_sigma'
    try:
        fp = open(f'./data/{path}.json', 'r', encoding='utf-8')
        data = json.load(fp)
        relative_errors = literal_eval(data['err'])
        ms = data['ms']
        alpha = data['alpha']
    except FileNotFoundError:
        print(f"{'-'*10} Running experiments using Lognormal distribution varying sigma {'-'*10}")
        m = int(m_start)
        ms = []
        relative_errors = {key:[] for key in kinds}
        rel_err = threshold + 1
        mesh = np.linspace(0, x*n, int(m_baseline) + 1)
        pdfs = np.array([lognorm.pdf(mesh, sigma) for sigma in sigmas])
        alpha, _ = convolution_estimate(x*n, int(m_baseline), f=pdfs)
        while rel_err > threshold and m < 17*m_baseline:
            print(f'm: {m}, error: {rel_err}')
            rel_err = 0
            mesh = np.linspace(0, x*n, m + 1)
            pdfs = np.array([lognorm.pdf(mesh, sigma) for sigma in sigmas])
            trap, simp, boole, _ = convolution_estimate(x*n, m, f=pdfs, kind='all') # pylint: disable=unbalanced-tuple-unpacking
            for kind, alpha_bar in zip(kinds, [trap, simp, boole]):
                rel_error_tmp = np.abs(alpha_bar-alpha)/alpha
                if len(relative_errors[kind]) < 1 or relative_errors[kind][-1] > threshold:
                    relative_errors[kind].append(rel_error_tmp)
                rel_err = np.max([rel_err, rel_error_tmp])
            ms.append(m)
            m *= 2
        data = {'err':str(relative_errors), 'ms':ms, 'alpha':alpha}
        with open(f'./data/{path}.json',
                  'w',
                  encoding='utf-8') as fp:
            json.dump(data, fp)

    fig, ax = plt.subplots()
    ax.hlines(threshold, ms[0], ms[-1], color='k', ls='--')
    ax.text(ms[-6], 1.25*threshold, f'Threshold: ${threshold:.0E}'.replace('-0', '^{-') + r'}$')
    for i, (kind, p) in enumerate(zip(kinds, ps)):
        y = np.array(relative_errors[kind])
        x_plot = np.array(ms[:len(y)], dtype=np.double)
        a = y[0]*x_plot[0]**(p)
        # Plot errors
        ax.plot(x_plot, y, '-o', label=kind.capitalize(), markersize=4)
        # Plot theoretical slope
        ax.plot(x_plot, 1.75*a*x_plot**(-p), 'k:')
        # Text with slope function
        ax.text(x_plot[int(len(x_plot)/2)] if i != 2 else x_plot[2],
                2*a*x_plot[int(len(x_plot)/2)]**(-p) if i != 2 else 3*a*x_plot[2]**(-p),
                f'$C_{i+1}N^{{-{p}}}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('N')
    ax.set_ylabel('Error relative to pseudo-reference solution')
    fig.tight_layout()
    fig.savefig(f'./figures/{path}.pdf')

def convergence_to_levy_cdf(m_start: int = 2**7,
                            m_end: float = 2**15) -> None:
    '''
    Check how fast the method converge to the levy distribution

    Parameters
    ----------
    m_start : int, default = 2**7
        Size of grid for first calculation
    m_end : float, default = 2**15
        Size of grid when we stop calculations
    '''
    n = 16
    x = 0.05
    params = [1/2, 1, 2, 4]

    path = 'convergence_to_levy'
    try:
        fp = open(f'./data/{path}.json', 'r', encoding='utf-8')
        data = json.load(fp)
        results = literal_eval(data['err'])
        ms = data['ms']
        alphas = literal_eval(data['alphas'])
        params = data['params']
    except FileNotFoundError:
        print(f"{'-'*10} Running experiments using Levy distribution {'-'*10}")
        alphas = {c:levy.cdf(x*n, scale=(n*np.sqrt(c))**2) for c in params}
        results = {c:[] for c in params}
        m = int(m_start)
        ms = []
        while m < m_end:
            for c in params:
                boole, _ = convolution_estimate(x*n, m, n, lambda x: levy.pdf(x, scale=c)) # pylint: disable=cell-var-from-loop
                rel_err = np.abs(boole-alphas[c])/alphas[c]
                results[c].append(rel_err)
                print(f'param: {c}, m: {m}, error: {rel_err}')
            ms.append(m)
            m = 2*m
        data = {'err':str(results), 'ms':ms, 'alphas':str(alphas), 'params':params}
        with open(f'./data/{path}.json',
                  'w',
                  encoding='utf-8') as fp:
            json.dump(data, fp)

    fig, ax = plt.subplots()
    idxs = [i for i in range(len(ms)) if ms[i]<=m_end and ms[i]>=m_start]
    ms = np.array([ms[idx] for idx in idxs], dtype=np.float64)
    for i, (c, p) in enumerate(zip(params, (0, 0, 0, 6))):
        results[c] = [results[c][idx] for idx in idxs]
        ax.plot(ms,
                results[c],
                '-o',
                label=f'$c={c}, \\alpha={alphas[c]:1.2g}}}$'.replace('-', '^{-'),
                markersize=4)
        if p != 0:
            a = results[params[3]][2]*ms[2]**(p)
            # Plot theoretical slope
            ax.plot(ms, (3*a*ms**(-p)), 'k:')
            # Text with slope function
            ax.text(ms[int(len(ms)/2)],
                    3.75*a*ms[int(len(ms)/2)]**(-p),
                    f'$C_{1}N^{{-{p}}}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(frameon=False, title='Shape parameter')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('N')
    ax.set_ylabel('Relative error')
    fig.tight_layout()
    fig.savefig(f'./figures/{path}.pdf')

def convergence_to_chi_squared_cdf(m_start: int = 2**7,
                                   m_end: float = 2**15) -> None:
    '''
    Check how fast the method converge to the chi squared distribution

    Parameters
    ----------
    m_start : int, default = 2**7
        Size of grid for first calculation
    m_end : float, default = 2**15
        Size of grid when we stop calculations
    '''
    n = 16
    x = 0.05
    params = [2, 3, 4, 6]

    path = 'convergence_to_chi_squared'
    try:
        fp = open(f'./data/{path}.json', 'r', encoding='utf-8')
        data = json.load(fp)
        results = literal_eval(data['err'])
        ms = data['ms']
        alphas = literal_eval(data['alphas'])
        params = data['params']
    except FileNotFoundError:
        print(f"{'-'*10} Running experiments using Chi squared distribution {'-'*10}")
        alphas = {df:chi2.cdf(x*n, df*n) for df in params}
        results = {df:[] for df in params}
        m = int(m_start)
        ms = []
        while m < m_end:
            for df in params:
                boole, _ = convolution_estimate(x*n, m, n, lambda x: chi2.pdf(x, df)) # pylint: disable=cell-var-from-loop
                rel_err = np.abs(boole-alphas[df])/alphas[df]
                results[df].append(rel_err)
                print(f'param: {df}, m: {m}, error: {rel_err}')
            ms.append(m)
            m = 2*m
        data = {'err':str(results), 'ms':ms, 'alphas':str(alphas), 'params':params}
        with open(f'./data/{path}.json',
                  'w',
                  encoding='utf-8') as fp:
            json.dump(data, fp)

    fig, ax = plt.subplots()
    idxs = [i for i in range(len(ms)) if ms[i]<=m_end]
    ms = np.array([ms[idx] for idx in idxs], dtype=np.float64)
    for i, (df, p) in enumerate(zip(params, (2, 1.5, 0, 4))):
        results[df] = [results[df][idx] for idx in idxs]
        ax.plot(ms,
                results[df],
                '-o',
                label=f'$df={df}, \\alpha={alphas[df]:1.2g}}}$'.replace('-','^{-'),
                markersize=4)
        a = results[df][0]*ms[0]**(p)
        if p != 0:
            # Plot theoretical slope
            ax.plot(ms, 1.8*a*ms**(-p) if i != 0 else 0.5*a*ms**(-p), 'k:')
            # Text with slope function
            ax.text(ms[int(3*len(ms)/4)] if i != 2 else ms[2],
                    2.75*a*ms[int(3*len(ms)/4)]**(-p) if i != 0 else 7e-5*a*ms[2]**(-p),
                    f'$C_{i+1}N^{{-{p}}}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(frameon=False, title='Degrees of freedom')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('N')
    ax.set_ylabel('Relative error')
    fig.tight_layout()
    fig.savefig(f'./figures/{path}.pdf')

def convergence_to_custom_cdf(pdf: Callable[[float], Callable[[float], float]],
                              params: List[int|float|Tuple[int|float]],
                              name: str,
                              legend_label: str,
                              legend_title: str,
                              m_baseline: int = 2**20,
                              m_start: int = 2**7,
                              m_end: int = 2**15,
                              x: float = 0.05,
                              ps: List[int] = None,
                              cs: List[float] = None) -> None:
    '''
    Function to check convergence of convolution method to a distibution determined by a 
    supplied pdf

    Parameters
    ----------
    pdf : Callable[[float], Callable[[float], float]]
        Function taking parameter value and returning pdf for distribution
    params : List[int|float|Tuple[int|float]]
        Parameter values used in experiments
    name : str
        Name of distribution used in paths for storing data and figure
    legend_label : str
        String used to denote parameter in figure legend
    legend_title : str
        Titel used in figure legend
    m_baseline : int, default = 2**20
        Size of grid used to calculate pseudo-reference solution
    m_start : int, default = 2**7
        Size of grid for first calculation
    m_end : int, default = 2**15
        Size of grid when we stop calculations
    x : float, default = 0.05
        Value of x used to decide cdf value we calculate in experiment
    ps : List[int] | Tuple[int], default = (2, 4, 6)
        Powers of reference slope to draw over plot of relative error for the respectiv methods
        passed in kinds argument
    cs : List[float]
        List of constants used to decide placement of text in plot stating rate of refrence slopes
    '''
    n = 16

    path = f'convergence_to_{name}'
    try:
        fp = open(f'./data/{path}.json', 'r', encoding='utf-8')
        data = json.load(fp)
        results = literal_eval(data['err'])
        results = {float(key):value for key, value in results.items()}
        ms = data['ms']
        alphas = literal_eval(data['alphas'])
        alphas = {float(key):value for key, value in alphas.items()}
        params_ = literal_eval(data['params'])
        params = [x for x in params_ if x in params]
    except FileNotFoundError:
        print(f"{'-'*10} Running experiments using {name} distribution {'-'*10}")
        alphas = {param: convolution_estimate(x*n, int(m_baseline), n, pdf(param)
                                              )[0] for param in params}
        results = {param:[] for param in params}
        m = int(m_start)
        ms = []
        while m < m_end:
            for param in params:
                boole, _ = convolution_estimate(x*n, m, n, pdf(param)) # pylint: disable=cell-var-from-loop
                rel_err = np.abs(boole-alphas[param])/alphas[param]
                results[param].append(rel_err)
                print(f'param: {param}, m: {m}, error: {rel_err}')
            ms.append(m)
            m = 2*m
        data = {'err':str(results), 'ms':ms, 'alphas':str(alphas), 'params':str(params)}
        with open(f'./data/{path}.json',
                  'w',
                  encoding='utf-8') as fp:
            json.dump(data, fp)

    fig, ax = plt.subplots()
    ms = np.array(ms, dtype=np.double)
    ms = ms[ms<=m_end]
    j = 1
    for i, param in enumerate(params):
        ax.plot(ms,
                results[param][:len(ms)],
                '-o',
                label=f'${legend_label}={param},\\alpha={alphas[param]:1.2g}}}$'.replace('-',
                                                                                         '^{-'),
                markersize=4)
        if ps is not None and ps[i] != 0:
            flag = True
            p = ps[i]
            a = results[param][0]*ms[0]**(p)
            c = cs[i] if cs is not None else 1.8
            y = [c*a*ms[i]**(-p) for i in range(len(ms)) if c*a*ms[i]**(-p) > results[param][i]]
            if len(y) == 0:
                flag = False
                y = [c*a*ms[i]**(-p) for i in range(len(ms)) if c*a*ms[i]**(-p) < results[param][i]]
            # Plot theoretical slope
            ax.plot(ms[:len(y)], y, 'k:')
            # Text with slope function
            ax.text(ms[int(len(y)/2)],
                    (c + 1)*a*ms[int(len(y)/2)]**(-p) if flag else 0.05*c*a*ms[int(len(y)/2)]**(-p),
                    f'$C_{j}N^{{-{p}}}$')
            j += 1
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(frameon=False, title=f'{legend_title}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('N')
    ax.set_ylabel('Error relative to pseudo-reference solution')
    fig.tight_layout()
    fig.savefig(f'./figures/{path}.pdf')

def convergence_to_nakagami_cdf():
    'Find convergence rate of convolution method when estimating the Nakagami distribution'
    ms = [1, 2, 3]
    def pdf(m):
        return lambda x: nakagami.pdf(x, m)
    convergence_to_custom_cdf(pdf, ms, 'Nakagami', 'm', 'Shape parameter',
                              ps=(2, 4, 6), cs=(2, 2, 3))

def convergence_to_rice_cdf():
    'Find convergence rate of convolution method when estimating the Rice distribution'
    nus = [0, 0.5, 1]
    def pdf(nu):
        return lambda x: rice.pdf(x, nu)
    convergence_to_custom_cdf(pdf, nus, 'Rice', '\\nu', '$\\nu$',
                              ps=(2, 0, 0, 0, 0), cs=(1.5, 0, 0, 0, 0))

if __name__ == '__main__':
    # Produce right plot in figure 3
    convergence_to_levy_cdf()
    # Produce left plot in figure 3
    convergence_to_chi_squared_cdf()
    # Produce left plot in figure 5
    convergence_to_nakagami_cdf()
    # Produce right plot in figure 5
    convergence_to_rice_cdf()
    # Produce left plot in figure 4
    convergence_of_method_lognormal()
    # Produce right plot in figure 4
    convergence_of_method_lognormal_varying_sigma()
    # Produce data in table 6
    comparison_with_saddlepoint_method()
