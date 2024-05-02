'''
Module to produce plots for numerical example in section 4.1 based on computations done in Matlab
'''
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 14

# --------------------------------------- Left plot figure 1 --------------------------------------
# Read in data
FFT_vs_conv = io.loadmat('./data/pdfFFTvsConvLognormal.mat')
refSol = FFT_vs_conv['refSol'][0]
solFFT_D = FFT_vs_conv['solFFT_D'][0]
solFFT_S = FFT_vs_conv['solFFT_S'][0]
solFFT_MP1 = FFT_vs_conv['solFFT_MP1'][0]
solFFT_MP2 = FFT_vs_conv['solFFT_MP2'][0]
yRef = FFT_vs_conv['yRef'][0]
yFFT_D = FFT_vs_conv['yFFT_D'][0]
yFFT_S = FFT_vs_conv['yFFT_S'][0]
yFFT_MP1 = FFT_vs_conv['yFFT_MP1'][0]
yFFT_MP2 = FFT_vs_conv['yFFT_MP2'][0]
# Remove the few points where the pdf is 0, as theese mess up the plot, and are not relevant
solFFT_D[solFFT_D==0] = np.nan
solFFT_S[solFFT_S==0] = np.nan
solFFT_MP1[solFFT_MP1==0] = np.nan
solFFT_MP2[solFFT_MP2==0] = np.nan
# Plot figure
fig, ax = plt.subplots()
ax.plot(yRef, refSol, label='Direct')
ax.plot(yFFT_MP2, solFFT_MP2, label='FFT 256-bit')
ax.plot(yFFT_MP1, solFFT_MP1, label='FFT 128-bit')
ax.plot(yFFT_D, solFFT_D, label='FFT 64-bit')
ax.plot(yFFT_S, solFFT_S, label='FFT 32-bit')
ax.set_yscale('log')
ax.set_xlim([0, 16])
ax.set_ylim([1e-100, 1.1])
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('y')
ax.set_ylabel('p(y)')
fig.tight_layout()
fig.savefig('./figures/pdfFFTvsConvLognormal.pdf')

# --------------------------------------- Right plot figure 1 -------------------------------------
# Read in data
FFT_vs_conv_runtime = io.loadmat('./data/runtimeConvVsFFT.mat')
N = FFT_vs_conv_runtime['N'][0].astype(int)
timeConv = FFT_vs_conv_runtime['timeConv'][0]
timeFFT_D = FFT_vs_conv_runtime['timeFFT_D'][0]
timeFFT_O = FFT_vs_conv_runtime['timeFFT_O'][0]
timeFFT_Q = FFT_vs_conv_runtime['timeFFT_Q'][0]
# Plot figure
fig, ax = plt.subplots()
ax.plot(N, timeConv, label='Direct')
ax.plot(N, timeFFT_O, label='FFT 256-bit')
ax.plot(N, timeFFT_Q, label='FFT 128-bit')
ax.plot(N, timeFFT_D, label='FFT 64-bit')
ax.plot(N, 10**(-8)*N*np.log2(N), 'k--')
ax.plot(N, 10**(-10.5)*N**2, 'k--')
ax.text(N[-3], 0.3*10**(-8)*N[-3]*np.log2(N[-3]), r'$C_1 N\log(N)$')
ax.text(N[-3], 0.3*10**(-10.5)*N[-3]**2, r'$C_2 N^2$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([2100, 2.1e6])
ax.set_ylim([1e-4, 1e3])
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('N')
ax.set_ylabel('Runtime (s)')
fig.tight_layout()
fig.savefig('./figures/runtimeConvVsFFT.pdf')

# ----------------------------------------- Plot figure 2 -----------------------------------------
# Read in data
FFT_vs_conv = io.loadmat('./data/pdfFFTvsConvLevy.mat')
pdfRef = FFT_vs_conv['pdfRef'][0]
solDir = FFT_vs_conv['solDir'][0]
solFFT_D = FFT_vs_conv['solFFT_D'][0]
solFFT_S = FFT_vs_conv['solFFT_S'][0]
solFFT_MP1 = FFT_vs_conv['solFFT_MP1'][0]
solFFT_MP2 = FFT_vs_conv['solFFT_MP2'][0]
yRef = FFT_vs_conv['yRef'][0]
yDir = FFT_vs_conv['yDir'][0]
yFFT_D = FFT_vs_conv['yFFT_D'][0]
yFFT_S = FFT_vs_conv['yFFT_S'][0]
yFFT_MP1 = FFT_vs_conv['yFFT_MP1'][0]
yFFT_MP2 = FFT_vs_conv['yFFT_MP2'][0]
# Remove the few points where the pdf is 0, as theese mess up the plot, and are not relevant
solFFT_D[solFFT_D==0] = np.nan
solFFT_S[solFFT_S==0] = np.nan
solFFT_MP1[solFFT_MP1==0] = np.nan
solFFT_MP2[solFFT_MP2==0] = np.nan
# Plot figure
fig, ax = plt.subplots()
ax.plot(yRef, pdfRef, '--k', lw=3, label='Exact')
ax.plot(yDir, solDir, label='Direct')
ax.plot(yFFT_MP2, solFFT_MP2, label='FFT 256-bit')
ax.plot(yFFT_MP1, solFFT_MP1, label='FFT 128-bit')
ax.plot(yFFT_D, solFFT_D, label='FFT 64-bit')
ax.plot(yFFT_S, solFFT_S, label='FFT 32-bit')
ax.set_yscale('log')
ax.set_xlim([0, 1])
ax.set_ylim([1e-150, 1.1])
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('y')
ax.set_ylabel('p(y)')
fig.tight_layout()
fig.savefig('./figures/pdfFFTvsConvLevy.pdf')
