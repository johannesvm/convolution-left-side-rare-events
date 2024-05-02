# convolution-left-side-rare-events
Source code for generating results from the paper A Fast and Accurate Numerical Method for the Left Tail of Sums of Independent Random Variables

The source code to produce figure 4, 5 and 6 are found in numerical_examples_main.py. This module use the modules convolution_method.py and saddlepoint_method.py. The script plot_direct_vs_fft_conv.py contains code to generate figure 1 and 2 based on computations done in MATLAB utilizing the Multiprecision computing toolbox for MATLAB.

The data produced from running the numerical examples are stored in the folder data, while the produced figures are stored in the directory figures. Note that running the script numerical_examples_main.py with the relevant data files present will use the data present in these files and not rerun the computations. In order to run the computations simply remove all .json-files from the data directory. Note that the experiment considering the lognormal distribution with varying sigma is not optimized, and therefore has a quite long run time. 
