Depolarization model for a two-layer system (applied to M51)

A python 2.7 code to visualize the depolarization model by Carl Shneider (Shneider et al. 2014)

##################################################################################################

This script uses Python Version 2.7
This script uses Matplotlib version 1.3.1

If you want to use a higher version of matplotlib, replace the keyword 'axisbg' with 'facecolor'.

##################################################################################################

Maja Kierdorf, Feb. 2020

This scripts models the normalized degree of polarization as a function of wavelength in a two-layer system applied to the nearby spiral galaxy M51. 
Details are discussed in Kierdorf et al. 2020 (in prep.) and in a PhD thesis from Maja Kierdorf (Univ. of Bonn, 2019): http://hss.ulb.uni-bonn.de/2019/5543/5543.htm

The model is based on Shneider et al. 2014: https://ui.adsabs.harvard.edu/abs/2014A%26A...567A..82S/abstract

##################################################################################################

x-axis: Wavelength

y-axis: Normalized degree of polarization (p/p_0) with p_0=0.7

Free parameters:

B_d - regular magnetic field in the disk in microGauss

B_h - regular magnetic field in the halo in microGauss

b_d - turbulent magnetic field in the disk in microGauss

b_h - turbulent magnetic field in the halo in microGauss

ne_d - thermal electron density in the disk in cm^-3

ne_h - thermal electron density in the halo in cm^-3

