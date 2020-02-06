Interactive tool for a depolarization model of a two-layer system (applied to the spiral galaxy M51).

A python 2.7 code to visualize the depolarization model by Carl Shneider (Shneider et al. 2014) and to change model parameters interactively. 

#################################################################################

The script uses Python version 2.7 and Matplotlib version 1.3.1

If you want to use a higher version of matplotlib (2.0+), replace the keyword 'axisbg' with 'facecolor'.

#################################################################################

Written by Maja Kierdorf, Feb. 2020

This script models the normalized degree of polarization as a function of wavelength in a two-layer system applied to the nearby spiral galaxy M51 and allows to interactively change some model parameters. 
Details are discussed in Kierdorf et al. (in prep.) and in a PhD thesis from Maja Kierdorf (Univ. of Bonn, 2019): http://hss.ulb.uni-bonn.de/2019/5543/5543.htm

The model is based on the work of Shneider et al. 2014: https://ui.adsabs.harvard.edu/abs/2014A%26A...567A..82S/abstract

#################################################################################

x-axis: Wavelength

y-axis: Normalized degree of polarization (p/p_0) with p_0=0.7

Changeable parameters:

B_d - regular magnetic field in the disk in microGauss

B_h - regular magnetic field in the halo in microGauss

b_d - turbulent magnetic field in the disk in microGauss

b_h - turbulent magnetic field in the halo in microGauss

ne_d - thermal electron density in the disk in cm^-3

ne_h - thermal electron density in the halo in cm^-3

MODELS 

D : Disk regular field

H : Halo regular field

A: Anisotropic turbulent fields

I: Isotropic turbulent fields

Example: DAHI means regular fields in Disk and Halo (DH), Anisotropic turbulent field in Disk (A), and Isotropic turbulent field in Halo (I)

#################################################################################

The 'Depoltool_Example_Data.pdf' file shows an example with datapoints showing the observed degree of polarization between ~ 3 and 27 cm (Kierdorf et al., in prep.).
