#!/usr/bin/env python

###################################################################################
# Maja Kierdorf, May. 2020
#
# This scripts creates an interactive tool to model the normalized degree 
# of polarization as a function of wavelength in a three-layer system applied to the
# nearby spiral galaxy M51. 
# Details are discussed in Kierdorf et al. 2020 (in prep.) and
# in a PhD thesis from Maja Kierdorf (Univ. of Bonn, 2019):
# http://hss.ulb.uni-bonn.de/2019/5543/5543.htm
# The model is based on Shneider et al. 2014:
# https://ui.adsabs.harvard.edu/abs/2014A%26A...567A..82S/abstract
# 
###################################################################################
#
# This script uses Python Version 2.7
# This script uses Matplotlib version 1.3.1
#
###################################################################################


import numpy as np
import pylab as pl
import sys
import os
from scipy.optimize import curve_fit
from matplotlib.widgets import Slider, Button, RadioButtons


lambda2_list = np.linspace(1e-8, 0.16000001,8000)

###########################################################################
########################## Define models for plot #########################
###########################################################################

phi = np.radians(100.) # Azimuthal angle in degrees
l = np.radians(-20.) # Inclination in degrees 
Ld = 800. # Path length through the disk in pc
Lh = 5000. # Path length through the halo in pc
ne_d = 0.11 # Electron density in the disk in cm^-3
ne_h = 0.01 # Electron density in the halo in cm^-3
D = 600. # Linear Diameter of the telescope beam in pc (from Fletcher+2011)
sigmaRM_D = 15. # Observed value of RM dispersion from Fletcher+2011 in rad / m^2

Btot_reg_d = 5. # Regular MF in the disk in microGauss
Btot_reg_h = 5. # Regular MF in the disk in microGauss

btot_ran_d = 14.0 # Turbulent MF in the disk in microGauss
btot_ran_h = 4.0 # Turbulent MF in the halo in microGauss

############################################################
##### Parameters from Fletcher et al. (2011), Table A1 #####
############################################################

p0 = np.radians(-20.) # pitch angle of the total horizontal MF with mode m=0 in the disk
p2 = np.radians(-12.) # pitch angle of the total horizontal MF with mode m=2 in the disk
ph0 = np.radians(-43.) # pitch angle of the total horizontal MF with mode m=0 in the halo
ph1 = np.radians(-45.) # pitch angle of the total horizontal MF with mode m=1 in the halo
beta2 = np.radians(-8.) # angle in azimuth at which the corresponding m!=0 mode is maximum
betah1 = np.radians(44.) # angle in azimuth at which the corresponding m!=0 mode is maximum

R2 = -33.
R0 = -46.
Rh0 = 23.
Rh1 = 76.


B0 = Btot_reg_d/np.sqrt(1. + (R2/R0)**2. * np.cos(2. * phi - beta2)**2. + 2. * (R2/R0) * np.cos(2. * phi - beta2) * np.cos(p0-p2))
B2 = Btot_reg_d/np.sqrt((R0/R2)**2. +  np.cos(2. * phi - beta2)**2. + 2. * (R0/R2) * np.cos(2. * phi - beta2) * np.cos(p0-p2))
Bh0 = Btot_reg_h/np.sqrt(1. + (Rh1/Rh0)**2. * np.cos(phi - betah1)**2. + 2. * (Rh1/Rh0) * np.cos(phi - betah1) * np.cos(ph0-ph1))
Bh1 = Btot_reg_h/np.sqrt((Rh0/Rh1)**2. +  np.cos(phi - betah1)**2. + 2. * (Rh0/Rh1) * np.cos(phi - betah1) * np.cos(ph0-ph1)) 

Br =  B0 * np.sin(p0) + B2 * np.sin(p2) * np.cos(2.*phi - beta2)
Bphi =  B0 * np.cos(p0) + B2 * np.cos(p2) * np.cos(2.*phi - beta2)
Bz = 0.
Bhr = Bh0 * np.sin(ph0) + Bh1 * np.sin(ph1) * np.cos(phi - betah1)
Bhphi = Bh0 * np.cos(ph0) + Bh1 * np.cos(ph1) * np.cos(phi - betah1)
Bhz = 0.

mean_Bx_d = Br * np.cos(phi) - Bphi * np.sin(phi)
mean_By_d = (Br * np.sin(phi) + Bphi * np.cos(phi))*np.cos(l) + Bz * np.sin(l)
mean_Bpara_d = -(Br * np.sin(phi) + Bphi * np.cos(phi))*np.sin(l) + Bz * np.cos(l)
mean_Bx_h = Bhr * np.cos(phi) - Bhphi * np.sin(phi)
mean_By_h = (Bhr * np.sin(phi) + Bhphi * np.cos(phi))*np.cos(l) + Bhz * np.sin(l)
mean_Bpara_h = -(Bhr * np.sin(phi) + Bhphi * np.cos(phi))*np.sin(l) + Bhz * np.cos(l)

mean_Bperp_2_d = mean_Bx_d**2. + mean_By_d**2.
mean_Bperp_2_h = mean_Bx_h**2. + mean_By_h**2.

################# Random part (Anisotropy only) ##############################

alpha_d_A = 2.0 # constant factor for anisotropy
alpha_h_A = 1.5 # constant factor for anisotropy

sigmar2_d_A = btot_ran_d**2. / (2. + alpha_d_A) 
sigmar2_h_A = btot_ran_h**2. / (2. + alpha_h_A)

sigmax2_d_A =  sigmar2_d_A * (np.cos(phi)**2. + alpha_d_A * np.sin(phi)**2.)
sigmay2_d_A = sigmar2_d_A * ((np.sin(phi)**2. + alpha_d_A * np.cos(phi)**2.)*np.cos(l)**2. + np.sin(l)**2.)
sigmax2_h_A =  sigmar2_h_A * (np.cos(phi)**2. + alpha_h_A * np.sin(phi)**2.)
sigmay2_h_A = sigmar2_h_A * ((np.sin(phi)**2. + alpha_h_A * np.cos(phi)**2.)*np.cos(l)**2. + np.sin(l)**2.)

bpara2_d_A = sigmar2_d_A * ((np.sin(phi)**2. + alpha_d_A * np.cos(phi)**2.)*np.sin(l)**2. + np.cos(l)**2.) # 
bpara2_h_A = sigmar2_h_A * ((np.sin(phi)**2. + alpha_h_A * np.cos(phi)**2.)*np.sin(l)**2. + np.cos(l)**2.) # 

d_d_A = ((D * sigmaRM_D)/(0.81 * ne_d * np.sqrt(bpara2_d_A) * np.sqrt(Ld)))**(2./3.)
d_h_A = ((D * sigmaRM_D)/(0.81 * ne_h * np.sqrt(bpara2_h_A) * np.sqrt(Lh)))**(2./3.)
#d_d_A = ((D * sigmaRM_D)/(0.81 * ne_d * btot_ran_d * np.sqrt(Ld)))**(2./3.)
#d_h_A = ((D * sigmaRM_D)/(0.81 * ne_h * btot_ran_h * np.sqrt(Lh)))**(2./3.)

################# Random part (Isotropy only) ##############################

alpha_d_I = 1.0 # constant factor for anisotropy
alpha_h_I = 1.0 # constant factor for anisotropy

sigmar2_d_I = btot_ran_d**2. / (2. + alpha_d_I) 
sigmar2_h_I = btot_ran_h**2. / (2. + alpha_h_I)

sigmax2_d_I =  sigmar2_d_I * (np.cos(phi)**2. + alpha_d_I * np.sin(phi)**2.)
sigmay2_d_I = sigmar2_d_I * ((np.sin(phi)**2. + alpha_d_I * np.cos(phi)**2.)*np.cos(l)**2. + np.sin(l)**2.)
sigmax2_h_I =  sigmar2_h_I * (np.cos(phi)**2. + alpha_h_I * np.sin(phi)**2.)
sigmay2_h_I = sigmar2_h_I * ((np.sin(phi)**2. + alpha_h_I * np.cos(phi)**2.)*np.cos(l)**2. + np.sin(l)**2.)

bpara2_d_I = sigmar2_d_I * ((np.sin(phi)**2. + alpha_d_I * np.cos(phi)**2.)*np.sin(l)**2. + np.cos(l)**2.) # 
bpara2_h_I = sigmar2_h_I * ((np.sin(phi)**2. + alpha_h_I * np.cos(phi)**2.)*np.sin(l)**2. + np.cos(l)**2.) # 

d_d_I = ((D * sigmaRM_D)/(0.81 * ne_d * np.sqrt(bpara2_d_I) * np.sqrt(Ld)))**(2./3.)
d_h_I = ((D * sigmaRM_D)/(0.81 * ne_h * np.sqrt(bpara2_h_I) * np.sqrt(Lh)))**(2./3.)
#d_d_I = ((D * sigmaRM_D)/(0.81 * ne_d * btot_ran_d * np.sqrt(Ld)))**(2./3.)
#d_h_I = ((D * sigmaRM_D)/(0.81 * ne_h * btot_ran_h * np.sqrt(Lh)))**(2./3.)


##############################################################################


##############################################################################

epsilon_d_DFR = 0.1 * (mean_Bperp_2_d) #add turbulent fields here if present! The factor 0.1 and 0.01 in the disk and halo are not significant since its cancels out wehn calculating p/p0
epsilon_h_DFR = 0.01 * (mean_Bperp_2_h) #add turbulent fields here if present!
epsilon_d_IFD = 0.1 * (mean_Bperp_2_d + sigmax2_d_A + sigmay2_d_A) #add turbulent fields here if present!
epsilon_h_IFD = 0.01 * (mean_Bperp_2_h + sigmax2_h_A + sigmay2_h_A) #add turbulent fields here if present!

Id_DFR = epsilon_d_DFR * Ld
Ih_DFR = epsilon_h_DFR * Lh
Id_IFD = epsilon_d_IFD * Ld
Ih_IFD = epsilon_h_IFD * Lh

R_d = 0.81 * ne_d * mean_Bpara_d * Ld
R_h = 0.81 * ne_h * mean_Bpara_h * Lh
#sigmaRM_d_A = 0.81 * ne_d * btot_ran_d * np.sqrt(Ld * d_d_A)
#sigmaRM_h_A = 0.81 * ne_h * btot_ran_h * np.sqrt(Lh * d_h_A)
sigmaRM_d_A = 0.81 * ne_d * np.sqrt(bpara2_d_A) * np.sqrt(Ld * d_d_A)
sigmaRM_h_A = 0.81 * ne_h * np.sqrt(bpara2_h_A) * np.sqrt(Lh * d_h_A)
sigmaRM_d_I = 0.81 * ne_d * np.sqrt(bpara2_d_I) * np.sqrt(Ld * d_d_I)
sigmaRM_h_I = 0.81 * ne_h * np.sqrt(bpara2_h_I) * np.sqrt(Lh * d_h_I)


Omega_d_A = 2. * sigmaRM_d_A**2. * lambda2_list**2
Omega_h_A = 2. * sigmaRM_h_A**2. * lambda2_list**2
Omega_d_I = 2. * sigmaRM_d_I**2. * lambda2_list**2
Omega_h_I = 2. * sigmaRM_h_I**2. * lambda2_list**2

Omega_d = 0.   #for purley regular field only in the disk
Omega_h = 0.   #for purley regular field only in the halo

C_d = 2. * R_d * lambda2_list
C_h = 2. * R_h * lambda2_list

F_DFR = C_d * C_h
G_DFR = 0.
F_IFD_A = Omega_d_A * Omega_h_A + C_d * C_h
G_IFD_A = Omega_h_A * C_d - Omega_d_A * C_h
G_DdA_H = - Omega_d_A * C_h
F_IFD_I = Omega_d_I * Omega_h_I + C_d * C_h
G_IFD_I = Omega_h_I * C_d - Omega_d_I * C_h
F_DdAI_HhI = Omega_d_A * Omega_h_I + C_d * C_h
G_DdAI_HhI = Omega_h_I * C_d - Omega_d_A * C_h

Psi0d_DFR = np.pi/2. - np.arctan(np.cos(l)*np.tan(phi)) + np.arctan2((mean_By_d),(mean_Bx_d))
Psi0h_DFR = np.pi/2. - np.arctan(np.cos(l)*np.tan(phi)) + np.arctan2((mean_By_h),(mean_Bx_h))

Psi0d_IFD = np.pi/2. - np.arctan(np.cos(l)*np.tan(phi)) + 0.5 * np.arctan2((2. * mean_Bx_d * mean_By_d),(mean_Bx_d**2. - mean_By_d**2. + sigmax2_d_A - sigmay2_d_A))
Psi0h_IFD = np.pi/2. - np.arctan(np.cos(l)*np.tan(phi)) + 0.5 * np.arctan2((2. * mean_Bx_h * mean_By_h),(mean_Bx_h**2. - mean_By_h**2. + sigmax2_h_A - sigmay2_h_A))


DeltaPsi_dh_DFR = Psi0d_DFR - Psi0h_DFR

DeltaPsi_dh_IFD = Psi0d_IFD - Psi0h_IFD

DeltaPsi_dh = Psi0d_IFD - Psi0h_DFR #for regular+turbulent in Disk but purley regular in halo

Ad_DFR = (Id_DFR/(Id_DFR+2.*Ih_DFR)) * np.abs((np.sinc(R_d * lambda2_list / np.pi)))
Ah_DFR = (Ih_DFR/(Id_DFR+2.*Ih_DFR)) * np.abs((np.sinc(R_h * lambda2_list / np.pi)))

Ad_IFD = (Id_IFD/(Id_IFD+2.*Ih_IFD)) * (np.sinh(sigmaRM_d_A**2. * lambda2_list**2.)/(sigmaRM_d_A**2. * lambda2_list**2.)) * np.exp(-sigmaRM_d_A**2. * lambda2_list**2.)
Ah_IFD = (Ih_IFD/(Id_IFD+2.*Ih_IFD)) * (np.sinh(sigmaRM_h_A**2. * lambda2_list**2.)/(sigmaRM_h_A**2. * lambda2_list**2.)) * np.exp(-sigmaRM_h_A**2. * lambda2_list**2.)

############### WAVELENGTH-INDEPENDENT DEPOLARIZATION ####################

W_d = 1.0 #no wavelength-dep depolarization for purley regular fields
W_h = 1.0 #no wavelength-dep depolarization for purley regular fields

W_d_AI = (mean_Bperp_2_d/(mean_Bperp_2_d + 2. * sigmax2_d_I)) * (np.sqrt((mean_Bx_d**2. - mean_By_d**2. + sigmax2_d_A - sigmay2_d_A)**2. + 4. * mean_Bx_d**2. * mean_By_d**2.)/(mean_Bperp_2_d + sigmax2_d_A + sigmay2_d_A)) #Aniotropic + Isotropic in Disk

W_h_AI = (mean_Bperp_2_h/(mean_Bperp_2_h + 2. * sigmax2_h_I)) * (np.sqrt((mean_Bx_h**2. - mean_By_h**2. + sigmax2_h_A - sigmay2_h_A)**2. + 4. * mean_Bx_h**2. * mean_By_h**2.)/(mean_Bperp_2_h + sigmax2_h_A + sigmay2_h_A)) #Aniotropic + Isotropic in Halo

W_d_A = (np.sqrt((mean_Bx_d**2. - mean_By_d**2. + sigmax2_d_A - sigmay2_d_A)**2. + 4. * mean_Bx_d**2. * mean_By_d**2.)/(mean_Bperp_2_d + sigmax2_d_A + sigmay2_d_A)) #only Anisotropic
W_h_A = (np.sqrt((mean_Bx_h**2. - mean_By_h**2. + sigmax2_h_A - sigmay2_h_A)**2. + 4. * mean_Bx_h**2. * mean_By_h**2.)/(mean_Bperp_2_h + sigmax2_h_A + sigmay2_h_A)) #only Anisotropic

W_d_I = (mean_Bperp_2_d/(mean_Bperp_2_d + 2. * sigmax2_d_I)) #only Isotropic
W_h_I = (mean_Bperp_2_h/(mean_Bperp_2_h + 2. * sigmax2_h_I)) #only Isotropic


######### MODELS ##########################################################
######### D : Disk regular ################################################
######### H : Halo regular ################################################
######### dI : Disk Isotropic random ######################################
######### hI : Halo Isotropic random ###################################### 
######### dA : Disk Anisotropic random ####################################
######### hA : Halo Anisotropic random ####################################
######### Example: DdA_HhI means regular fields in Disk and Halo (DH), ####
######### Anisotropic random field in Disk (dA), ##########################
######### and Isotropic random field in Halo (hI) #########################
###########################################################################

H_D_H = np.sqrt(2. * Ah_DFR**2. * (1. + np.cos(2. * (R_d + R_h) * lambda2_list)) + Ad_DFR**2. + 2. * Ad_DFR * Ah_DFR * (np.cos(-2. * DeltaPsi_dh_DFR + (R_d + R_h) * lambda2_list) + np.cos(2. * DeltaPsi_dh_DFR + (R_d + R_h) * lambda2_list))) #ONLY regular magnetic fields in D and H and far side H

hI_dI_hI = 2. * W_h_I * Ah_IFD + W_d_I * Ad_IFD #ONLY iotropic random MFs in d and h and h2
hA_dA_hA = 2. * W_h_A * Ah_IFD + W_d_A * Ad_IFD #ONLY aniotropic random MFs in d and h and h2

HDAH = np.sqrt(2. * W_h**2. * (Ih_DFR/(2.*Ih_DFR+Id_IFD))**2. * (((1. -2. * np.exp(-Omega_h) * np.cos(C_h) + np.exp(-2. * Omega_h)) * (1. + np.cos(C_d + C_h))) / (Omega_h**2. + C_h**2.)) + W_d_A**2. * (Id_IFD/(2.*Ih_DFR+Id_IFD))**2. * ((1. - 2. * np.exp(-Omega_d_A) * np.cos(C_d) + np.exp(-2. * Omega_d_A)) / (Omega_d_A**2. + C_d**2.)) + W_d_A * W_h * (Id_IFD * Ih_DFR/(Id_IFD+2.*Ih_DFR)**2.) * (2./(F_DFR**2. + G_DdA_H**2.)) * ((F_DFR * np.cos(-2. * DeltaPsi_dh + C_d) + G_DdA_H * np.sin(-2. * DeltaPsi_dh + C_d)) + (F_DFR * np.cos(2. * DeltaPsi_dh + C_h) - G_DdA_H * np.sin(2. * DeltaPsi_dh + C_h)) + np.exp(-(Omega_d_A + Omega_h)) * ((F_DFR * np.cos(2. * DeltaPsi_dh + C_d) - G_DdA_H * np.sin(2. * DeltaPsi_dh + C_d)) + (F_DFR * np.cos(-2. * DeltaPsi_dh + C_h) + G_DdA_H * np.sin(-2. * DeltaPsi_dh + C_h))) - np.exp(-Omega_d_A) * ((F_DFR * np.cos(2. * DeltaPsi_dh + C_d + C_h) - G_DdA_H * np.sin(2. * DeltaPsi_dh + C_d + C_h)) + (F_DFR * np.cos(-2. * DeltaPsi_dh) + G_DdA_H * np.sin(-2. * DeltaPsi_dh))) - np.exp(-Omega_h) * ((F_DFR * np.cos(-2. * DeltaPsi_dh + C_d + C_h) + G_DdA_H * np.sin(-2. * DeltaPsi_dh + C_d + C_h)) + (F_DFR * np.cos(2. * DeltaPsi_dh) - G_DdA_H * np.sin(2. * DeltaPsi_dh)))))


HDIH = np.sqrt(2. * W_h**2. * (Ih_DFR/(2.*Ih_DFR+Id_IFD))**2. * (((1. -2. * np.exp(-Omega_h) * np.cos(C_h) + np.exp(-2. * Omega_h)) * (1. + np.cos(C_d + C_h))) / (Omega_h**2. + C_h**2.)) + W_d_I**2. * (Id_IFD/(2.*Ih_DFR+Id_IFD))**2. * ((1. - 2. * np.exp(-Omega_d_I) * np.cos(C_d) + np.exp(-2. * Omega_d_I)) / (Omega_d_I**2. + C_d**2.)) + W_d_I * W_h * (Id_IFD * Ih_DFR/(Id_IFD+2.*Ih_DFR)**2.) * (2./(F_DFR**2. + G_DdA_H**2.)) * ((F_DFR * np.cos(-2. * DeltaPsi_dh + C_d) + G_DdA_H * np.sin(-2. * DeltaPsi_dh + C_d)) + (F_DFR * np.cos(2. * DeltaPsi_dh + C_h) - G_DdA_H * np.sin(2. * DeltaPsi_dh + C_h)) + np.exp(-(Omega_d_I + Omega_h)) * ((F_DFR * np.cos(2. * DeltaPsi_dh + C_d) - G_DdA_H * np.sin(2. * DeltaPsi_dh + C_d)) + (F_DFR * np.cos(-2. * DeltaPsi_dh + C_h) + G_DdA_H * np.sin(-2. * DeltaPsi_dh + C_h))) - np.exp(-Omega_d_I) * ((F_DFR * np.cos(2. * DeltaPsi_dh + C_d + C_h) - G_DdA_H * np.sin(2. * DeltaPsi_dh + C_d + C_h)) + (F_DFR * np.cos(-2. * DeltaPsi_dh) + G_DdA_H * np.sin(-2. * DeltaPsi_dh))) - np.exp(-Omega_h) * ((F_DFR * np.cos(-2. * DeltaPsi_dh + C_d + C_h) + G_DdA_H * np.sin(-2. * DeltaPsi_dh + C_d + C_h)) + (F_DFR * np.cos(2. * DeltaPsi_dh) - G_DdA_H * np.sin(2. * DeltaPsi_dh)))))



HIDIHI = np.sqrt(2. * W_h_I**2. * (Ih_IFD/(2.*Ih_IFD+Id_IFD))**2. * (((1. -2. * np.exp(-Omega_h_I) * np.cos(C_h) + np.exp(-2. * Omega_h_I)) * (1. + np.cos(C_d + C_h))) / (Omega_h_I**2. + C_h**2.)) + W_d_I**2. * (Id_IFD/(2.*Ih_IFD+Id_IFD))**2. * ((1. - 2. * np.exp(-Omega_d_I) * np.cos(C_d) + np.exp(-2. * Omega_d_I)) / (Omega_d_I**2. + C_d**2.)) + W_d_I * W_h_I * (Id_IFD * Ih_IFD/(Id_IFD+2.*Ih_IFD)**2.) * (2./(F_IFD_I**2. + G_IFD_I**2.)) * ((F_IFD_I * np.cos(-2. * DeltaPsi_dh_IFD + C_d) + G_IFD_I * np.sin(-2. * DeltaPsi_dh_IFD + C_d)) + (F_IFD_I * np.cos(2. * DeltaPsi_dh_IFD + C_h) - G_IFD_I * np.sin(2. * DeltaPsi_dh_IFD + C_h)) + np.exp(-(Omega_d_I + Omega_h_I)) * ((F_IFD_I * np.cos(2. * DeltaPsi_dh_IFD + C_d) - G_IFD_I * np.sin(2. * DeltaPsi_dh_IFD + C_d)) + (F_IFD_I * np.cos(-2. * DeltaPsi_dh_IFD + C_h) + G_IFD_I * np.sin(-2. * DeltaPsi_dh_IFD + C_h))) - np.exp(-Omega_d_I) * ((F_IFD_I * np.cos(2. * DeltaPsi_dh_IFD + C_d + C_h) - G_IFD_I * np.sin(2. * DeltaPsi_dh_IFD + C_d + C_h)) + (F_IFD_I * np.cos(-2. * DeltaPsi_dh_IFD) + G_IFD_I * np.sin(-2. * DeltaPsi_dh_IFD))) - np.exp(-Omega_h_I) * ((F_IFD_I * np.cos(-2. * DeltaPsi_dh_IFD + C_d + C_h) + G_IFD_I * np.sin(-2. * DeltaPsi_dh_IFD + C_d + C_h)) + (F_IFD_I * np.cos(2. * DeltaPsi_dh_IFD) - G_IFD_I * np.sin(2. * DeltaPsi_dh_IFD)))))


#HIDIHI = np.sqrt(W_d_I**2. * (Id_IFD/(Id_IFD+2.*Ih_IFD))**2. * ((1. - 2. * np.exp(-Omega_d_I) * np.cos(C_d) + np.exp(-2. * Omega_d_I))/(Omega_d_I**2. + C_d**2.)) + W_h_I**2. * (Ih_IFD/(Id_IFD+2.*Ih_IFD))**2. * ((1. - 2. * np.exp(-Omega_h_I) * np.cos(C_h) + np.exp(-2. * Omega_h_I))/(Omega_h_I**2. + C_h**2.)) + W_d_I * W_h_I * (Id_IFD * Ih_IFD/(Id_IFD+2.*Ih_IFD)**2.) * (2./(F_IFD_I**2. + G_IFD_I**2.)) * ((F_IFD_I * np.cos(2. * DeltaPsi_dh_IFD + C_h) - G_IFD_I * np.sin(2. * DeltaPsi_dh_IFD + C_h)) + np.exp(-(Omega_d_I + Omega_h_I)) * (F_IFD_I * np.cos(2. * DeltaPsi_dh_IFD + C_d) - G_IFD_I * np.sin(2. * DeltaPsi_dh_IFD + C_d)) - np.exp(-Omega_d_I) * (F_IFD_I * np.cos(2. * DeltaPsi_dh_IFD + C_d + C_h) - G_IFD_I * np.sin(2. * DeltaPsi_dh_IFD + C_d + C_h)) - np.exp(-Omega_h_I) * (F_IFD_I * np.cos(2. * DeltaPsi_dh_IFD) - G_IFD_I * np.sin(2. * DeltaPsi_dh_IFD)))) #Regular in D and H and isotropic in d and h 


HIDAIHI = np.sqrt(2. * W_h_I**2. * (Ih_IFD/(2.*Ih_IFD+Id_IFD))**2. * (((1. -2. * np.exp(-Omega_h_I) * np.cos(C_h) + np.exp(-2. * Omega_h_I)) * (1. + np.cos(C_d + C_h))) / (Omega_h_I**2. + C_h**2.)) + W_d_AI**2. * (Id_IFD/(2.*Ih_IFD+Id_IFD))**2. * ((1. - 2. * np.exp(-Omega_d_A) * np.cos(C_d) + np.exp(-2. * Omega_d_A)) / (Omega_d_A**2. + C_d**2.)) + W_d_AI * W_h_I * (Id_IFD * Ih_IFD/(Id_IFD+2.*Ih_IFD)**2.) * (2./(F_DdAI_HhI**2. + G_DdAI_HhI**2.)) * ((F_DdAI_HhI * np.cos(-2. * DeltaPsi_dh_IFD + C_d) + G_DdAI_HhI * np.sin(-2. * DeltaPsi_dh_IFD + C_d)) + (F_DdAI_HhI * np.cos(2. * DeltaPsi_dh_IFD + C_h) - G_DdAI_HhI * np.sin(2. * DeltaPsi_dh_IFD + C_h)) + np.exp(-(Omega_d_A + Omega_h_I)) * ((F_DdAI_HhI * np.cos(2. * DeltaPsi_dh_IFD + C_d) - G_DdAI_HhI * np.sin(2. * DeltaPsi_dh_IFD + C_d)) + (F_DdAI_HhI * np.cos(-2. * DeltaPsi_dh_IFD + C_h) + G_DdAI_HhI * np.sin(-2. * DeltaPsi_dh_IFD + C_h))) - np.exp(-Omega_d_A) * ((F_DdAI_HhI * np.cos(2. * DeltaPsi_dh_IFD + C_d + C_h) - G_DdAI_HhI * np.sin(2. * DeltaPsi_dh_IFD + C_d + C_h)) + (F_DdAI_HhI * np.cos(-2. * DeltaPsi_dh_IFD) + G_DdAI_HhI * np.sin(-2. * DeltaPsi_dh_IFD))) - np.exp(-Omega_h_I) * ((F_DdAI_HhI * np.cos(-2. * DeltaPsi_dh_IFD + C_d + C_h) + G_DdAI_HhI * np.sin(-2. * DeltaPsi_dh_IFD + C_d + C_h)) + (F_DdAI_HhI * np.cos(2. * DeltaPsi_dh_IFD) - G_DdAI_HhI * np.sin(2. * DeltaPsi_dh_IFD)))))


#HIDAIHI = np.sqrt(W_d_AI**2. * (Id_IFD/(Id_IFD+2.*Ih_IFD))**2. * ((1. - 2. * np.exp(-Omega_d_A) * np.cos(C_d) + np.exp(-2. * Omega_d_A))/(Omega_d_A**2. + C_d**2.)) + W_h_I**2. * (Ih_IFD/(Id_IFD+2.*Ih_IFD))**2. * ((1. - 2. * np.exp(-Omega_h_I) * np.cos(C_h) + np.exp(-2. * Omega_h_I))/(Omega_h_I**2. + C_h**2.)) + W_d_AI * W_h_I * (Id_IFD * Ih_IFD/(Id_IFD+Ih_IFD)**2.) * (2./(F_DdAI_HhI**2. + G_DdAI_HhI**2.)) * ((F_DdAI_HhI * np.cos(2. * DeltaPsi_dh_IFD + C_h) - G_DdAI_HhI * np.sin(2. * DeltaPsi_dh_IFD + C_h)) + np.exp(-(Omega_d_A + Omega_h_I)) * (F_DdAI_HhI * np.cos(2. * DeltaPsi_dh_IFD + C_d) - G_DdAI_HhI * np.sin(2. * DeltaPsi_dh_IFD + C_d)) - np.exp(-Omega_d_A) * (F_DdAI_HhI * np.cos(2. * DeltaPsi_dh_IFD + C_d + C_h) - G_DdAI_HhI * np.sin(2. * DeltaPsi_dh_IFD + C_d + C_h)) - np.exp(-Omega_h_I) * (F_DdAI_HhI * np.cos(2. * DeltaPsi_dh_IFD) - G_DdAI_HhI * np.sin(2. * DeltaPsi_dh_IFD)))) #Regular in Disk and Halo and A+I in Disk and I in Halo


###########################################################################
########################## p plot data ####################################
########################## 2 layer     ####################################
###########################################################################

   
fig, ax = pl.subplots()
pl.subplots_adjust(left=None, bottom=None, right=0.7, top=None)

   
HDHplot, = pl.plot(np.sqrt(lambda2_list)*100, H_D_H, linestyle='-', lw=2, color='k', label=r'5 $\mu$G DFR 2-layer')
HDAHplot, = pl.plot(np.sqrt(lambda2_list)*100, HDAH, linestyle='--', lw=3, color='r', label=r'5 $\mu$G DFR 2-layer')
HDIHplot, = pl.plot(np.sqrt(lambda2_list)*100, HDIH, linestyle='-', lw=3, color='r', label=r'5 $\mu$G DFR 2-layer')
HIDIHIplot, = pl.plot(np.sqrt(lambda2_list)*100, HIDIHI, linestyle='-', lw=3, color='y', label=r'5 $\mu$G DFR 2-layer')
HIDAIHIplot, = pl.plot(np.sqrt(lambda2_list)*100, HIDAIHI, linestyle='--', lw=3, color='g', label=r'5 $\mu$G DFR 2-layer')




pl.ylim(0,1.0)
pl.xlim(0,40)
pl.xlabel(r'$\lambda$ (cm)', fontsize=15)
pl.ylabel(r'Normalized Polarization Fraction p/p$_0$', fontsize=15)
pl.title('Three-layer system')




pl.legend(('HDH', 'HDAH', 'HDIH', 'HIDIHI', 'HIDAIHI'),prop={'size': 15}, numpoints=1)

############ SLIDER #####################
#### left, bottom, width, height    #####
#### for the free parameters        #####
#### B_d, B_h, b_d, b_h, ne_d, ne_h #####
#########################################


axBd = pl.axes([0.75, 0.7, 0.2, 0.03], axisbg='white') #bars for slicing Btot_reg_d
axBh = pl.axes([0.75, 0.65, 0.2, 0.03], axisbg='white') #bars for slicing Btot_reg_h
axbd = pl.axes([0.75, 0.6, 0.2, 0.03], axisbg='white') #bars for slicing btot_ran_d
axbh = pl.axes([0.75, 0.55, 0.2, 0.03], axisbg='white') #bars for slicing btot_ran_h
axned = pl.axes([0.75, 0.5, 0.2, 0.03], axisbg='white') #bars for slicing ne_d
axneh = pl.axes([0.75, 0.45, 0.2, 0.03], axisbg='white') #bars for slicing ne_h
#axLd = pl.axes([0.75, 0.40, 0.2, 0.03], axisbg='white') #bars for slicing Ld
#axLh = pl.axes([0.75, 0.34, 0.2, 0.03], axisbg='white') #bars for slicing Ld

   
   
sBd = Slider(axBd, r'B$_d$', 0.0, 20.0, valinit=5.0) #Slider command Btot_reg_d (Axes,label,valmin,valmax,valinterval)
sBd.label.set_size(15) 
sBh = Slider(axBh, r'B$_h$', 0.0, 20.0, valinit=5.0) #Slider command Btot_reg_h
sBh.label.set_size(15)    
sbd = Slider(axbd, r'b$_d$', 0.0, 28.0, valinit=14.0) #Slider command btot_ran_d (Axes,label,valmin,valmax,valinterval)
sbd.label.set_size(15) 
sbh = Slider(axbh, r'b$_h$', 0.0, 8.0, valinit=4.0) #Slider command btot_ran_h
sbh.label.set_size(15) 
sned = Slider(axned, r'ne$_d$', 0.00, 0.22, valinit=0.11, valfmt='%1.4f') #Slider command ne_d (Axes,label,valmin,valmax,valinterval)
sned.label.set_size(15) 
sneh = Slider(axneh, r'ne$_h$', 0.000, 0.020, valinit=0.01, valfmt='%1.4f') #Slider command ne_h --> unfortunately python can not show steps smaller than 0.01 but it still works at least for the model.
sneh.label.set_size(15) 
#   sLd = Slider(axLd, r'L$_d$', 0.0, 1600., valinit=800.) #Slider command L_d (Axes,label,valmin,valmax,valinterval)
#   sLd.label.set_size(15) 
#   sLh = Slider(axLh, r'L$_h$', 0.0, 10000., valinit=5000.) #Slider command L_d (Axes,label,valmin,valmax,valinterval)
#   sLh.label.set_size(15) 


def update(val):
   #Ld_update = sLd.val
   #Lh_update = sLh.val
   
   ne_d_update = sned.val
   ne_h_update = sneh.val
   
   Btot_reg_d_update = sBd.val
   Btot_reg_h_update = sBh.val 
      
   btot_ran_d_update = sbd.val
   btot_ran_h_update = sbh.val 
   
   B0 = Btot_reg_d_update/np.sqrt(1. + (R2/R0)**2. * np.cos(2. * phi - beta2)**2. + 2. * (R2/R0) * np.cos(2. * phi - beta2) * np.cos(p0-p2))
   B2 = Btot_reg_d_update/np.sqrt((R0/R2)**2. +  np.cos(2. * phi - beta2)**2. + 2. * (R0/R2) * np.cos(2. * phi - beta2) * np.cos(p0-p2))
   Bh0 = Btot_reg_h_update/np.sqrt(1. + (Rh1/Rh0)**2. * np.cos(phi - betah1)**2. + 2. * (Rh1/Rh0) * np.cos(phi - betah1) * np.cos(ph0-ph1))
   Bh1 = Btot_reg_h_update/np.sqrt((Rh0/Rh1)**2. +  np.cos(phi - betah1)**2. + 2. * (Rh0/Rh1) * np.cos(phi - betah1) * np.cos(ph0-ph1)) 
   
   Br =  B0 * np.sin(p0) + B2 * np.sin(p2) * np.cos(2.*phi - beta2)
   Bphi =  B0 * np.cos(p0) + B2 * np.cos(p2) * np.cos(2.*phi - beta2)
   Bz = 0.
   Bhr = Bh0 * np.sin(ph0) + Bh1 * np.sin(ph1) * np.cos(phi - betah1)
   Bhphi = Bh0 * np.cos(ph0) + Bh1 * np.cos(ph1) * np.cos(phi - betah1)
   Bhz = 0.
   
   mean_Bx_d = Br * np.cos(phi) - Bphi * np.sin(phi)
   mean_By_d = (Br * np.sin(phi) + Bphi * np.cos(phi))*np.cos(l) + Bz * np.sin(l)
   mean_Bpara_d = -(Br * np.sin(phi) + Bphi * np.cos(phi))*np.sin(l) + Bz * np.cos(l)
   mean_Bx_h = Bhr * np.cos(phi) - Bhphi * np.sin(phi)
   mean_By_h = (Bhr * np.sin(phi) + Bhphi * np.cos(phi))*np.cos(l) + Bhz * np.sin(l)
   mean_Bpara_h = -(Bhr * np.sin(phi) + Bhphi * np.cos(phi))*np.sin(l) + Bhz * np.cos(l)
   
   mean_Bperp_2_d = mean_Bx_d**2. + mean_By_d**2.
   mean_Bperp_2_h = mean_Bx_h**2. + mean_By_h**2.
   
   ################# Random part (Anisotropy only) ##############################
   
   alpha_d_A = 2.0 # constant factor for anisotropy
   alpha_h_A = 1.5 # constant factor for anisotropy
   
   sigmar2_d_A = btot_ran_d_update**2. / (2. + alpha_d_A) 
   sigmar2_h_A = btot_ran_h_update**2. / (2. + alpha_h_A)
   
   sigmax2_d_A =  sigmar2_d_A * (np.cos(phi)**2. + alpha_d_A * np.sin(phi)**2.)
   sigmay2_d_A = sigmar2_d_A * ((np.sin(phi)**2. + alpha_d_A * np.cos(phi)**2.)*np.cos(l)**2. + np.sin(l)**2.)
   sigmax2_h_A =  sigmar2_h_A * (np.cos(phi)**2. + alpha_h_A * np.sin(phi)**2.)
   sigmay2_h_A = sigmar2_h_A * ((np.sin(phi)**2. + alpha_h_A * np.cos(phi)**2.)*np.cos(l)**2. + np.sin(l)**2.)
   
   bpara2_d_A = sigmar2_d_A * ((np.sin(phi)**2. + alpha_d_A * np.cos(phi)**2.)*np.sin(l)**2. + np.cos(l)**2.) # 
   bpara2_h_A = sigmar2_h_A * ((np.sin(phi)**2. + alpha_h_A * np.cos(phi)**2.)*np.sin(l)**2. + np.cos(l)**2.) # 
   
   d_d_A = ((D * sigmaRM_D)/(0.81 * ne_d_update * np.sqrt(bpara2_d_A) * np.sqrt(Ld)))**(2./3.)
   d_h_A = ((D * sigmaRM_D)/(0.81 * ne_h_update * np.sqrt(bpara2_h_A) * np.sqrt(Lh)))**(2./3.)
   #d_d_A = ((D * sigmaRM_D)/(0.81 * ne_d_update * btot_ran_d * np.sqrt(Ld_update)))**(2./3.)
   #d_h_A = ((D * sigmaRM_D)/(0.81 * ne_h_update * btot_ran_h * np.sqrt(Lh_update)))**(2./3.)
   
   ################# Random part (Isotropy only) ##############################
   
   alpha_d_I = 1.0 # constant factor for anisotropy
   alpha_h_I = 1.0 # constant factor for anisotropy
   
   sigmar2_d_I = btot_ran_d_update**2. / (2. + alpha_d_I) 
   sigmar2_h_I = btot_ran_h_update**2. / (2. + alpha_h_I)
   
   sigmax2_d_I =  sigmar2_d_I * (np.cos(phi)**2. + alpha_d_I * np.sin(phi)**2.)
   sigmay2_d_I = sigmar2_d_I * ((np.sin(phi)**2. + alpha_d_I * np.cos(phi)**2.)*np.cos(l)**2. + np.sin(l)**2.)
   sigmax2_h_I =  sigmar2_h_I * (np.cos(phi)**2. + alpha_h_I * np.sin(phi)**2.)
   sigmay2_h_I = sigmar2_h_I * ((np.sin(phi)**2. + alpha_h_I * np.cos(phi)**2.)*np.cos(l)**2. + np.sin(l)**2.)
   
   bpara2_d_I = sigmar2_d_I * ((np.sin(phi)**2. + alpha_d_I * np.cos(phi)**2.)*np.sin(l)**2. + np.cos(l)**2.) # 
   bpara2_h_I = sigmar2_h_I * ((np.sin(phi)**2. + alpha_h_I * np.cos(phi)**2.)*np.sin(l)**2. + np.cos(l)**2.) # 
   
   d_d_I = ((D * sigmaRM_D)/(0.81 * ne_d_update * np.sqrt(bpara2_d_I) * np.sqrt(Ld)))**(2./3.)
   d_h_I = ((D * sigmaRM_D)/(0.81 * ne_h_update * np.sqrt(bpara2_h_I) * np.sqrt(Lh)))**(2./3.)
   #d_d_I = ((D * sigmaRM_D)/(0.81 * ne_d_update * btot_ran_d * np.sqrt(Ld)))**(2./3.)
   #d_h_I = ((D * sigmaRM_D)/(0.81 * ne_h_update * btot_ran_h * np.sqrt(Lh)))**(2./3.)
   
   
   ##############################################################################
   
   
   ##############################################################################
   
   epsilon_d_DFR = 0.1 * (mean_Bperp_2_d) #add turbulent fields here if present!
   epsilon_h_DFR = 0.01 * (mean_Bperp_2_h) #add turbulent fields here if present!
   epsilon_d_IFD = 0.1 * (mean_Bperp_2_d + sigmax2_d_A + sigmay2_d_A) #add turbulent fields here if present!
   epsilon_h_IFD = 0.01 * (mean_Bperp_2_h + sigmax2_h_A + sigmay2_h_A) #add turbulent fields here if present!
   
   Id_DFR = epsilon_d_DFR * Ld
   Ih_DFR = epsilon_h_DFR * Lh
   Id_IFD = epsilon_d_IFD * Ld
   Ih_IFD = epsilon_h_IFD * Lh
   
   R_d = 0.81 * ne_d_update * mean_Bpara_d * Ld
   R_h = 0.81 * ne_h_update * mean_Bpara_h * Lh
   #sigmaRM_d_A = 0.81 * ne_d * btot_ran_d * np.sqrt(Ld * d_d_A)
   #sigmaRM_h_A = 0.81 * ne_h * btot_ran_h * np.sqrt(Lh * d_h_A)
   sigmaRM_d_A = 0.81 * ne_d_update * np.sqrt(bpara2_d_A) * np.sqrt(Ld * d_d_A)
   sigmaRM_h_A = 0.81 * ne_h_update * np.sqrt(bpara2_h_A) * np.sqrt(Lh * d_h_A)
   sigmaRM_d_I = 0.81 * ne_d_update * np.sqrt(bpara2_d_I) * np.sqrt(Ld * d_d_I)
   sigmaRM_h_I = 0.81 * ne_h_update * np.sqrt(bpara2_h_I) * np.sqrt(Lh * d_h_I)
   
   
   Omega_d_A = 2. * sigmaRM_d_A**2. * lambda2_list**2
   Omega_h_A = 2. * sigmaRM_h_A**2. * lambda2_list**2
   Omega_d_I = 2. * sigmaRM_d_I**2. * lambda2_list**2
   Omega_h_I = 2. * sigmaRM_h_I**2. * lambda2_list**2
   
   Omega_d = 0.   #for purley regular field only in the disk
   Omega_h = 0.   #for purley regular field only in the halo
   
   C_d = 2. * R_d * lambda2_list
   C_h = 2. * R_h * lambda2_list
   
   F_DFR = C_d * C_h
   G_DFR = 0.
   F_IFD_A = Omega_d_A * Omega_h_A + C_d * C_h
   G_IFD_A = Omega_h_A * C_d - Omega_d_A * C_h
   G_DdA_H = - Omega_d_A * C_h
   F_IFD_I = Omega_d_I * Omega_h_I + C_d * C_h
   G_IFD_I = Omega_h_I * C_d - Omega_d_I * C_h
   F_DdAI_HhI = Omega_d_A * Omega_h_I + C_d * C_h
   G_DdAI_HhI = Omega_h_I * C_d - Omega_d_A * C_h
   
   Psi0d_DFR = np.pi/2. - np.arctan(np.cos(l)*np.tan(phi)) + np.arctan2((mean_By_d),(mean_Bx_d))
   Psi0h_DFR = np.pi/2. - np.arctan(np.cos(l)*np.tan(phi)) + np.arctan2((mean_By_h),(mean_Bx_h))
   
   Psi0d_IFD = np.pi/2. - np.arctan(np.cos(l)*np.tan(phi)) + 0.5 * np.arctan2((2. * mean_Bx_d * mean_By_d),(mean_Bx_d**2. - mean_By_d**2. + sigmax2_d_A - sigmay2_d_A))
   Psi0h_IFD = np.pi/2. - np.arctan(np.cos(l)*np.tan(phi)) + 0.5 * np.arctan2((2. * mean_Bx_h * mean_By_h),(mean_Bx_h**2. - mean_By_h**2. + sigmax2_h_A - sigmay2_h_A))
   
   
   DeltaPsi_dh_DFR = Psi0d_DFR - Psi0h_DFR
   
   DeltaPsi_dh_IFD = Psi0d_IFD - Psi0h_IFD
   
   DeltaPsi_dh = Psi0d_IFD - Psi0h_DFR #for regular+turbulent in Disk but purley regular in halo
   
   Ad_DFR = (Id_DFR/(Id_DFR+2.*Ih_DFR)) * np.abs((np.sinc(R_d * lambda2_list / np.pi)))
   Ah_DFR = (Ih_DFR/(Id_DFR+2.*Ih_DFR)) * np.abs((np.sinc(R_h * lambda2_list / np.pi)))
   
   Ad_IFD = (Id_IFD/(Id_IFD+2.*Ih_IFD)) * (np.sinh(sigmaRM_d_A**2. * lambda2_list**2.)/(sigmaRM_d_A**2. * lambda2_list**2.)) * np.exp(-sigmaRM_d_A**2. * lambda2_list**2.)
   Ah_IFD = (Ih_IFD/(Id_IFD+2.*Ih_IFD)) * (np.sinh(sigmaRM_h_A**2. * lambda2_list**2.)/(sigmaRM_h_A**2. * lambda2_list**2.)) * np.exp(-sigmaRM_h_A**2. * lambda2_list**2.)
   
   ############### WAVELENGTH-INDEPENDENT DEPOLARIZATION ####################
   
   W_d = 1.0 #no wavelength-dep depolarization for purley regular fields
   W_h = 1.0 #no wavelength-dep depolarization for purley regular fields
   
   W_d_AI = (mean_Bperp_2_d/(mean_Bperp_2_d + 2. * sigmax2_d_I)) * (np.sqrt((mean_Bx_d**2. - mean_By_d**2. + sigmax2_d_A - sigmay2_d_A)**2. + 4. * mean_Bx_d**2. * mean_By_d**2.)/(mean_Bperp_2_d + sigmax2_d_A + sigmay2_d_A)) #Aniotropic + Isotropic in Disk
   
   W_h_AI = (mean_Bperp_2_h/(mean_Bperp_2_h + 2. * sigmax2_h_I)) * (np.sqrt((mean_Bx_h**2. - mean_By_h**2. + sigmax2_h_A - sigmay2_h_A)**2. + 4. * mean_Bx_h**2. * mean_By_h**2.)/(mean_Bperp_2_h + sigmax2_h_A + sigmay2_h_A)) #Aniotropic + Isotropic in Halo
   
   W_d_A = (np.sqrt((mean_Bx_d**2. - mean_By_d**2. + sigmax2_d_A - sigmay2_d_A)**2. + 4. * mean_Bx_d**2. * mean_By_d**2.)/(mean_Bperp_2_d + sigmax2_d_A + sigmay2_d_A)) #only Anisotropic
   W_h_A = (np.sqrt((mean_Bx_h**2. - mean_By_h**2. + sigmax2_h_A - sigmay2_h_A)**2. + 4. * mean_Bx_h**2. * mean_By_h**2.)/(mean_Bperp_2_h + sigmax2_h_A + sigmay2_h_A)) #only Anisotropic
   
   W_d_I = (mean_Bperp_2_d/(mean_Bperp_2_d + 2. * sigmax2_d_I)) #only Isotropic
   W_h_I = (mean_Bperp_2_h/(mean_Bperp_2_h + 2. * sigmax2_h_I)) #only Isotropic
   
   
   ######### MODELS ##########################################################
   ######### D : Disk regular ################################################
   ######### H : Halo regular ################################################
   ######### dI : Disk Isotropic random ######################################
   ######### hI : Halo Isotropic random ###################################### 
   ######### dA : Disk Anisotropic random ####################################
   ######### hA : Halo Anisotropic random ####################################
   ######### Example: DdA_HhI means regular fields in Disk and Halo (DH), ####
   ######### Anisotropic random field in Disk (dA), ##########################
   ######### and Isotropic random field in Halo (hI) #########################
   ###########################################################################
   
   H_D_H = np.sqrt(2. * Ah_DFR**2. * (1. + np.cos(2. * (R_d + R_h) * lambda2_list)) + Ad_DFR**2. + 2. * Ad_DFR * Ah_DFR * (np.cos(-2. * DeltaPsi_dh_DFR + (R_d + R_h) * lambda2_list) + np.cos(2. * DeltaPsi_dh_DFR + (R_d + R_h) * lambda2_list))) #ONLY regular magnetic fields in D and H and far side H
   
   hI_dI_hI = 2. * W_h_I * Ah_IFD + W_d_I * Ad_IFD #ONLY iotropic random MFs in d and h and h2
   hA_dA_hA = 2. * W_h_A * Ah_IFD + W_d_A * Ad_IFD #ONLY aniotropic random MFs in d and h and h2
   
   HDAH = np.sqrt(2. * W_h**2. * (Ih_DFR/(2.*Ih_DFR+Id_IFD))**2. * (((1. -2. * np.exp(-Omega_h) * np.cos(C_h) + np.exp(-2. * Omega_h)) * (1. + np.cos(C_d + C_h))) / (Omega_h**2. + C_h**2.)) + W_d_A**2. * (Id_IFD/(2.*Ih_DFR+Id_IFD))**2. * ((1. - 2. * np.exp(-Omega_d_A) * np.cos(C_d) + np.exp(-2. * Omega_d_A)) / (Omega_d_A**2. + C_d**2.)) + W_d_A * W_h * (Id_IFD * Ih_DFR/(Id_IFD+2.*Ih_DFR)**2.) * (2./(F_DFR**2. + G_DdA_H**2.)) * ((F_DFR * np.cos(-2. * DeltaPsi_dh + C_d) + G_DdA_H * np.sin(-2. * DeltaPsi_dh + C_d)) + (F_DFR * np.cos(2. * DeltaPsi_dh + C_h) - G_DdA_H * np.sin(2. * DeltaPsi_dh + C_h)) + np.exp(-(Omega_d_A + Omega_h)) * ((F_DFR * np.cos(2. * DeltaPsi_dh + C_d) - G_DdA_H * np.sin(2. * DeltaPsi_dh + C_d)) + (F_DFR * np.cos(-2. * DeltaPsi_dh + C_h) + G_DdA_H * np.sin(-2. * DeltaPsi_dh + C_h))) - np.exp(-Omega_d_A) * ((F_DFR * np.cos(2. * DeltaPsi_dh + C_d + C_h) - G_DdA_H * np.sin(2. * DeltaPsi_dh + C_d + C_h)) + (F_DFR * np.cos(-2. * DeltaPsi_dh) + G_DdA_H * np.sin(-2. * DeltaPsi_dh))) - np.exp(-Omega_h) * ((F_DFR * np.cos(-2. * DeltaPsi_dh + C_d + C_h) + G_DdA_H * np.sin(-2. * DeltaPsi_dh + C_d + C_h)) + (F_DFR * np.cos(2. * DeltaPsi_dh) - G_DdA_H * np.sin(2. * DeltaPsi_dh)))))
   
   HDIH = np.sqrt(2. * W_h**2. * (Ih_DFR/(2.*Ih_DFR+Id_IFD))**2. * (((1. -2. * np.exp(-Omega_h) * np.cos(C_h) + np.exp(-2. * Omega_h)) * (1. + np.cos(C_d + C_h))) / (Omega_h**2. + C_h**2.)) + W_d_I**2. * (Id_IFD/(2.*Ih_DFR+Id_IFD))**2. * ((1. - 2. * np.exp(-Omega_d_I) * np.cos(C_d) + np.exp(-2. * Omega_d_I)) / (Omega_d_I**2. + C_d**2.)) + W_d_I * W_h * (Id_IFD * Ih_DFR/(Id_IFD+2.*Ih_DFR)**2.) * (2./(F_DFR**2. + G_DdA_H**2.)) * ((F_DFR * np.cos(-2. * DeltaPsi_dh + C_d) + G_DdA_H * np.sin(-2. * DeltaPsi_dh + C_d)) + (F_DFR * np.cos(2. * DeltaPsi_dh + C_h) - G_DdA_H * np.sin(2. * DeltaPsi_dh + C_h)) + np.exp(-(Omega_d_I + Omega_h)) * ((F_DFR * np.cos(2. * DeltaPsi_dh + C_d) - G_DdA_H * np.sin(2. * DeltaPsi_dh + C_d)) + (F_DFR * np.cos(-2. * DeltaPsi_dh + C_h) + G_DdA_H * np.sin(-2. * DeltaPsi_dh + C_h))) - np.exp(-Omega_d_I) * ((F_DFR * np.cos(2. * DeltaPsi_dh + C_d + C_h) - G_DdA_H * np.sin(2. * DeltaPsi_dh + C_d + C_h)) + (F_DFR * np.cos(-2. * DeltaPsi_dh) + G_DdA_H * np.sin(-2. * DeltaPsi_dh))) - np.exp(-Omega_h) * ((F_DFR * np.cos(-2. * DeltaPsi_dh + C_d + C_h) + G_DdA_H * np.sin(-2. * DeltaPsi_dh + C_d + C_h)) + (F_DFR * np.cos(2. * DeltaPsi_dh) - G_DdA_H * np.sin(2. * DeltaPsi_dh)))))
   
   HIDIHI = np.sqrt(2. * W_h_I**2. * (Ih_IFD/(2.*Ih_IFD+Id_IFD))**2. * (((1. -2. * np.exp(-Omega_h_I) * np.cos(C_h) + np.exp(-2. * Omega_h_I)) * (1. + np.cos(C_d + C_h))) / (Omega_h_I**2. + C_h**2.)) + W_d_I**2. * (Id_IFD/(2.*Ih_IFD+Id_IFD))**2. * ((1. - 2. * np.exp(-Omega_d_I) * np.cos(C_d) + np.exp(-2. * Omega_d_I)) / (Omega_d_I**2. + C_d**2.)) + W_d_I * W_h_I * (Id_IFD * Ih_IFD/(Id_IFD+2.*Ih_IFD)**2.) * (2./(F_IFD_I**2. + G_IFD_I**2.)) * ((F_IFD_I * np.cos(-2. * DeltaPsi_dh_IFD + C_d) + G_IFD_I * np.sin(-2. * DeltaPsi_dh_IFD + C_d)) + (F_IFD_I * np.cos(2. * DeltaPsi_dh_IFD + C_h) - G_IFD_I * np.sin(2. * DeltaPsi_dh_IFD + C_h)) + np.exp(-(Omega_d_I + Omega_h_I)) * ((F_IFD_I * np.cos(2. * DeltaPsi_dh_IFD + C_d) - G_IFD_I * np.sin(2. * DeltaPsi_dh_IFD + C_d)) + (F_IFD_I * np.cos(-2. * DeltaPsi_dh_IFD + C_h) + G_IFD_I * np.sin(-2. * DeltaPsi_dh_IFD + C_h))) - np.exp(-Omega_d_I) * ((F_IFD_I * np.cos(2. * DeltaPsi_dh_IFD + C_d + C_h) - G_IFD_I * np.sin(2. * DeltaPsi_dh_IFD + C_d + C_h)) + (F_IFD_I * np.cos(-2. * DeltaPsi_dh_IFD) + G_IFD_I * np.sin(-2. * DeltaPsi_dh_IFD))) - np.exp(-Omega_h_I) * ((F_IFD_I * np.cos(-2. * DeltaPsi_dh_IFD + C_d + C_h) + G_IFD_I * np.sin(-2. * DeltaPsi_dh_IFD + C_d + C_h)) + (F_IFD_I * np.cos(2. * DeltaPsi_dh_IFD) - G_IFD_I * np.sin(2. * DeltaPsi_dh_IFD)))))
   
   
   #HIDIHI = np.sqrt(W_d_I**2. * (Id_IFD/(Id_IFD+2.*Ih_IFD))**2. * ((1. - 2. * np.exp(-Omega_d_I) * np.cos(C_d) + np.exp(-2. * Omega_d_I))/(Omega_d_I**2. + C_d**2.)) + W_h_I**2. * (Ih_IFD/(Id_IFD+2.*Ih_IFD))**2. * ((1. - 2. * np.exp(-Omega_h_I) * np.cos(C_h) + np.exp(-2. * Omega_h_I))/(Omega_h_I**2. + C_h**2.)) + W_d_I * W_h_I * (Id_IFD * Ih_IFD/(Id_IFD+2.*Ih_IFD)**2.) * (2./(F_IFD_I**2. + G_IFD_I**2.)) * ((F_IFD_I * np.cos(2. * DeltaPsi_dh_IFD + C_h) - G_IFD_I * np.sin(2. * DeltaPsi_dh_IFD + C_h)) + np.exp(-(Omega_d_I + Omega_h_I)) * (F_IFD_I * np.cos(2. * DeltaPsi_dh_IFD + C_d) - G_IFD_I * np.sin(2. * DeltaPsi_dh_IFD + C_d)) - np.exp(-Omega_d_I) * (F_IFD_I * np.cos(2. * DeltaPsi_dh_IFD + C_d + C_h) - G_IFD_I * np.sin(2. * DeltaPsi_dh_IFD + C_d + C_h)) - np.exp(-Omega_h_I) * (F_IFD_I * np.cos(2. * DeltaPsi_dh_IFD) - G_IFD_I * np.sin(2. * DeltaPsi_dh_IFD)))) #Regular in D and H and isotropic in d and h 
   
   
   HIDAIHI = np.sqrt(2. * W_h_I**2. * (Ih_IFD/(2.*Ih_IFD+Id_IFD))**2. * (((1. -2. * np.exp(-Omega_h_I) * np.cos(C_h) + np.exp(-2. * Omega_h_I)) * (1. + np.cos(C_d + C_h))) / (Omega_h_I**2. + C_h**2.)) + W_d_AI**2. * (Id_IFD/(2.*Ih_IFD+Id_IFD))**2. * ((1. - 2. * np.exp(-Omega_d_A) * np.cos(C_d) + np.exp(-2. * Omega_d_A)) / (Omega_d_A**2. + C_d**2.)) + W_d_AI * W_h_I * (Id_IFD * Ih_IFD/(Id_IFD+2.*Ih_IFD)**2.) * (2./(F_DdAI_HhI**2. + G_DdAI_HhI**2.)) * ((F_DdAI_HhI * np.cos(-2. * DeltaPsi_dh_IFD + C_d) + G_DdAI_HhI * np.sin(-2. * DeltaPsi_dh_IFD + C_d)) + (F_DdAI_HhI * np.cos(2. * DeltaPsi_dh_IFD + C_h) - G_DdAI_HhI * np.sin(2. * DeltaPsi_dh_IFD + C_h)) + np.exp(-(Omega_d_A + Omega_h_I)) * ((F_DdAI_HhI * np.cos(2. * DeltaPsi_dh_IFD + C_d) - G_DdAI_HhI * np.sin(2. * DeltaPsi_dh_IFD + C_d)) + (F_DdAI_HhI * np.cos(-2. * DeltaPsi_dh_IFD + C_h) + G_DdAI_HhI * np.sin(-2. * DeltaPsi_dh_IFD + C_h))) - np.exp(-Omega_d_A) * ((F_DdAI_HhI * np.cos(2. * DeltaPsi_dh_IFD + C_d + C_h) - G_DdAI_HhI * np.sin(2. * DeltaPsi_dh_IFD + C_d + C_h)) + (F_DdAI_HhI * np.cos(-2. * DeltaPsi_dh_IFD) + G_DdAI_HhI * np.sin(-2. * DeltaPsi_dh_IFD))) - np.exp(-Omega_h_I) * ((F_DdAI_HhI * np.cos(-2. * DeltaPsi_dh_IFD + C_d + C_h) + G_DdAI_HhI * np.sin(-2. * DeltaPsi_dh_IFD + C_d + C_h)) + (F_DdAI_HhI * np.cos(2. * DeltaPsi_dh_IFD) - G_DdAI_HhI * np.sin(2. * DeltaPsi_dh_IFD)))))
   
   
   #HIDAIHI = np.sqrt(W_d_AI**2. * (Id_IFD/(Id_IFD+2.*Ih_IFD))**2. * ((1. - 2. * np.exp(-Omega_d_A) * np.cos(C_d) + np.exp(-2. * Omega_d_A))/(Omega_d_A**2. + C_d**2.)) + W_h_I**2. * (Ih_IFD/(Id_IFD+2.*Ih_IFD))**2. * ((1. - 2. * np.exp(-Omega_h_I) * np.cos(C_h) + np.exp(-2. * Omega_h_I))/(Omega_h_I**2. + C_h**2.)) + W_d_AI * W_h_I * (Id_IFD * Ih_IFD/(Id_IFD+Ih_IFD)**2.) * (2./(F_DdAI_HhI**2. + G_DdAI_HhI**2.)) * ((F_DdAI_HhI * np.cos(2. * DeltaPsi_dh_IFD + C_h) - G_DdAI_HhI * np.sin(2. * DeltaPsi_dh_IFD + C_h)) + np.exp(-(Omega_d_A + Omega_h_I)) * (F_DdAI_HhI * np.cos(2. * DeltaPsi_dh_IFD + C_d) - G_DdAI_HhI * np.sin(2. * DeltaPsi_dh_IFD + C_d)) - np.exp(-Omega_d_A) * (F_DdAI_HhI * np.cos(2. * DeltaPsi_dh_IFD + C_d + C_h) - G_DdAI_HhI * np.sin(2. * DeltaPsi_dh_IFD + C_d + C_h)) - np.exp(-Omega_h_I) * (F_DdAI_HhI * np.cos(2. * DeltaPsi_dh_IFD) - G_DdAI_HhI * np.sin(2. * DeltaPsi_dh_IFD)))) #Regular in Disk and Halo and A+I in Disk and I in Halo
   
   
   
   HDHplot.set_ydata(H_D_H) #DH is getting updated
   #AAplot.set_ydata(dA_hA) #AA is getting updated
   HDAHplot.set_ydata(HDAH) #DAH is getting updated
   HDIHplot.set_ydata(HDIH) #DAH is getting updated
   HIDIHIplot.set_ydata(HIDIHI) #DIHI is getting updated
   HIDAIHIplot.set_ydata(HIDAIHI) #DAIHI is getting updated
   fig.canvas.draw_idle()



sBd.on_changed(update)
sBh.on_changed(update)
sbd.on_changed(update)
sbh.on_changed(update)
sned.on_changed(update)
sneh.on_changed(update)
#sLd.on_changed(update)
#sLh.on_changed(update)
   
   
   
resetax = pl.axes([0.8, 0.25, 0.1, 0.04]) #left, bottom, width, height
button = Button(resetax, 'Reset', color='white', hovercolor='0.975')
def reset(event):
   sBd.reset()
   sBh.reset()
   sbd.reset()
   sbh.reset()
   sned.reset()
   sneh.reset()
   #sLd.reset()
   #sLh.reset()
button.on_clicked(reset)
   
###### show plot ########   
pl.show()
pl.figure(1)
pl.clf()
   
   
   
