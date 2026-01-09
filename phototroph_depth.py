import os, math, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(__file__)+'/../NutMEG')

import matplotlib.pyplot as plt
import numpy as np

import NutMEG as nm


def initial_conditions(R, comp={}):
    """
    Set up the a reactor R to use for photosynthesis
    by populating it with reagents. To change
    the concentrations used, pass them in the dict comp in the  format
    {Name : conc}. Currently, this only sets up the reagents for oxygenic
    photosynthesis from CO2.
    """
    # metabolic concentrations
    mol_CO2 = comp.pop('CO2(aq)', 0.001)
    mol_O2 = comp.pop('O2(g)', 1e-5)
    mol_CH2O =  comp.pop('Formaldehyde(aq)', 0.001) #CH2O

    # life also needs a source of N and P
    mol_NH3 = comp.pop('NH3(aq)', 0.1)
    mol_H2PO4 = comp.pop('H2PO4-', 0.1)

    # concentration of H is tied to pH.
    mol_H=10**(-R.pH)

    # now set up these chemical species as reagents
    CO2 = nm.reaction.reagent('CO2(aq)', R.env, phase='aq', conc=mol_CO2,
      activity=mol_CO2)
    O2 = nm.reaction.reagent('O2(g)', R.env, phase='g', conc=mol_O2,
      activity=mol_O2)

    CH2O = nm.reaction.reagent('Formaldehyde(aq)', R.env, phase='aq', conc=mol_CH2O,
      activity=mol_CH2O)
    H2O = nm.reaction.reagent('H2O(aq)', R.env, phase='l', conc=55.5,
      activity=1, phase_ss=True)
    H = nm.reaction.reagent('H+', R.env, charge=1, conc=mol_H,
      phase='aq', activity=mol_H)


    # add these to the composition of R. This will be shared between
    # the organisms
    R.composition = {CO2.name:CO2, O2.name:O2,
      CH2O.name:CH2O, H2O.name:H2O, H.name:H}

    # add in the extra nutrients
    R.composition['NH3(aq)'] = nm.reaction.reagent('NH3(aq)', R.env,
      phase='aq', conc=mol_NH3, activity=mol_NH3)
    R.composition['H2PO4-'] = nm.reaction.reagent('H2PO4-', R.env, phase='aq',
      conc=mol_H2PO4, activity=mol_H2PO4)

    rxn = nm.reaction.reaction({CO2:1, H2O:1}, {CH2O:1, O2:1}, R.env)
    R.add_reaction(rxn, overwrite=False)

    return R, rxn


def Idepth(I0, d, k=-0.1):
    """
    Compute irradiance at depth d with decay factor k. From Middelburg 2019.

    Parameters
    ----------
    I0 : float
        Irradiance at surface in W / m2
    d : float, np.array
        Depth position in m
    k : float, optional
        Decay factor in irradiance (defaul is -0.1, from Middelburg 2019 example)
    """
    return I0*np.exp(k*d)



def Ndepth(N0, Nmax, d, k=0.1):
    """
    Compute bioavailable N concentration at depth D following example from
    Middelburg 2019 (i.e., an example profile, not modelling a specific system).

    In this example, N is constant at N0 for the first 25m, then exponentially
    increases toward Nmax.

    Parameters
    ----------
    N0 : float
        Bioavailable N concentration at surface in mol/kg
    Nmax : float
        Maximumm bioavailable N concentration at infinite depth in mol/kg
    d : float, np.array
        Depth position in m
    k : float, optional
        Exponent factor  for increase in bioavailable N concentration.
        (default is 0.1, from Middelburg 2019 example)
    """
    if d<=25:
        return N0
    else:
        return Nmax *(1-np.exp(-k*(d-25)))



def Platt_tanh(resp, alpha, Pmax, I):
    """
    Forcing function for growth inhibition based on low irradiance (Platt et al., 1976)
    Returns forcing factor beween 0 and 1.

    Parameters
    ----------
    resp : Nonetype,
        Required arg for forcing_functions when they depend on properties
        accessible by the host organism's respiration (e.g., a local substrate
        concentration). This function does not have such a dependence, so
        resp may be passed as None when calling outside a
        NutMEG.base_organism.respirator object
    alpha : float
        Platt fitting parameter which defines the slope of the P vs I curve
        at I=0.
    Pmax : float
        Platt fitting parameter which defines the maximum productivity in
        mg C / mg Chl a / h (or commensurate with alpha)
    I : float
        Irradiance in W / m2 (or commensurate with alpha)
    """
    return np.tanh(alpha * I / Pmax)



def Monod_fixed(resp, N, KN):
    """
    Forcing function for growth inhibition based on bioavailable N concentration
    using a Monod model.  This is a simplified implementation of the formal
    NutMEG-integrated option in saved_organisms.KineticallyLimitedOrganism which
    does not look into the host.reactor, and instead takes the concentration
    directly as an argument.

    Parameters
    ----------
    resp : Nonetype,
        Required arg for forcing_functions when they depend on properties
        accessible by the host organism's respiration (e.g., a local substrate
        concentration). This function does not have such a dependence, so
        resp may be passed as None when calling outside a
        NutMEG.base_organism.respirator object
    N : float
        Concentration of bioavailable N in mol/kg
    KN : float
        Monod half-saturation constant of N uptake in mol/kg
    """
    return N / (N + KN)



def get_phototroph(R, rxn, Pm, I, alpha, N, K_N, mmr, mgr, num=1e6, name='Phototroph'):
    """
    Create a NutMEG.horde object representing a phototroph.


    Parameters
    ----------
    R : NutMEG.reactor
        reactor object hosting the organism
    rxn : NutMEG.reaction
        reaction object hosting the overall metabolic reaction
    Pm : float
        Platt fitting parameter which defines the maximum productivity in
        mg C / mg Chl a / h (or commensurate with alpha)
    I : float
        Irradiance in W / m2 (or commensurate with alpha)
    alpha : float
        Platt fitting parameter which defines the slope of the P vs I curve
        at I=0.
    N : float
        Concentration of bioavailable N in mol/kg
    K_N : float
        Monod half-saturation constant of N uptake in mol/kg
    mmr : float
        Maximum metabolic rate (aka zeroth order rate constant k_max) in
        mol of reaction / s.
    mgr : float
        Maximum growth rate (aka mu_max) in /s.
    num : float, optional
        Number of organisms in the horde. Larger hordes will yield more precise
        growth rates but risk consuming all resources in long time-steps.
        Default 1e6 (per kg water)
    name : str, optional
        String identifier for this organism. Default 'Phototroph'

    """

    _phototroph = nm.horde(name, R, rxn, num,
      unit='cells', # alternative units are not yet supported.
      workoutID=False, E_synth=8e-10,
      respiration_kwargs = {
        'rate_func':'zeroth order',
        'max_metabolic_rate':mmr,
        'G_ATP':'default',
        'G_net_pathway':-1000000, # set arbitrarily high. See rxn.molar_gibbs
        # calculation below for reasoning.
        'G_C':600000, # J/mol of CO2 fixed;
        # 3 ATP + 2 NADP, 1 NADPH = 3 ATP is typical for oxygenic photosynthesis
        ##
        ## The kinetic forcing factors below are commented out and have been
        ## included as growth forcing factors instead. This is because including
        ## irradiance limitation as a kinetic factor does not work when we use
        ## arbitrarily large energy yields. Thus, for the time being, we will
        ## only consider growth forcing functions for photosynthesis.
        ##
        # 'kinetic_forcing_parameters':{'Platt':(Platt_tanh, ['alpha', 'Pmax', 'I'])},
        # 'kinetic_F_attrs':{'alpha':alpha, 'I':I, 'Pmax':Pm},
        'rate_constant_env':mmr,
        },
      CHNOPS_kwargs = {
        'max_growth_rate' : mgr,
        ##
        ## Adding forcing functions is currently quite complicated. Two attributes
        ## need to be adjusted in the host's CHNOPS attribute: CHNOPS_forcing_parameters
        ## is a dictionary containing function identifiers as keys, and tuples
        ## as values containing: 1. the forcing function, 2. a list of identifiers
        ## that correspond to the variables the function needs to run.
        ## The second attribute is CHNOPS_F_attrs, which is a dictionary that
        ## associates the identifies above to a value.
        ## The implementation is arranged in this way to accomodate scenarios
        ## where the multiple forcing functions depend on the same attributes.
        'CHNOPS_forcing_parameters':{
          'Monod':(Monod_fixed, ['N', 'K_N']),
          'Platt':(Platt_tanh, ['alpha', 'Pmax', 'I'])
          },
        'CHNOPS_F_attrs':{
          'N':N, 'K_N':K_N,
          'alpha':alpha, 'I':I, 'Pmax':Pm
          }
        })

    return _phototroph



def get_phototroph_rate(_phototroph, stepsize=600):
    """
    Returns growth rate of a horde using defined time step.
    """

    _phototroph.take_step(stepsize)

    return _phototroph.growth_rate




def __main__():

    R = nm.reactor('reactor1', workoutID=False, pH=7.0)
    R, rxn = initial_conditions(R)

    rxn.update_molar_gibbs_from_quotient()
    print(r'$\Delta G$', rxn.molar_gibbs)


    # Pm and alpha are platt model properties. The ones identified below are
    # global averages based on the dataset in Bouman et al., (2018).
    Pm = 3.1145 # mg C / mg Chl /h
    alpha = 0.04278

    # ratio between g of Chl a and g of cells.
    CChl_to_Cbm = 0.01 # 0.003 to 0.055 (Middelburg 2019)
    Cbm_to_percell = 3e-13 # number of cells in 1g of biomass. Use nutmeg default (Higgins & Cockell 2020)

    # Maximum metabolic and growth rates
    mmr = Pm * CChl_to_Cbm * Cbm_to_percell / (12*3600)  # mol CO2 / cell / s
    mgr = Pm * CChl_to_Cbm / 3600  # growth rate in /s

    # values from Middelburg 2019 example
    I_0 = 116
    K_N = 1e-9
    N_0 = 0.1e-9
    N_max = 10e-9


    depths = np.linspace(0,80,num=801)

    Idepths = Idepth(I_0, depths)

    F_E = Platt_tanh(None, alpha, Pm, Idepths) # an array of irradiance forcing factors with depth

    Ndepths = np.array([Ndepth(N_0, N_max, d) for d in depths])

    F_N = Monod_fixed(None, Ndepths, K_N) # an array of nitrogen forcing factors with depth

    # array of phototroph growth rates with depth
    Prod = np.array([get_phototroph_rate(get_phototroph(R, rxn, Pm, I, alpha, N, K_N, mmr, mgr)) for N, I in zip(Ndepths, Idepths)])

    plt.plot(F_E, depths, label='F_I (irradiance)')
    plt.plot(F_N, depths, label='F_N (nutrients)')
    plt.plot(Prod*1e6, depths, label=r'Growth rate (*10$^6$)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('_depth.png')
    plt.close()

    print("\nPlot saved as '_depth.png'\n")



__main__()
