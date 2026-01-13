import os, math, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(__file__)+'/../NutMEG')

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ssl

# Import NutMEG
import NutMEG as nm

# SSL fix for Google Sheets
ssl._create_default_https_context = ssl._create_unverified_context


def load_matano_data(url, interpolate=False):
    """
    Load Lake Matano data from Google Sheets in long format.

    Parameters
    ----------
    url : str
        URL of the CSV containing parameter, depth_m, and value columns.
    interpolate : bool, optional
        If True, interpolate data to a 1 m grid from 0 to 550 m.
        If False, return data on 0-550 m grid with NaN for unmeasured depths.

    Returns
    -------
    data : dict
        Dictionary containing 'depth' and parameter arrays.
    """
    df_long = pd.read_csv(url)

    # TEST: Add row with H2S
    df_long = pd.concat([df_long, pd.DataFrame([{'parameter': 'H2S', 'depth_m': 0, 'value': 0}])], ignore_index=True)
    df_long = pd.concat([df_long, pd.DataFrame([{'parameter': 'H2S', 'depth_m': 100, 'value': 0}])], ignore_index=True)
    # Change first two NO3 data points
    df_long.loc[(df_long['parameter'] == 'NO3') & (df_long['depth_m'] == 9), 'value'] = 0.0
    df_long.loc[(df_long['parameter'] == 'NO3') & (df_long['depth_m'] == 20), 'value'] = 0.0

    # Parameters we want to extract, mapped to their specific years
    # Empty string '' means no year value (NaN in the 'year' column)
    params = {
        'NH4': '',
        'NO3': 2010,
        'P': 2005,
        'par': 2007,
        'temp': 2004,
        'H2S': '',
        'SO4': '',
        'O2': 2004
    }

    # Create depth grid (every 1 meter from 0 to 550)
    depths = np.arange(0, 551, 1)
    data = {'depth': depths}

    for param, year in params.items():
        # Filter data for this parameter and its specific year
        param_data = df_long[df_long['parameter'] == param].copy()
        
        if year == '':
            # Empty string means look for rows where year is NaN
            param_data = param_data[param_data['year'].isna()]
        else:
            # Filter for the specific year
            param_data = param_data[param_data['year'] == year]
        
        param_data = param_data.sort_values('depth_m')

        if len(param_data) > 0:
            if interpolate:
                # Interpolate values to fill gaps
                values = np.interp(
                    depths,
                    param_data['depth_m'],
                    param_data['value'],
                    left=np.nan,
                    right=np.nan
                )
            else:
                # Place measured values on grid, leave NaN for unmeasured depths
                values = np.full(len(depths), np.nan)
                for _, row in param_data.iterrows():
                    depth_idx = int(row['depth_m'])
                    if 0 <= depth_idx < len(depths):
                        values[depth_idx] = row['value']
            
            data[param] = values
        else:
            data[param] = np.full(len(depths), np.nan)

    return data


def get_incremented_filename(base_name, extension, directory='.'):
    """
    Generate an incremented filename if the base name already exists.
    e.g., if 'plot.png' exists, return 'plot_1.png', then 'plot_2.png', etc.
    """
    filepath = os.path.join(directory, f"{base_name}{extension}")
    if not os.path.exists(filepath):
        return filepath
    
    counter = 1
    while True:
        filepath = os.path.join(directory, f"{base_name}{counter}{extension}")
        if not os.path.exists(filepath):
            return filepath
        counter += 1


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


def Platt_tanh(resp, alpha, Pmax, I):
    """
    Forcing function for growth inhibition based on low irradiance (Platt et al., 1976)
    Returns forcing factor beween 0 and 1.

    Parameters
    ----------
    resp : NoneType,
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
        Irradiance (e.g., µmol photons m⁻² s⁻¹ or W m⁻²),
        must be consistent with the units used for alpha.
    """
    return np.tanh(alpha * I / Pmax)


def Monod_n_p(resp, NO3, NH4, P, R_no3, R_a, k_p):
    """
    Forcing function for growth inhibition based on bioavailable nitrogen and phosphorus.
    Based on Matano paper page 40, equation 9.

    This follows the Matano model formulation:
        beta_t = beta_NO3 + beta_a (NH4)

    Parameters
    ----------
    resp : NoneType
        Placeholder for NutMEG forcing function interface
    NO3 : float
        Nitrate concentration (mol/kg)
    NH4 : float
        Ammonium concentration (mol/kg)
    R_no3: float
        Monod half-saturation constant for NO3 uptake
    R_a : float
        Monod half-saturation constant for NH4 uptake
    """
    return min(Monod_nitrogen(NO3, NH4, R_no3, R_a), Monod(None, P, k_p))


def Monod_gsb(resp, NO3, NH4, P, H2S, R_no3, R_a, k_p, k_h2s):
    return min(Monod_nitrogen(NO3, NH4, R_no3, R_a), Monod(None, P, k_p), Monod(None, H2S, k_h2s))

def Monod_nitrogen(NO3, NH4, R_no3, R_a):
    beta_NO3 = NO3 / (R_no3 + NO3)
    beta_a = NH4 / (R_a + NH4)
    return beta_NO3 + beta_a


def Monod(resp, A, k):
    return A / (k + A)


def temperature_modifier(t, k_t):
    return np.exp(k_t * (t - 20))


def inhibition(resp, species, a_inh, species_inh):
    """
    Forcing function for growth inhibition based on H2S concentration.

    Parameters
    ----------
    species : float
        Concentration of the inhibiting species (mol/kg)
    a_inh : float
        Inhibition constant
    species_inh : float
        Inhibition constant for the species
    """
    return 0.5 * (1 - np.tanh(a_inh * (species - species_inh)))


def get_opnnf(R, rxn, Pm, I, k_l_opnnf, alpha, NO3, NH4, P, R_no3, R_a, k_p_opnnf, H2S, a_inh, H2S_inh, mmr, mgr, num=1e6, name='Non-Nitrogen-Fixing Oxygenic Phototroph'):
    """
    Create a NutMEG.horde object representing a non-nitrogen-fixing oxygenic phototroph.

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

    _phototroph = nm.horde(
        name, R, rxn, num,
        unit='cells',  # alternative units are not yet supported
        workoutID=False,
        E_synth=8e-10,
        respiration_kwargs={
            'rate_func': 'zeroth order',
            'max_metabolic_rate': mmr,
            'G_ATP': 'default',
            'G_net_pathway': -1000000,  # set arbitrarily high
            'G_C': 600000,              # J/mol of CO2 fixed
            'rate_constant_env': mmr,
        },
        CHNOPS_kwargs={
            'max_growth_rate': mgr,
            ##
            ## Adding forcing functions is currently quite complicated.
            ## Two attributes need to be adjusted in the host's CHNOPS attribute:
            ##
            ## CHNOPS_forcing_parameters:
            ##   dict of {name: (function, [required attributes])}
            ##
            ## CHNOPS_F_attrs:
            ##   dict mapping required attributes to values
            ##
            ## This structure accommodates multiple forcing functions
            ## depending on shared environmental attributes.
            ##
            'CHNOPS_forcing_parameters': {
                'Phi': (Monod_n_p, ['NO3', 'NH4', 'P', 'R_no3', 'R_a', 'k_p_opnnf']),
                # 'Platt': (Platt_tanh, ['alpha', 'Pmax', 'I']),
                'Light': (Monod, ['I', 'k_l_opnnf']),
                'Sulfur': (inhibition, ['H2S', 'a_inh', 'H2S_inh'])
            },
            'CHNOPS_F_attrs': {
                'NO3': NO3,
                'NH4': NH4,
                'P': P,
                'R_no3': R_no3,
                'R_a': R_a,
                'k_p_opnnf': k_p_opnnf,
                'alpha': alpha,
                'I': I,
                'k_l_opnnf': k_l_opnnf,
                'Pmax': Pm,
                'H2S': H2S,
                'a_inh': a_inh,
                'H2S_inh': H2S_inh
            }
        }
    )

    return _phototroph


def get_gsb(R, rxn, Pm, I, k_l_gsb, alpha, NO3, NH4, P, O2, R_no3, R_a, k_p_gsb, k_h2s_gsb, H2S, a_inh, O2_inh, mmr, mgr, num=1e6, name='Green Sulfur Bacterium'):
    _phototroph = nm.horde(
        name, R, rxn, num,
        unit='cells',  # alternative units are not yet supported
        workoutID=False,
        E_synth=8e-10,
        respiration_kwargs={
            'rate_func': 'zeroth order',
            'max_metabolic_rate': mmr,
            'G_ATP': 'default',
            'G_net_pathway': -1000000,  # set arbitrarily high
            'G_C': 600000,              # J/mol of CO2 fixed
            'rate_constant_env': mmr,
        },
        CHNOPS_kwargs={
            'max_growth_rate': mgr,
            'CHNOPS_forcing_parameters': {
                'Phi': (Monod_gsb, ['NO3', 'NH4', 'P', 'H2S', 'R_no3', 'R_a', 'k_p_gsb', 'k_h2s_gsb']),
                # 'Platt': (Platt_tanh, ['alpha', 'Pmax', 'I']),
                'Light': (Monod, ['I', 'k_l_gsb']),
                'Oxygen': (inhibition, ['O2', 'a_inh', 'O2_inh'])
            }, 
            'CHNOPS_F_attrs': {
                'NO3': NO3,
                'NH4': NH4,
                'P': P,
                'R_no3': R_no3,
                'R_a': R_a,
                'k_p_gsb': k_p_gsb,
                'alpha': alpha,
                'I': I,
                'k_l_gsb': k_l_gsb,
                'Pmax': Pm,
                'H2S': H2S,
                'k_h2s_gsb': k_h2s_gsb,
                'a_inh': a_inh,
                'O2': O2,
                'O2_inh': O2_inh
            }
        }
    )

    return _phototroph


def get_phototroph_rate(_phototroph, stepsize=600):
    """
    Returns growth rate of a horde using defined time step.
    """

    _phototroph.take_step(stepsize)

    return _phototroph.growth_rate


def main():
    # Load Lake Matano data from Google Sheets
    url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTgMvmTJ9LGCeC6vAJyHyh-X6fL3AHzKY9R0PuJLdTMGTE1qq7ZChWN2VL6qrtD8ib1r5l2UQyj6phf/pub?gid=1636861519&single=true&output=csv'
    
    print("Loading Lake Matano data...")
    m_raw = load_matano_data(url, interpolate=False) # Matano raw data at measured depths
    m_data = load_matano_data(url, interpolate=True) # Matano data interpolated to 1 m grid

    # Set up reactor
    R = nm.reactor('Matano_reactor', workoutID=False, pH=7.0)
    R, rxn = initial_conditions(R)
    rxn.update_molar_gibbs_from_quotient()
    print(r'ΔG =', rxn.molar_gibbs, 'J/mol')
    
    # Pm and alpha are platt model properties. The ones identified below are
    # global averages based on the dataset in Bouman et al., (2018).
    Pm = 3.1145  # mg C / mg Chl /h
    alpha = 0.04278  # uses units containing μmol photons m⁻² s⁻¹
    
    # ratio between g of Chl a and g of cells.
    CChl_to_Cbm = 0.01 # 0.003 to 0.055 (Middelburg 2019)
    Cbm_to_percell = 3e-13 # number of cells in 1g of biomass. Use nutmeg default (Higgins & Cockell 2020)
    
    # Maximum metabolic and growth rates
    mmr = Pm * CChl_to_Cbm * Cbm_to_percell / (12*3600)  # mol CO2 / cell / s
    mgr = Pm * CChl_to_Cbm / 3600  # growth rate in /s

    R_no3 = 2.5  # Half-saturation constant for NO3 (μM)
    R_a = 2.5  # Half-saturation constant for NH4+ (μM)
    k_p_opnnf = 0.014 # Half-saturation constant for P in non-nitrogen fixing oxygenic phototrophs (μM)
    k_t = 0.0693 # Temperature dependence (°C^-1)
    a_inh = 1e9 # Inhibition constant (mol/kg)^-1 (converted from 1e6 mmol/L^-1)
    H2S_inh = 1.0  # Inhibition constant for H2S (μM)
    k_l_opnnf = 100.0 # Light limitation constant for non-nitrogen fixing oxygenic phototrophs (μmol photons m⁻²) <- check units

    k_p_gsb = 1.6 # Phosphorus limitation for green sulfur bacteria (μM)
    k_h2s_gsb = 2.0 # Sulfide limitation for green sulfur bacteria (μM)
    O2_inh = 1.0 # Inhibition constant for O2 (μM)
    k_l_gsb = 1.0 # Light limitation constant for green sulfur bacteria (μmol photons m⁻²) <- check units

    # Calculate forcing factors for graphing

    # Nitrogen forcing factor (β_t)
    NO3_no_nan = np.nan_to_num(m_data['NO3'], nan=0.0)
    NH4_no_nan = np.nan_to_num(m_data['NH4'], nan=0.0) 
    F_N = Monod_nitrogen(NO3_no_nan, NH4_no_nan, R_no3, R_a)
    F_N[np.isnan(m_data['NO3']) & np.isnan(m_data['NH4'])] = np.nan  # hide depths with no N data at all

    # ---- OPNNF ----
    # Irradiance forcing factor
    F_E_opnnf = Platt_tanh(None, alpha, Pm, m_data['par'])

    # Phosphorus forcing factor
    F_P_opnnf = Monod(None, m_data['P'], k_p_opnnf)

    # Sulfur inhibition forcing factor
    F_S_inh_opnnf = inhibition(None, np.nan_to_num(m_data['H2S'], nan=0.0), a_inh, H2S_inh)

    # ---- GSB ----
    # Irradiance forcing factor
    F_E_gsb = Monod(None, m_data['par'], k_l_gsb)

    # Phosphorus forcing factor
    F_P_gsb = Monod(None, m_data['P'], k_p_gsb)

    # Sulfur forcing factor
    F_H2S_gsb = Monod(None, m_data['H2S'], k_h2s_gsb)

    # Oxygen inhibition forcing factor
    F_O2_inh_gsb = inhibition(None, np.nan_to_num(m_data['O2'], nan=0.0), a_inh, O2_inh)
    
    # Calculate phototroph growth rates (where we have nitrogen data)
    print("Calculating growth rates...")
    prod_opnnf = np.full(len(m_data['depth']), np.nan)
    prod_gsb = np.full(len(m_data['depth']), np.nan)
    
    for i in range(len(m_data['depth'])):
        # Only calculate if we have at least one nitrogen source
        if not (np.isnan(m_data['NO3'][i]) and np.isnan(m_data['NH4'][i])):
            # Replace NaN with 0 for calculation
            NO3_calc = 0 if np.isnan(m_data['NO3'][i]) else m_data['NO3'][i]
            NH4_calc = 0 if np.isnan(m_data['NH4'][i]) else m_data['NH4'][i]
            H2S_calc = 0 if np.isnan(m_data['H2S'][i]) else m_data['H2S'][i]
            
            try:
                opnnf = get_opnnf(R, rxn, Pm, m_data['par'][i], k_l_opnnf, alpha, NO3_calc, NH4_calc, m_data['P'][i], R_no3, R_a, k_p_opnnf, H2S_calc, a_inh, H2S_inh, mmr, mgr * temperature_modifier(m_data['temp'][i], k_t))
                prod_opnnf[i] = get_phototroph_rate(opnnf)
            except:
                prod_opnnf[i] = np.nan
            
            try:
                gsb = get_gsb(R, rxn, Pm, m_data['par'][i], k_l_gsb, alpha, NO3_calc, NH4_calc, m_data['P'][i], m_data['O2'][i], R_no3, R_a, k_p_gsb, k_h2s_gsb, H2S_calc, a_inh, O2_inh, mmr, mgr * temperature_modifier(m_data['temp'][i], k_t))
                prod_gsb[i] = get_phototroph_rate(gsb)
            except Exception as e:
                prod_gsb[i] = np.nan
                # print("error depth:", i, ": ", e)
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    
    # Row 1, Plot 1: Growth rate
    axes[0, 0].plot(prod_opnnf*1e6, m_data['depth'], label='Non-Nitrogen Fixing Oxygenic Phototrophs', color='red', linewidth=1)
    axes[0, 0].plot(prod_gsb*1e6, m_data['depth'], label='Green Sulfur Bacteria', color='green', linewidth=1)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_ylim(200, 0)
    axes[0, 0].set_xlabel('Growth Rate (×10⁶ s⁻¹)', fontsize=10)
    axes[0, 0].set_ylabel('Depth (m)', fontsize=10)
    axes[0, 0].tick_params(axis='both', labelsize=8)
    axes[0, 0].set_title('Phototroph Growth Rate', fontsize=13, fontweight='bold')
    axes[0, 0].legend(loc='lower right', fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    # Row 1, Plot 2: OPNNF Forcing factors
    axes[0, 1].plot(F_E_opnnf, m_data['depth'], label='F_I (irradiance)', linewidth=1, color='orange')
    axes[0, 1].plot(F_N, m_data['depth'], label='β_t (nitrogen availability)', linewidth=1, color='blue')
    axes[0, 1].plot(F_P_opnnf, m_data['depth'], label='F_P (phosphorus availability)', linewidth=1, color='green')
    axes[0, 1].plot(F_S_inh_opnnf, m_data['depth'], label='F_S (sulfur inhibition)', linewidth=1, color='red')
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlim(0, 1.05)
    axes[0, 1].set_ylim(200, 0)
    axes[0, 1].set_xlabel('Forcing Factor', fontsize=10)
    axes[0, 1].tick_params(axis='both', labelsize=8)
    axes[0, 1].set_title('OPNNF Forcing Factors', fontsize=13, fontweight='bold')
    axes[0, 1].legend(loc='lower left', fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)

    # Row 1, Plot 3: GSB Forcing factors
    axes[0, 2].plot(F_E_gsb, m_data['depth'], label='F_I (irradiance)', linewidth=1, color='orange')
    axes[0, 2].plot(F_N, m_data['depth'], label='β_t (total nitrogen availability)', linewidth=1, color='blue')
    axes[0, 2].plot(F_P_gsb, m_data['depth'], label='F_P (total phosphorus availability)', linewidth=1, color='green')
    axes[0, 2].plot(F_H2S_gsb, m_data['depth'], label='F_H2S (total sulfur availability)', linewidth=1, color='purple')
    axes[0, 2].plot(F_O2_inh_gsb, m_data['depth'], label='F_O2 (O2 inhibition)', linewidth=1, color='red')
    axes[0, 2].invert_yaxis()
    axes[0, 2].set_xlim(0, 1.05)
    axes[0, 2].set_ylim(200, 0)
    axes[0, 2].set_xlabel('Forcing Factor', fontsize=10)
    axes[0, 2].set_title('GSB Forcing Factors', fontsize=13, fontweight='bold')
    axes[0, 2].tick_params(axis='both', labelsize=8)
    axes[0, 2].legend(loc='lower right', fontsize=6)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2, Plot 1: PAR vs depth
    axes[1, 0].scatter(m_raw['par'], m_raw['depth'], color='none', edgecolors='orange', s=15, marker='o')
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_ylim(200, 0)
    axes[1, 0].set_xlabel('PAR (μmol photons m⁻² s⁻¹)', fontsize=10)
    axes[1, 0].set_ylabel('Depth (m)', fontsize=10)
    axes[1, 0].set_title('Light Extinction', fontsize=13, fontweight='bold')
    axes[1, 0].tick_params(axis='both', labelsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Row 2, Plot 2: Chemical species concentrations (NO3, NH4, P)
    axes[1, 1].scatter(m_raw['NO3'] * 100, m_raw['depth'], label='NO₃⁻ (μM) ×100', color='none', edgecolors='red', s=25, marker='D')
    axes[1, 1].scatter(m_raw['NH4'], m_raw['depth'], label='NH₄⁺ (μM)', color='none', edgecolors='mediumorchid', s=25, marker='o')
    axes[1, 1].scatter(m_raw['P'] * 100, m_raw['depth'], label='P (μM) ×100', color='none', edgecolors='blue', s=25, marker='P')
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlim(0, 650)
    axes[1, 1].set_ylim(200, 0)
    axes[1, 1].set_title('Chemical Species', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Concentration (μM)', fontsize=10)
    axes[1, 1].tick_params(axis='both', labelsize=8)
    axes[1, 1].legend(loc='upper right', fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Row 2, Plot 3: Chemical species concentrations (O2, H2S, SO4)
    axes[1, 2].scatter(m_raw['H2S'] * 1000, m_raw['depth'], label='H₂S (μM) ×1000', color='none', edgecolors='green', s=25, marker='X')
    axes[1, 2].scatter(m_raw['SO4'] * 10, m_raw['depth'], label='SO₄ (μM) ×10', color='none', edgecolors='lightseagreen', s=25, marker='X')
    axes[1, 2].scatter(m_raw['O2'], m_raw['depth'], label='O₂ (μM)', color='none', edgecolors='red', s=25, marker='P')
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_xlim(0, 400)
    axes[1, 2].set_ylim(200, 0)
    axes[1, 2].set_title('Chemical Species', fontsize=13, fontweight='bold')
    axes[1, 2].set_xlabel('Concentration (μM)', fontsize=10)
    axes[1, 2].tick_params(axis='both', labelsize=8)
    axes[1, 2].legend(loc='lower right', fontsize=7)
    axes[1, 2].grid(True, alpha=0.3)
    
    fig.suptitle('Predicted Primary Production in Lake Matano', fontsize=15, fontweight='bold')
    plt.tight_layout()

    filename = get_incremented_filename('matano_phototroph_growth', '.png')


    # Command line argument "--save" to save the plot instead of displaying it
    parser = argparse.ArgumentParser(description='Generate Lake Matano phototroph growth plots')
    parser.add_argument('--save', action='store_true', help='Save plot to file instead of displaying')
    args = parser.parse_args()

    if args.save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as '{filename}'\n")
    else:
        plt.show()

if __name__ == "__main__":
    main()