import os, math, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(__file__)+'/../NutMEG')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ssl

# Import NutMEG
import NutMEG as nm

# SSL fix for Google Sheets
ssl._create_default_https_context = ssl._create_unverified_context


def load_matano_data(url, interpolate=True):
    """
    Load Lake Matano data from Google Sheets in long format.

    Parameters
    ----------
    url : str
        URL of the CSV containing parameter, depth_m, and value columns.
    interpolate : bool, optional
        If True (default), interpolate data to a 1 m grid from 0 to 550 m.
        If False, return only measured depths without interpolation.

    Returns
    -------
    data : dict
        Dictionary containing 'depth' and parameter arrays.
    """
    df_long = pd.read_csv(url)

    # # TEST
    df_long.loc[(df_long['parameter'] == 'NO3_umol_L') & (df_long['depth_m'] == 9), 'value'] = 0.0
    df_long.loc[(df_long['parameter'] == 'NO3_umol_L') & (df_long['depth_m'] == 20), 'value'] = 0.0
    df_long.loc[(df_long['parameter'] == 'NO3_umol_L') & (df_long['depth_m'] == 25), 'value'] = 0.0
    df_long.loc[(df_long['parameter'] == 'NO3_umol_L') & (df_long['depth_m'] == 40), 'value'] = 0.0
    # TEST

    # Filter to only include rows where year is NaN or 2005
    df_long = df_long[(df_long['year'].isna()) | (df_long['year'] == 2005)]

    # Parameters we want to extract
    params = ['NH4_umol_L', 'NO3_umol_L', 'P_umol_L']

    # Interpolated: return interpolated concentrations for each chemical species
    if interpolate:
        # Create depth grid (every 1 meter from 0 to 550)
        depths = np.arange(0, 551, 1)

        # Dictionary to store interpolated data
        data = {'depth': depths}

        for param in params:
            param_data = df_long[df_long['parameter'] == param].copy()
            param_data = param_data.sort_values('depth_m')

            if len(param_data) > 0:
                interp_values = np.interp(
                    depths,
                    param_data['depth_m'],
                    param_data['value'],
                    left=np.nan,
                    right=np.nan
                )
                data[param] = interp_values
            else:
                data[param] = np.full(len(depths), np.nan)

        return data

    # Non-interpolated: return only measured depths for each chemical species
    df_filtered = df_long[df_long['parameter'].isin(params)].copy()
    df_filtered = df_filtered.dropna(subset=['depth_m', 'value'])

    if df_filtered.empty:
        empty_arr = np.array([])
        return {'depth': empty_arr, **{p: empty_arr for p in params}}

    depths = np.sort(df_filtered['depth_m'].unique())
    data = {'depth': depths}

    for param in params:
        param_series = (
            df_filtered[df_filtered['parameter'] == param]
            .groupby('depth_m')['value']
            .mean()
        )
        data[param] = param_series.reindex(depths).to_numpy()

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
        Decay factor in irradiance (default is -0.1, from Middelburg 2019 example)
    """
    return I0*np.exp(k*d)


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


def Monod_n_p(resp, NO3, NH4, P, R_n, R_a, k_popnf):
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
    R_n : float
        Monod half-saturation constant for NO3 uptake
    R_a : float
        Monod half-saturation constant for NH4 uptake
    """
    return min(Monod_nitrogen(NO3, NH4, R_n, R_a), Monod_phosphorus(P, k_popnf))


def Monod_nitrogen(NO3, NH4, R_n, R_a):
    beta_NO3 = NO3 / (R_n + NO3)
    beta_a = NH4 / (R_a + NH4)
    return beta_NO3 + beta_a


def Monod_phosphorus(P, k_popnf):
    return P / (k_popnf + P)


def get_phototroph(R, rxn, Pm, I, alpha, NO3, NH4, P, R_n, R_a, k_popnf, mmr, mgr, num=1e6, name='Phototroph'):
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
                'Monod': (Monod_n_p, ['NO3', 'NH4', 'P', 'R_n', 'R_a', 'k_popnf']),
                'Platt': (Platt_tanh, ['alpha', 'Pmax', 'I'])
            },
            'CHNOPS_F_attrs': {
                'NO3': NO3,
                'NH4': NH4,
                'P': P,
                'R_n': R_n,
                'R_a': R_a,
                'k_popnf': k_popnf,
                'alpha': alpha,
                'I': I,
                'Pmax': Pm,
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
    matano_data = load_matano_data(url, interpolate=True)
    matano_raw = load_matano_data(url, interpolate=False)
    depths = matano_data['depth']
    NH4_data = matano_data['NH4_umol_L']  # NH4 in μmol/L
    NO3_data = matano_data['NO3_umol_L']  # NO3 in μmol/L
    P_data = matano_data['P_umol_L']      # P in μmol/L
    raw_depths = matano_raw['depth']
    raw_NH4 = matano_raw['NH4_umol_L']
    raw_NO3 = matano_raw['NO3_umol_L']
    raw_P = matano_raw['P_umol_L']

    # Convert from μmol/L to mol/kg (1 μmol/L = 1e-6 mol/kg for water)
    NH4_molkg = NH4_data * 1e-6
    NO3_molkg = NO3_data * 1e-6
    P_molkg = P_data * 1e-6
    
    # Set up reactor
    R = nm.reactor('Matano_reactor', workoutID=False, pH=7.0)
    R, rxn = initial_conditions(R)
    rxn.update_molar_gibbs_from_quotient()
    print(r'ΔG =', rxn.molar_gibbs, 'J/mol')
    
    # Pm and alpha are platt model properties. The ones identified below are
    # global averages based on the dataset in Bouman et al., (2018).
    Pm = 3.1145  # mg C / mg Chl /h
    alpha = 0.04278
    
    # ratio between g of Chl a and g of cells.
    CChl_to_Cbm = 0.01 # 0.003 to 0.055 (Middelburg 2019)
    Cbm_to_percell = 3e-13 # number of cells in 1g of biomass. Use nutmeg default (Higgins & Cockell 2020)
    
    # Maximum metabolic and growth rates
    mmr = Pm * CChl_to_Cbm * Cbm_to_percell / (12*3600)  # mol CO2 / cell / s
    mgr = Pm * CChl_to_Cbm / 3600  # growth rate in /s
    
    # Parameters from Matano model paper
    I_0 = 283  # Surface irradiance (W/m²) 1300 umol Eins m^-2
    R_n = 2.5e-6  # Half-saturation constant for NO3 (mol/kg)
    R_a = 2.5e-6  # Half-saturation constant for NH4 (mol/kg)
    k_popnf = 1e-8 # Half-saturation constant for P (mol/kg)
    k = -0.06 # PAR extinction coefficient (m^-1)

    # Calculate irradiance at each depth
    Idepths = Idepth(I_0, depths, k)
    
    # Calculate array of irradiance forcing factors with depth
    F_E = Platt_tanh(None, alpha, Pm, Idepths)

    # Calculate array of nitrogen forcing factors with depth (just for graphing)
    F_N = Monod_nitrogen(np.nan_to_num(NO3_molkg, nan=0.0), np.nan_to_num(NH4_molkg, nan=0.0), R_n, R_a)
    F_N[np.isnan(NO3_molkg) & np.isnan(NH4_molkg)] = np.nan  # hide depths with no N data at all

    # Calculate array of phosphorus forcing factors with depth (just for graphing)
    F_P = Monod_phosphorus(P_molkg, k_popnf)
    
    # Calculate phototroph growth rates (where we have nitrogen data)
    print("Calculating growth rates...")
    Prod = np.full(len(depths), np.nan)
    
    for i, (depth, NO3, NH4, P, I) in enumerate(zip(depths, NO3_molkg, NH4_molkg, P_molkg, Idepths)):
        # Only calculate if we have at least one nitrogen source
        if not (np.isnan(NO3) and np.isnan(NH4)):
            # Replace NaN with 0 for calculation
            NO3_calc = 0 if np.isnan(NO3) else NO3
            NH4_calc = 0 if np.isnan(NH4) else NH4
            
            try:
                phototroph = get_phototroph(R, rxn, Pm, I, alpha, NO3_calc, NH4_calc, P, R_n, R_a, k_popnf, mmr, mgr)
                Prod[i] = get_phototroph_rate(phototroph)

            except:
                Prod[i] = np.nan
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    
    # Plot 1: Forcing factors (light, nitrogen, phosphorus)
    axes[0].plot(F_E, depths, label='F_I (irradiance)', linewidth=2, color='orange')
    axes[0].plot(F_N, depths, label='β_t (total nitrogen availability)', linewidth=2, color='blue')
    axes[0].plot(F_P, depths, label='F_P (total phosphorus availability)', linewidth=2, color='green')
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.75)
    axes[0].set_ylim(200, 0)
    axes[0].set_xlabel('Forcing Factor', fontsize=12)
    axes[0].set_ylabel('Depth (m)', fontsize=12)
    axes[0].set_title('Individual Forcing Factors', fontsize=13, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: NO3-, NH4+, and P concentrations
    axes[1].scatter(raw_NO3 * 100, raw_depths, label='NO₃⁻ (μmol/L) ×100', color='red', s=30, marker='D')
    axes[1].scatter(raw_NH4, raw_depths, label='NH₄⁺ (μmol/L)', color='purple', s=30, marker='X')
    axes[1].scatter(raw_P * 100, raw_depths, label='P (μmol/L) ×100', color='blue', s=30, marker='+')
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 650)
    axes[1].set_ylim(200, 0)
    axes[1].set_title('Nitrogen and Phosphorus Compounds', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Concentration (μmol/L)', fontsize=12)
    axes[1].set_ylabel('Depth (m)', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Growth rate
    axes[2].plot(Prod*1e6, depths, color='green', linewidth=2, label='Growth rate')
    axes[2].invert_yaxis()
    # axes[2].set_xlim(0,)
    axes[2].set_ylim(200, 0)
    axes[2].set_xlabel('Growth Rate (×10⁶ s⁻¹)', fontsize=12)
    axes[2].set_ylabel('Depth (m)', fontsize=12)
    axes[2].set_title('Phototroph Growth Rate', fontsize=13, fontweight='bold')
    axes[2].legend(loc='lower right')
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle('Primary Production Model - Lake Matano (NO₃⁻ + NH₄⁺)', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    filename = get_incremented_filename('matano_phototroph_growth', '.png')


    save = False

    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as '{filename}'\n")
    else:
        plt.show()

if __name__ == "__main__":
    main()