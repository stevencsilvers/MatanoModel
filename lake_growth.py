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

from dataclasses import dataclass
from typing import Dict, Tuple, List


from forcing_functions import (
    Monod,
    inhibition,
    light_opnf,
    Haldane,
    Monod_nitrogen,
    temperature_modifier,
)

from get_organisms import (
    get_opnnf,
    get_opnf,
    get_gsb,
    get_psb,
)


@dataclass
class LakeConfig:
    """Configuration for a lake model."""
    name: str
    params: Dict[str, Tuple[int, str]]
    max_depth: float
    vertical_resolution: float
    max_graphing_depth: float


LAKES = {
    'matano': LakeConfig(
        name='matano',
        params={
            'NH4': (2007, 'february'),
            'NO3': (2010, 'may'),
            'P': (2005, 'july'),
            'par': (2007, 'february'),
            'temp': (2004, 'september'),
            'H2S': (2007, 'february'),
            'O2': (2004, 'september')
        },
        max_depth=550,
        vertical_resolution=1.0,
        max_graphing_depth=200,
    ),
    'cadagno': LakeConfig(
        name='cadagno',
        params={
            'NH4': (1999, 'august'),
            'NO3': (1999, 'august'),
            'P': (1999, 'august'),
            'par': (1999, 'august'),
            'temp': (1999, 'august'),
            'H2S': (1999, ''),
            'O2': (1999, 'august')
        },
        max_depth=20,
        vertical_resolution=0.1,
        max_graphing_depth=20,
    )
}


# Pm and alpha are platt model properties. The ones identified below are
# global averages based on the dataset in Bouman et al., (2018).
Pm = 3.1145  # mg C / mg Chl /h
alpha = 0.04278  # uses units containing μmol photons m⁻² s⁻¹

# ratio between g of Chl a and g of cells.
CChl_to_Cbm = 0.01 # 0.003 to 0.055 (Middelburg 2019)
Cbm_to_percell = 3e-13 # number of g biomass in 1 cell. Use nutmeg default (Higgins & Cockell 2020)

# Maximum metabolic and growth rates
mmr = Pm * CChl_to_Cbm * Cbm_to_percell / (12*3600)  # mol CO2 / cell / s
mgr = Pm * CChl_to_Cbm / 3600  # growth rate in /s

# Values from Matano paper
mu_max_opnnf = 0.67 / 86400  # Maximum growth rate for non-nitrogen fixing oxygenic phototrophs (/s)
mu_max_gsb = 0.25 / 86400  # Maximum growth rate for green sulfur bacteria (/s)
mu_max_opnf = 0.23 / 86400  # Maximum growth rate for nitrogen-fixing oxygenic phototrophs (/s)

R_no3 = 2.5  # Half-saturation constant for NO3 (μM)
R_a = 2.5  # Half-saturation constant for NH4+ (μM)
k_p_opnnf = 0.014 # Half-saturation constant for P in non-nitrogen fixing oxygenic phototrophs (μM)
k_t = 0.0693 # Temperature dependence (°C^-1)
a_inh = 1e9 # Inhibition constant μM^-1 (converted from 1e6 mM^-1)
H2S_inh = 1.0  # Inhibition constant for H2S (μM)
k_l_opnnf = 100.0 # Light limitation constant for non-nitrogen fixing oxygenic phototrophs (μmol photons m⁻²) <- check units

k_p_gsb = 1.6 # Phosphorus limitation for green sulfur bacteria (μM)
k_h2s_gsb = 2.0 # Sulfide limitation for green sulfur bacteria (μM)
O2_inh = 1.0 # Inhibition constant for O2 (μM)
k_l_gsb = 1.0 # Light limitation constant for green sulfur bacteria (μmol photons m⁻²) <- check units

I_opt = 200 # Optimal irradiance for nitrogen-fixing oxygenic phototrophs (μmol photons m⁻²) <- check units
k_p_opnf = 0.05 # Phosphorus limitation for nitrogen-fixing oxygenic phototrophs (μM)

# From Gemerden 1974 - PSB
K_s_psb = 10 # Half-saturation constant for H2S uptake in PSB Chromatium weissei (μM)
K_i_psb = 700 # Inhibition constant for H2S in PSB Chromatium weissei (μM)

k_p_psb = 1.6 # From GSB
k_l_psb = 20.0 # ARBITRARY VALUE Light limitation constant for purple sulfur bacteria (μmol photons m⁻²) <- check units


def load_lake_data(lake, interpolate=False):
    """
    Load Lake Lake data from Google Sheets in long format.

    Parameters
    ----------
    lake : LakeConfig
        Configuration object for the lake model containing parameters, max depth, and resolution.
    interpolate : bool, optional
        If True, interpolate data to a 1 m grid from 0 to max_depth m.
        If False, return data on 0-max_depth m grid with NaN for unmeasured depths.

    Returns
    -------
    data : dict
        Dictionary containing 'depth' and parameter arrays.
    """
    csv_path = os.path.join(os.path.dirname(__file__), f"{lake.name}_data.csv")
    df_long = pd.read_csv(csv_path)

    # Create depth grid
    depths = np.linspace(0, lake.max_depth, int(lake.max_depth / lake.vertical_resolution) + 1)
    data = {'depth': depths}

    for param, ym in lake.params.items():
        if not isinstance(ym, (list, tuple)) or len(ym) != 2:
            raise ValueError("Each params entry must be a (year, month) pair")

        year, month = ym
        # Filter data for this parameter and its specific year
        param_data = df_long[df_long['parameter'] == param].copy()
        
        if year != '':
            # Filter for a specific year when provided
            param_data = param_data[param_data['year'] == year]

        if month != '':
            # Filter for a specific month when provided
            param_data = param_data[param_data['month'] == month]
        
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
                    depth_idx = int(round(row['depth_m'] / lake.vertical_resolution))
                    if 0 <= depth_idx < len(depths):
                        values[depth_idx] = row['value']
            
            data[param] = values
        else: # No data found for this parameter in year/month
            if month != '' and year != '':
                missing_desc = f"from {month} {year}"
            elif month != '' and year == '':
                missing_desc = f"from {month} (any year)"
            elif month == '' and year != '':
                missing_desc = f"from any month {year}"
            else:
                missing_desc = "from any month any year"

            print(f"load_lake_data(): Couldn't find {param} data {missing_desc}")
            return {}

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



def get_phototroph_rate(_phototroph, stepsize=600):
    """
    Returns growth rate of a horde using defined time step.
    """

    _phototroph.take_step(stepsize)

    return _phototroph.growth_rate


def calculate_depth_profiles(lake, data, R, rxn):
    # Calculate phototroph growth rates (where nitrogen data exists)
    print("Calculating growth rates...")

    prod = {'opnnf': np.full(len(data['depth']), np.nan),
            'sb': np.full(len(data['depth']), np.nan),
            'opnf': np.full(len(data['depth']), np.nan)}
    
    for i in range(len(data['depth'])):
        # Only calculate if we have at least one nitrogen source
        if not (np.isnan(data['NO3'][i]) and np.isnan(data['NH4'][i])):
            # Replace NaN with 0 for calculation
            NO3_calc = 0 if np.isnan(data['NO3'][i]) else data['NO3'][i]
            NH4_calc = 0 if np.isnan(data['NH4'][i]) else data['NH4'][i]
            H2S_calc = 0 if np.isnan(data['H2S'][i]) else data['H2S'][i]
            
            # OPNNF
            try:
                opnnf = get_opnnf(R, rxn, Pm, data['par'][i], k_l_opnnf, alpha, NO3_calc, NH4_calc, data['P'][i], H2S_calc, R_no3, R_a, k_p_opnnf, a_inh, H2S_inh, mmr, mu_max_opnnf * temperature_modifier(data['temp'][i], k_t))
                prod['opnnf'][i] = get_phototroph_rate(opnnf)
            except:
                prod['opnnf'][i] = np.nan
            
            # GSB
            try:
                gsb = get_gsb(R, rxn, Pm, data['par'][i], k_l_gsb, NO3_calc, NH4_calc, data['P'][i], data['O2'][i], H2S_calc, R_no3, R_a, k_p_gsb, k_h2s_gsb, a_inh, O2_inh, mmr, mu_max_gsb * temperature_modifier(data['temp'][i], k_t))
                prod['sb'][i] = get_phototroph_rate(gsb)
            except:
                prod['sb'][i] = np.nan

            # OPNF
            try:
                opnf = get_opnf(R, rxn, Pm, data['par'][i], I_opt, data['P'][i], data['O2'][i], k_p_opnf, a_inh, O2_inh, mmr, mu_max_opnf * temperature_modifier(data['temp'][i], k_t))
                prod['opnf'][i] = get_phototroph_rate(opnf)
            except Exception as e:
                prod['opnf'][i] = np.nan
    
    return prod


def calculate_forcing_factors(lake, data, organisms):
    F = {}

    NO3_no_nan = np.nan_to_num(data['NO3'], nan=0.0)
    NH4_no_nan = np.nan_to_num(data['NH4'], nan=0.0) 

    # Non-nitrogen-fixing oxygenic phototrophs
    if 'opnnf' in organisms:
        F['opnnf'] = {
            'I': [Monod(None, data['par'], k_l_opnnf), 'orange'], # Irradiance forcing factor (F_I)
            'N': [Monod_nitrogen(NO3_no_nan, NH4_no_nan, R_no3, R_a), 'blue'], # Nitrogen forcing factor (β_t)
            'P': [Monod(None, data['P'], k_p_opnnf), 'green'], # Phosphorus forcing factor (F_P)
            'H2S_inh': [inhibition(None, np.nan_to_num(data['H2S'], nan=0.0), a_inh, H2S_inh), 'mediumorchid'] # Sulfur inhibition forcing factor (F_S)'
        }
        F['opnnf']['N'][0][np.isnan(data['NO3']) & np.isnan(data['NH4'])] = np.nan  # hide depths with no N data at all
    
    # Nitrogen-fixing oxygenic phototrophs
    if 'opnf' in organisms:
        F_I_opnf = []
        for I in data['par']:
            F_I_opnf.append(light_opnf(None, I, I_opt))
        
        F['opnf'] = {
            'I': [F_I_opnf, 'orange'], # Irradiance forcing factor (F_I)
            'P': [Monod(None, data['P'], k_p_opnf), 'green'], # Phosphorus forcing factor (F_P)
            'O2_inh': [inhibition(None, np.nan_to_num(data['O2'], nan=0.0), a_inh, O2_inh), 'red'] # Oxygen inhibition forcing factor (F_O2)'
        }
    
    # Green sulfur bacteria
    if 'gsb' in organisms:
        F['gsb'] = {
            'I': [Monod(None, data['par'], k_l_gsb), 'orange'], # Irradiance forcing factor
            'N': [Monod_nitrogen(NO3_no_nan, NH4_no_nan, R_no3, R_a), 'blue'], # Nitrogen forcing factor (β_t)
            'P': [Monod(None, data['P'], k_p_gsb), 'green'], # Phosphorus forcing factor
            'H2S': [Monod(None, data['H2S'], k_h2s_gsb), 'mediumorchid'], # Sulfide forcing factor
            'O2_inh': [inhibition(None, np.nan_to_num(data['O2'], nan=0.0), a_inh, O2_inh), 'red'] # Oxygen inhibition forcing factor (F_O2)'
        }
        F['gsb']['N'][0][np.isnan(data['NO3']) & np.isnan(data['NH4'])] = np.nan  # hide depths with no N data at all
    
    if 'psb' in organisms:
        F['psb'] = {
            'I': [Monod(None, data['par'], k_l_psb), 'orange'], # Irradiance forcing factor
            'N': [Monod_nitrogen(NO3_no_nan, NH4_no_nan, R_no3, R_a), 'blue'], # Nitrogen forcing factor (β_t)
            'P': [Monod(None, data['P'], k_p_psb), 'green'], # Phosphorus forcing factor
            'H2S': [Haldane(None, data['H2S'], K_s_psb, K_i_psb), 'mediumorchid']
        }
        F['psb']['N'][0][np.isnan(data['NO3']) & np.isnan(data['NH4'])] = np.nan  # hide depths with no N data at all
    
    return F



def main():
    
    lake = LAKES['matano']

    organisms = ['opnnf', 'opnf', 'gsb']#, 'psb']
    
    
    print("Loading Lake data...")
    raw = load_lake_data(lake, interpolate=False) # Lake raw data at measured depths

    if raw == {}: # Quit if load_lake_data can't load all parameters
        return

    data = load_lake_data(lake, interpolate=True) # Lake data interpolated to grid of specified resolution

    # Set up reactor
    R = nm.reactor('Lake_reactor', workoutID=False, pH=7.0)
    R, rxn = initial_conditions(R)
    rxn.update_molar_gibbs_from_quotient()
    print(r'ΔG =', rxn.molar_gibbs, 'J/mol')


    F = calculate_forcing_factors(lake, data, organisms)


    prod = calculate_depth_profiles(lake, data, R, rxn)
    
    # Plot results
    fig, axes = plt.subplots(2, 4, figsize=(12, 8))
    
    # Row 1, Plot 1: Growth rate
    axes[0, 0].plot(prod['opnnf']*1e6, data['depth'], label='Non-Nitrogen-Fixing Oxygenic Phototrophs', color='red', linewidth=1)
    axes[0, 0].plot(prod['sb']*1e6, data['depth'], label=f'{"Green" if "gsb" in organisms else "Purple"} Sulfur Bacteria', color=f'{"green" if "gsb" in organisms else "purple"}', linewidth=1)
    axes[0, 0].plot(prod['opnf']*1e6, data['depth'], label='Nitrogen-Fixing Oxygenic Phototrophs', color='blue', linewidth=1)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_ylim(lake.max_graphing_depth, 0)
    axes[0, 0].set_xlabel('Growth Rate (×10⁶ s⁻¹)', fontsize=10)
    axes[0, 0].set_ylabel('Depth (m)', fontsize=10)
    axes[0, 0].tick_params(axis='both', labelsize=8)
    axes[0, 0].set_title('Phototroph Growth Rate', fontsize=13, fontweight='bold')
    axes[0, 0].legend(loc='lower right', fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    # Row 1, Plot 2: Chemical species concentrations (NO3, NH4, P)
    axes[0, 1].scatter(raw['NO3'] * 100, raw['depth'], label='NO₃⁻ (μM) ×100', color='none', edgecolors='red', s=25, marker='D')
    axes[0, 1].scatter(raw['NH4'], raw['depth'], label='NH₄⁺ (μM)', color='none', edgecolors='mediumorchid', s=25, marker='o')
    axes[0, 1].scatter(raw['P'] * 100, raw['depth'], label='P (μM) ×100', color='none', edgecolors='blue', s=25, marker='P')
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlim(0, 650)
    axes[0, 1].set_ylim(lake.max_graphing_depth, 0)
    axes[0, 1].set_title('Chemical Species', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Concentration (μM)', fontsize=10)
    axes[0, 1].tick_params(axis='both', labelsize=8)
    axes[0, 1].legend(loc='upper right', fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Row 1, Plot 3: Chemical species concentrations (O2, H2S)
    axes[0, 2].scatter(raw['H2S'] * 1000, raw['depth'], label='H₂S (μM) ×1000', color='none', edgecolors='green', s=25, marker='X') # MATANO
    axes[0, 2].scatter(raw['O2'], raw['depth'], label='O₂ (μM)', color='none', edgecolors='lightseagreen', s=25, marker='P')
    axes[0, 2].invert_yaxis()
    axes[0, 2].set_xlim(0, 400) # MATANO
    axes[0, 2].set_ylim(lake.max_graphing_depth, 0)
    axes[0, 2].set_title('Chemical Species', fontsize=13, fontweight='bold')
    axes[0, 2].set_xlabel('Concentration (μM)', fontsize=10)
    axes[0, 2].tick_params(axis='both', labelsize=8)
    axes[0, 2].legend(loc='lower right', fontsize=7)
    axes[0, 2].grid(True, alpha=0.3)

    # Row 1, Plot 4: PAR vs depth
    axes[0, 3].scatter(raw['par'], raw['depth'], color='none', edgecolors='orange', s=15, marker='o')
    axes[0, 3].invert_yaxis()
    axes[0, 3].set_xscale('log')
    axes[0, 3].set_ylim(lake.max_graphing_depth, 0)
    axes[0, 3].set_xlabel('PAR (μmol photons m⁻² s⁻¹)', fontsize=10)
    axes[0, 3].set_ylabel('Depth (m)', fontsize=10)
    axes[0, 3].set_title('Light Extinction', fontsize=13, fontweight='bold')
    axes[0, 3].tick_params(axis='both', labelsize=8)
    axes[0, 3].grid(True, alpha=0.3)

    # Row 2, Plot 4: (Hide)
    axes[1, 3].axis('off')

    for i in range(4):
        if i < len(organisms):
            for key in F[organisms[i]]:
                axes[1, i].plot(F[organisms[i]][key][0], data['depth'], label=f'F_{key}', linewidth=1, color=F[organisms[i]][key][1])
            axes[1, i].invert_yaxis()
            axes[1, i].set_xlim(-0.03, 1.03)
            axes[1, i].set_ylim(lake.max_graphing_depth, 0)
            axes[1, i].set_xlabel('Forcing Factor', fontsize=10)
            axes[1, i].tick_params(axis='both', labelsize=8)
            axes[1, i].set_title(f'{organisms[i].upper()} Forcing Factors', fontsize=13, fontweight='bold')
            axes[1, i].legend(loc='lower right', fontsize=6)
            axes[1, i].grid(True, alpha=0.3)
        else:
            axes[1, i].axis('off')

    '''
    # Row 1, Plot 2: OPNNF Forcing factors
    axes[0, 1].plot(F['opnnf']['I'], data['depth'], label='F_I (irradiance)', linewidth=1, color='orange')
    axes[0, 1].plot(F['opnnf']['N'], data['depth'], label='β_t (nitrogen availability)', linewidth=1, color='blue')
    axes[0, 1].plot(F['opnnf']['P'], data['depth'], label='F_P (phosphorus availability)', linewidth=1, color='green')
    axes[0, 1].plot(F['opnnf']['H2S_inh'], data['depth'], label='F_S (sulfur inhibition)', linewidth=1, color='red')
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlim(-0.03, 1.03)
    axes[0, 1].set_ylim(lake.max_graphing_depth, 0)
    axes[0, 1].set_xlabel('Forcing Factor', fontsize=10)
    axes[0, 1].tick_params(axis='both', labelsize=8)
    axes[0, 1].set_title('OPNNF Forcing Factors', fontsize=13, fontweight='bold')
    axes[0, 1].legend(loc='lower left', fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)

    # Row 1, Plot 3: GSB Forcing factors
    axes[0, 2].plot(F['gsb']['I'], data['depth'], label='F_I (irradiance)', linewidth=1, color='orange')
    axes[0, 2].plot(F['gsb']['N'], data['depth'], label='β_t (total nitrogen availability)', linewidth=1, color='blue')
    axes[0, 2].plot(F['gsb']['P'], data['depth'], label='F_P (total phosphorus availability)', linewidth=1, color='green')
    if 'gsb' in organisms:
        axes[0, 2].plot(F['gsb']['O2_inh'], data['depth'], label='F_O2 (O2 inhibition)', linewidth=1, color='red')
        axes[0, 2].plot(F['gsb']['H2S'], data['depth'], label='F_H2S (total sulfur availability)', linewidth=1, color='purple')
    else:
        axes[0, 2].plot(F['psb']['H2S'], data['depth'], label='F_H2S (sulfur limitation + inhibition)', linewidth=1, color='purple')

    axes[0, 2].invert_yaxis()
    axes[0, 2].set_xlim(-0.03, 1.03)
    axes[0, 2].set_ylim(lake.max_graphing_depth, 0)
    axes[0, 2].set_xlabel('Forcing Factor', fontsize=10)
    axes[0, 2].set_title(f'{"GSB" if "gsb" in organisms else "PSB"} Forcing Factors', fontsize=13, fontweight='bold')
    axes[0, 2].tick_params(axis='both', labelsize=8)
    axes[0, 2].legend(loc='lower right', fontsize=6)
    axes[0, 2].grid(True, alpha=0.3)

    # Row 1, Plot 4: OPNF Forcing factors
    axes[0, 3].plot(F['opnf']['I'], data['depth'], label='F_I (irradiance)', linewidth=1, color='orange')
    axes[0, 3].plot(F['opnf']['P'], data['depth'], label='F_P (total phosphorus availability)', linewidth=1, color='green')
    axes[0, 3].plot(F['opnf']['O2_inh'], data['depth'], label='F_O2 (O2 inhibition)', linewidth=1, color='red')
    axes[0, 3].invert_yaxis()
    axes[0, 3].set_xlim(-0.03, 1.03)
    axes[0, 3].set_ylim(lake.max_graphing_depth, 0)
    axes[0, 3].set_xlabel('Forcing Factor', fontsize=10)
    axes[0, 3].set_title('OPNF Forcing Factors', fontsize=13, fontweight='bold')
    axes[0, 3].tick_params(axis='both', labelsize=8)
    axes[0, 3].legend(loc='lower right', fontsize=6)
    axes[0, 3].grid(True, alpha=0.3)
    '''
    
    fig.suptitle(f'Predicted Primary Production in Lake {lake.name.capitalize()}', fontsize=15, fontweight='bold')
    plt.tight_layout()

    filename = get_incremented_filename(f'{lake.name}_phototroph_growth', '.png', directory='matplotlib')

    # Command line argument "--save" to save the plot instead of displaying it
    parser = argparse.ArgumentParser(description='Generate lake growth plots')
    parser.add_argument('--save', action='store_true', help='Save plot to file instead of displaying')
    args = parser.parse_args()

    if args.save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as '{filename}'\n")
    else:
        plt.show()

if __name__ == "__main__":
    main()