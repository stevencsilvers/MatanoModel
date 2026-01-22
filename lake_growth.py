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


# ================================================================================================================================
# LAKE CONFIGURATION
# ================================================================================================================================

@dataclass
class LakeConfig:
    """Configuration for a lake model."""
    name: str
    params: Dict[str, Tuple[int, str]]
    max_depth: float
    vertical_resolution: float
    max_graphing_depth: float
    organisms: List[str]


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
        organisms = ['opnnf', 'gsb', 'opnf']
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
        organisms = ['opnnf', 'psb', 'opnf']
    )
}


# ================================================================================================================================
# CONSTANTS AND PARAMETERS
# ================================================================================================================================

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

# Values from Kanke et al. "Modeling the biogeochemical cycles in Lake Matano, a modern analog for ancient ferruginous oceans"
mu_max_opnnf = 0.67 / 86400  # Maximum growth rate for non-nitrogen fixing oxygenic phototrophs (/s)
mu_max_opnf = 0.23 / 86400  # Maximum growth rate for nitrogen-fixing oxygenic phototrophs (/s)
mu_max_gsb = 0.25 / 86400  # Maximum growth rate for green sulfur bacteria (/s)

R_no3 = 2.5  # Half-saturation constant for NO3 (μM)
R_a = 2.5  # Half-saturation constant for NH4+ (μM)
k_p_opnnf = 0.014 # Half-saturation constant for P in non-nitrogen-fixing oxygenic phototrophs (μM)
k_t = 0.0693 # Temperature dependence (°C^-1)
a_inh = 1e9 # Inhibition constant μM^-1 (converted from 1e6 mM^-1)
H2S_inh = 1.0  # Inhibition constant for H2S (μM)
k_l_opnnf = 100.0 # Light limitation constant for non-nitrogen fixing oxygenic phototrophs (μmol photons m⁻² s⁻¹)

k_p_gsb = 1.6 # Phosphorus limitation for green sulfur bacteria (μM)
k_h2s_gsb = 2.0 # Sulfide limitation for green sulfur bacteria (μM)
k_l_gsb = 1.0 # Light limitation constant for green sulfur bacteria (μmol photons m⁻² s⁻¹)
O2_inh = 1.0 # Inhibition constant for O2 (μM)

I_opt = 200 # Optimal irradiance for nitrogen-fixing oxygenic phototrophs (μmol photons m⁻² s⁻¹)
k_p_opnf = 0.05 # Phosphorus limitation for nitrogen-fixing oxygenic phototrophs (μM)

# Values from Gemerden 1974 for purple sulfur bacteria Chromatium weissei
mu_max_psb = 0.05 / 3600 # Maximum growth rate for PSB (/s)

K_s_psb = 10 # Half-saturation constant for H2S uptake in PSB (μM)
K_i_psb = 700 # Inhibition constant for H2S in PSB (μM)

k_p_psb = 1.6 # From GSB
k_l_psb = 20.0 # (Arbitrary value) Light limitation constant for purple sulfur bacteria (μmol photons m⁻² s⁻¹)


# ================================================================================================================================
# FUNCTIONS
# ================================================================================================================================

def load_lake_data(lake, interpolate=False):
    """
    Load lake data from csv in long format.

    Parameters
    ----------
    lake : LakeConfig
        Configuration object for the lake model containing parameters, max depth, and resolution.
    interpolate : bool, optional
        If True, interpolate data to a grid from 0 to max depth, with specified vertical resolution.
        If False, return data on 0 to max depth grid with NaN for unmeasured depths.

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


def calculate_growth_profiles(lake, data, R, rxn):
    """
    Calculate growth profiles for each organism in the lake by depth.
    Returns dictionary where each key corresponds to an organism, each containing growth curve, color, and label.
    """
    print("Calculating growth rates...")

    prod = {}
    
    for org in lake.organisms:
        prod[org] = {}
        prod[org]['growth'] = np.full(len(data['depth']), np.nan)
    
    for i in range(len(data['depth'])):
        # OPNNF
        if 'opnnf' in lake.organisms:
            prod['opnnf']['color'] = 'red'
            prod['opnnf']['label'] = 'Non-Nitrogen-Fixing Oxygenic Phototrophs'
            try:
                opnnf = get_opnnf(R, rxn, Pm, data['par'][i], k_l_opnnf, alpha, data['NO3'][i], data['NH4'][i], data['P'][i], data['H2S'][i], R_no3, R_a, k_p_opnnf, a_inh, H2S_inh, mmr, mu_max_opnnf * temperature_modifier(data['temp'][i], k_t))
                prod['opnnf']['growth'][i] = get_phototroph_rate(opnnf)
            except:
                prod['opnnf']['growth'][i] = np.nan

        # OPNF
        if 'opnf' in lake.organisms:
            prod['opnf']['color'] = 'blue'
            prod['opnf']['label'] = 'Nitrogen-Fixing Oxygenic Phototrophs'
            try:
                opnf = get_opnf(R, rxn, Pm, data['par'][i], I_opt, data['P'][i], data['O2'][i], k_p_opnf, a_inh, O2_inh, mmr, mu_max_opnf * temperature_modifier(data['temp'][i], k_t))
                prod['opnf']['growth'][i] = get_phototroph_rate(opnf)
            except Exception as e:
                prod['opnf']['growth'][i] = np.nan
        
        # GSB
        if 'gsb' in lake.organisms:
            prod['gsb']['color'] = 'green'
            prod['gsb']['label'] = 'Green Sulfur Bacteria'
            try:
                gsb = get_gsb(R, rxn, Pm, data['par'][i], k_l_gsb, data['NO3'][i], data['NH4'][i], data['P'][i], data['O2'][i], data['H2S'][i], R_no3, R_a, k_p_gsb, k_h2s_gsb, a_inh, O2_inh, mmr, mu_max_gsb * temperature_modifier(data['temp'][i], k_t))
                prod['gsb']['growth'][i] = get_phototroph_rate(gsb)
            except:
                prod['gsb']['growth'][i] = np.nan
        
        # PSB
        if 'psb' in lake.organisms:
            prod['psb']['color'] = 'mediumorchid'
            prod['psb']['label'] = 'Purple Sulfur Bacteria'
            try:
                psb = get_psb(R, rxn, Pm, data['par'][i], k_l_psb, data['NO3'][i], data['NH4'][i], data['P'][i], data['H2S'][i], R_no3, R_a, k_p_psb, K_s_psb, K_i_psb, mmr, mu_max_psb * temperature_modifier(data['temp'][i], k_t))
                prod['psb']['growth'][i] = get_phototroph_rate(psb)
            except:
                prod['psb']['growth'][i] = np.nan
    
    return prod


def calculate_forcing_factors(lake, data):
    """
    Calculate forcing factors for each organism in the lake by depth, for plotting purposes only.
    """
    F = {}

    # Non-nitrogen-fixing oxygenic phototrophs (OPNNF)
    if 'opnnf' in lake.organisms:
        F['opnnf'] = {
            'I': [Monod(None, data['par'], k_l_opnnf), 'orange'], # Irradiance forcing factor (F_I)
            'N': [Monod_nitrogen(data['NO3'], data['NH4'], R_no3, R_a), 'blue'], # Nitrogen forcing factor (β_t)
            'P': [Monod(None, data['P'], k_p_opnnf), 'green'], # Phosphorus forcing factor (F_P)
            'H2S_inh': [inhibition(None, np.nan_to_num(data['H2S'], nan=0.0), a_inh, H2S_inh), 'mediumorchid'] # Sulfur inhibition forcing factor (F_S)'
        }
    
    # Nitrogen-fixing oxygenic phototrophs (OPNF)
    if 'opnf' in lake.organisms:
        F['opnf'] = {
            'I': [light_opnf(None, data['par'], I_opt), 'orange'], # Irradiance forcing factor (F_I)
            'P': [Monod(None, data['P'], k_p_opnf), 'green'], # Phosphorus forcing factor (F_P)
            'O2_inh': [inhibition(None, np.nan_to_num(data['O2'], nan=0.0), a_inh, O2_inh), 'red'] # Oxygen inhibition forcing factor (F_O2)'
        }
    
    # Green sulfur bacteria (GSB)
    if 'gsb' in lake.organisms:
        F['gsb'] = {
            'I': [Monod(None, data['par'], k_l_gsb), 'orange'], # Irradiance forcing factor
            'N': [Monod_nitrogen(data['NO3'], data['NH4'], R_no3, R_a), 'blue'], # Nitrogen forcing factor (β_t)
            'P': [Monod(None, data['P'], k_p_gsb), 'green'], # Phosphorus forcing factor
            'H2S': [Monod(None, data['H2S'], k_h2s_gsb), 'mediumorchid'], # Sulfide forcing factor
            'O2_inh': [inhibition(None, np.nan_to_num(data['O2'], nan=0.0), a_inh, O2_inh), 'red'] # Oxygen inhibition forcing factor (F_O2)'
        }
    
    # Purple sulfur bacteria (PSB)
    if 'psb' in lake.organisms:
        F['psb'] = {
            'I': [Monod(None, data['par'], k_l_psb), 'orange'], # Irradiance forcing factor
            'N': [Monod_nitrogen(data['NO3'], data['NH4'], R_no3, R_a), 'blue'], # Nitrogen forcing factor (β_t)
            'P': [Monod(None, data['P'], k_p_psb), 'green'], # Phosphorus forcing factor
            'H2S': [Haldane(None, data['H2S'], K_s_psb, K_i_psb), 'mediumorchid']
        }
    
    return F


def calculate_scaling_factors(*species_arrays):
    """
    Determine optimal power-of-10 scaling factors for multiple chemical species arrays
    so they display on comparable scales.
    
    Finds the species with the highest maximum concentration, then calculates scaling
    factors (as powers of 10) for the others to match it.
    
    Returns
    -------
    scaling_factors : list of float
        List of scaling multipliers (1, 10, 100, 1000, etc.) matching input order.
        The species with the highest max concentration will have a multiplier of 1.
    """
    if len(species_arrays) < 2 or len(species_arrays) > 3:
        raise ValueError("Function accepts 2 or 3 species arrays")
    
    # Get max value for each species (ignoring NaN)
    max_values = [np.nanmax(arr) for arr in species_arrays]
    
    # Handle edge cases (all NaN or zero)
    if all(np.isnan(mv) or mv == 0 for mv in max_values):
        return [1.0] * len(species_arrays)
    
    # Find the species with the highest concentration
    reference_max = max(mv for mv in max_values if not np.isnan(mv) and mv != 0)
    
    scaling_factors = []
    for max_val in max_values:
        if np.isnan(max_val) or max_val == 0:
            # If this species has no data, don't scale it
            scaling_factors.append(1.0)
        else:
            # Calculate how much we need to scale THIS species to match the reference
            # If max_val is much smaller than reference_max, ratio will be > 1
            ratio = reference_max / max_val
            
            # If already close to 1:1, no scaling needed
            if 0.3 <= ratio <= 3:
                scaling_factors.append(1.0)
            else:
                # Find the power of 10 that gets us closest to matching the reference
                power = np.log10(ratio)
                power_rounded = round(power)
                multiplier = 10.0 ** power_rounded
                # Ensure we only scale UP (never below 1)
                scaling_factors.append(max(1.0, multiplier))
    
    return scaling_factors


def format_scale_label(scale_factor):
    """
    Format a scale factor for use in plot labels.
    """
    if scale_factor == 1:
        return ""
    else:
        return f" ×{int(scale_factor)}"


def generate_plots(lake, data, raw, prod, F):
    """
    Generates matplotlib plots for phototroph growth rates, chemical species,
    light extinction, and forcing factors.

    Parameters
    ----------
    lake : LakeConfig
        Configuration object for the lake model.
    data : dict
        Dictionary containing interpolated lake data.
    raw : dict
        Dictionary containing raw lake data.
    prod : dict
        Dictionary containing phototroph growth rates with depth, as well as colors and labels.
    F : dict
        Dictionary containing forcing factors with depth for each organism.
    """
    fig, axes = plt.subplots(2, 4, figsize=(12, 8))
    
    # Row 1, Plot 1: Growth rate
    for key in prod:
        axes[0, 0].plot(prod[key]['growth']*1e6, data['depth'], label=prod[key]['label'], color=prod[key]['color'], linewidth=1)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_ylim(lake.max_graphing_depth, 0)
    axes[0, 0].set_xlabel('Growth Rate (×10⁶ s⁻¹)', fontsize=10)
    axes[0, 0].set_ylabel('Depth (m)', fontsize=10)
    axes[0, 0].tick_params(axis='both', labelsize=8)
    axes[0, 0].set_title('Phototroph Growth Rate', fontsize=13, fontweight='bold')
    axes[0, 0].legend(loc='lower right', fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    # Row 1, Plot 2: Chemical species concentrations (NO3, NH4, P)
    NO3_scale, NH4_scale, P_scale = calculate_scaling_factors(raw['NO3'], raw['NH4'], raw['P'])
    axes[0, 1].scatter(raw['NO3'] * NO3_scale, raw['depth'], label=f'NO₃⁻ (μM){format_scale_label(NO3_scale)}', color='none', edgecolors='red', s=25, marker='D')
    axes[0, 1].scatter(raw['NH4'] * NH4_scale, raw['depth'], label=f'NH₄⁺ (μM){format_scale_label(NH4_scale)}', color='none', edgecolors='mediumorchid', s=25, marker='o')
    axes[0, 1].scatter(raw['P'] * P_scale, raw['depth'], label=f'P (μM){format_scale_label(P_scale)}', color='none', edgecolors='blue', s=25, marker='P')
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlim(left=0)
    axes[0, 1].set_ylim(lake.max_graphing_depth, 0)
    axes[0, 1].set_title('Chemical Species', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Concentration (μM)', fontsize=10)
    axes[0, 1].tick_params(axis='both', labelsize=8)
    axes[0, 1].legend(loc='upper right', fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Row 1, Plot 3: Chemical species concentrations (O2, H2S)
    H2S_scale, O2_scale = calculate_scaling_factors(raw['H2S'], raw['O2'])
    axes[0, 2].scatter(raw['H2S'] * H2S_scale, raw['depth'], label=f'H₂S (μM){format_scale_label(H2S_scale)}', color='none', edgecolors='green', s=25, marker='X')
    axes[0, 2].scatter(raw['O2'] * O2_scale, raw['depth'], label=f'O₂ (μM){format_scale_label(O2_scale)}', color='none', edgecolors='lightseagreen', s=25, marker='P')
    axes[0, 2].invert_yaxis()
    axes[0, 2].set_xlim(left=0)
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

    for i in range(4):
        if i < len(lake.organisms):
            for key in F[lake.organisms[i]]:
                axes[1, i].plot(F[lake.organisms[i]][key][0], data['depth'], label=f'F_{key}', linewidth=1, color=F[lake.organisms[i]][key][1])
            axes[1, i].invert_yaxis()
            axes[1, i].set_xlim(-0.03, 1.03)
            axes[1, i].set_ylim(lake.max_graphing_depth, 0)
            axes[1, i].set_xlabel('Forcing Factor', fontsize=10)
            axes[1, i].tick_params(axis='both', labelsize=8)
            axes[1, i].set_title(f'{lake.organisms[i].upper()} Forcing Factors', fontsize=13, fontweight='bold')
            axes[1, i].legend(loc='lower right', fontsize=6)
            axes[1, i].grid(True, alpha=0.3)
        else:
            axes[1, i].axis('off')
    
    fig.suptitle(f'Predicted Primary Production in Lake {lake.name.capitalize()}', fontsize=15, fontweight='bold')
    plt.tight_layout()

    return plt


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate lake growth plots')
    parser.add_argument('lake', type=str, help='Lake name (matano or cadagno)')
    parser.add_argument('--save', action='store_true', help='Save plot to file instead of displaying')
    args = parser.parse_args()
    
    # Validate lake name
    if args.lake not in LAKES:
        print(f"Error: Lake '{args.lake}' not found. Available lakes: {', '.join(LAKES.keys())}")
        return
    
    lake = LAKES[args.lake]
    
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

    # Dictionaries containing forcing factors and growth profiles with depth for each organism
    F = calculate_forcing_factors(lake, data)
    prod = calculate_growth_profiles(lake, data, R, rxn)
    
    plot = generate_plots(lake, data, raw, prod, F)

    filename = get_incremented_filename(f'{lake.name}_phototroph_growth', '.png', directory='matplotlib')

    if args.save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as '{filename}'\n")
    else:
        plt.show()

if __name__ == "__main__":
    main()