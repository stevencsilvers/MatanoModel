# Import NutMEG
import NutMEG as nm

# Forcing functions
from forcing_functions import (
    phi_opnnf,
    Monod,
    inhibition,
    light_opnf,
    phi_gsb,
    Haldane,
)


def get_opnnf(R, rxn, Pm, I, k_l_opnnf, alpha, NO3, NH4, P, H2S, R_no3, R_a, k_p_opnnf, a_inh, H2S_inh, mmr, mgr, num=1e6, name='Non-Nitrogen-Fixing Oxygenic Phototroph'):
    """
    Create a NutMEG.horde object representing non-nitrogen-fixing oxygenic phototrophs.

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
        Irradiance (µmol photons m⁻² s⁻¹)
    k_l_opnnf : float
        Monod half-saturation constant of light uptake for a non-nitrogen-fixing oxygenic phototroph (µmol photons m⁻² s⁻¹)
    alpha : float
        Platt fitting parameter which defines the slope of the P vs I curve
        at I=0.
    NO3 : float
        Concentration of NO3 (µM)
    NH4 : float
        Concentration of NH4 (µM)
    P : float
        Concentration of P (µM)
    R_no3 : float
        Monod half-saturation constant of NO3 uptake (µM)
    R_a : float
        Monod half-saturation constant of NH4 uptake (µM)
    k_p_opnnf : float
        Monod half-saturation constant of P uptake for a non-nitrogen-fixing oxygenic phototroph (µM)
    mmr : float
        Maximum metabolic rate (aka zeroth order rate constant k_max) (mol of reaction / s)
    mgr : float
        Maximum growth rate (aka mu_max) (/s)
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
            'CHNOPS_forcing_parameters': {
                'Phi': (phi_opnnf, ['NO3', 'NH4', 'P', 'R_no3', 'R_a', 'k_p_opnnf']),
                # 'Platt': (Platt_tanh, ['alpha', 'Pmax', 'I']),
                'Light': (Monod, ['I', 'k_l_opnnf']),
                'Sulfur': (inhibition, ['H2S', 'a_inh', 'H2S_inh'])
            },
            'CHNOPS_F_attrs': {
                'Pmax': Pm,
                'I': I, 'k_l_opnnf': k_l_opnnf, 'alpha': alpha,
                'NO3': NO3, 'NH4': NH4, 'P': P, 'H2S': H2S,
                'R_no3': R_no3, 'R_a': R_a,
                'k_p_opnnf': k_p_opnnf,
                'a_inh': a_inh, 'H2S_inh': H2S_inh
            }
        }
    )

    return _phototroph


def get_opnf(R, rxn, Pm, I, I_opt, P, O2, k_p_opnf, a_inh, O2_inh, mmr, mgr, num=1e6, name='Nitrogen-Fixing Oxygenic Phototroph'):
    """
    Create a NutMEG.horde object representing nitrogen-fixing oxygenic phototrophs.

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
        Irradiance (µmol photons m⁻² s⁻¹)
    I_opt : float
        Optimal irradiance for nitrogen-fixing oxygenic phototrophs (µmol photons m⁻² s⁻¹)
    P : float
        Concentration of P (µM)
    O2 : float
        Concentration of O2 (µM)
    k_p_opnf : float
        Monod half-saturation constant of P uptake for a nitrogen-fixing oxygenic phototroph (µM)
    a_inh : float
        Inhibition constant
    O2_inh : float
        Inhibition constant for O2
    mmr : float
        Maximum metabolic rate (aka zeroth order rate constant k_max) (mol of reaction / s)
    mgr : float
        Maximum growth rate (aka mu_max) (/s)
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
            'CHNOPS_forcing_parameters': {
                'Phi': (Monod, ['P', 'k_p_opnf']),
                'Light': (light_opnf, ['I', 'I_opt']),
                'Oxygen': (inhibition, ['O2', 'a_inh', 'O2_inh'])
            },
            'CHNOPS_F_attrs': {
                'Pmax': Pm,
                'I': I, 'I_opt': I_opt,
                'P': P, 'O2': O2,
                'k_p_opnf': k_p_opnf,
                'a_inh': a_inh, 'O2_inh': O2_inh
            }
        }
    )

    return _phototroph


def get_gsb(R, rxn, Pm, I, k_l_gsb, NO3, NH4, P, O2, H2S, R_no3, R_a, k_p_gsb, k_h2s_gsb, a_inh, O2_inh, mmr, mgr, num=1e6, name='Green Sulfur Bacterium'):
    """
    Create a NutMEG.horde object representing green sulfur bacteria.

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
        Irradiance (µmol photons m⁻² s⁻¹)
    k_l_gsb : float
        Monod half-saturation constant of light uptake for green sulfur bacteria (µmol photons m⁻² s⁻¹)
    NO3 : float
        Concentration of NO3 (µM)
    NH4 : float
        Concentration of NH4 (µM)
    P : float
        Concentration of P (µM)
    O2 : float
        Concentration of O2 (µM)
    H2S : float
        Concentration of H2S (µM)
    R_no3 : float
        Monod half-saturation constant of NO3 uptake (µM)
    R_a : float
        Monod half-saturation constant of NH4 uptake (µM)
    k_p_gsb : float
        Monod half-saturation constant of P uptake for green sulfur bacteria (µM)
    k_h2s_gsb : float
        Monod half-saturation constant of H2S uptake for green sulfur bacteria (µM)
    a_inh : float
        Inhibition constant
    O2_inh : float
        Inhibition constant for O2
    mmr : float
        Maximum metabolic rate (aka zeroth order rate constant k_max) (mol of reaction / s)
    mgr : float
        Maximum growth rate (aka mu_max) (/s)
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
            'CHNOPS_forcing_parameters': {
                'Phi': (phi_gsb, ['NO3', 'NH4', 'P', 'H2S', 'R_no3', 'R_a', 'k_p_gsb', 'k_h2s_gsb']),
                'Light': (Monod, ['I', 'k_l_gsb']),
                'Oxygen': (inhibition, ['O2', 'a_inh', 'O2_inh'])
            }, 
            'CHNOPS_F_attrs': {
                'Pmax': Pm,
                'I': I, 'k_l_gsb': k_l_gsb,
                'NO3': NO3, 'NH4': NH4, 'P': P, 'O2': O2, 'H2S': H2S,
                'R_no3': R_no3, 'R_a': R_a,
                'k_p_gsb': k_p_gsb, 'k_h2s_gsb': k_h2s_gsb,
                'a_inh': a_inh, 'O2_inh': O2_inh
            }
        }
    )

    return _phototroph


def get_psb(R, rxn, Pm, I, k_l_psb, NO3, NH4, P, H2S, R_no3, R_a, k_p_psb, K_s, K_i, mmr, mgr, num=1e6, name='Purple Sulfur Bacterium'):
    """
    Create a NutMEG.horde object representing purple sulfur bacteria.

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
        Irradiance (µmol photons m⁻² s⁻¹)
    k_l_psb : float
        Monod half-saturation constant of light uptake for purple sulfur bacteria (µmol photons m⁻² s⁻¹)
    NO3 : float
        Concentration of NO3 (µM)
    NH4 : float
        Concentration of NH4 (µM)
    P : float
        Concentration of P (µM)
    H2S : float
        Concentration of H2S (µM)
    R_no3 : float
        Monod half-saturation constant of NO3 uptake (µM)
    R_a : float
        Monod half-saturation constant of NH4 uptake (µM)
    k_p_psb : float
        Monod half-saturation constant of P uptake for purple sulfur bacteria (µM)

    K_s : float
        Half-saturation constant for H2S (µM)
    K_i : float
        Inhibition constant for H2S (µM)
    mmr : float
        Maximum metabolic rate (aka zeroth order rate constant k_max) (mol of reaction / s)
    mgr : float
        Maximum growth rate (aka mu_max) (/s)
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
            'CHNOPS_forcing_parameters': {
                'Phi': (phi_opnnf, ['NO3', 'NH4', 'P', 'R_no3', 'R_a', 'k_p_psb']),
                'Light': (Monod, ['I', 'k_l_psb']),
                'Sulfide': (Haldane, ['H2S', 'K_s', 'K_i'])
            }, 
            'CHNOPS_F_attrs': {
                'Pmax': Pm,
                'I': I, 'k_l_psb': k_l_psb,
                'NO3': NO3, 'NH4': NH4, 'P': P, 'H2S': H2S,
                'R_no3': R_no3, 'R_a': R_a, 'k_p_psb': k_p_psb,
                'K_s': K_s, 'K_i': K_i
            }
        }
    )

    return _phototroph