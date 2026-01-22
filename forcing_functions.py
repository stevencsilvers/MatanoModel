import numpy as np


def phi_opnnf(resp, NO3, NH4, P, R_no3, R_a, k_p):
    """
    Forcing function for growth inhibition of non-nitrogen-fixing oxygenic phototrophs based on bioavailable nitrogen and phosphorus.
    Based on Matano paper page 40, equation 9.

    This follows the Matano model formulation:
        β_t = β_NO3 + β_a (NH4)
        Where β_NO3 and β_a are Monod terms for NO3 and NH4 respectively.
        NO2 not included as there is no data and there is a very low concentration of it in Lake Matano

    Parameters
    ----------
    resp : NoneType
        Placeholder for NutMEG forcing function interface
    NO3 : float
        Nitrate concentration (μM)
    NH4 : float
        Ammonium concentration (μM)
    P : float
        Phosphorus concentration (μM)
    R_no3: float
        Monod half-saturation constant for NO3 uptake (μM)
    R_a : float
        Monod half-saturation constant for NH4 uptake (μM)
    k_p : float
        Monod half-saturation constant for P uptake (μM)
    """
    return min(Monod_nitrogen(NO3, NH4, R_no3, R_a), Monod(None, P, k_p))


def phi_gsb(resp, NO3, NH4, P, H2S, R_no3, R_a, k_p, k_h2s):
    """
    Forcing function for growth inhibition of green sulfur bacteria based on bioavailable nitrogen, phosphorus, and hydrogen sulfide.
    Based on Matano paper page 41, equation 5.

    Parameters
    ----------
    resp : NoneType
        Placeholder for NutMEG forcing function interface
    NO3 : float
        Nitrate concentration (μM)
    NH4 : float
        Ammonium concentration (μM)
    P : float
        Phosphorus concentration (μM)
    H2S : float
        Hydrogen sulfide concentration (μM)
    R_no3: float
        Monod half-saturation constant for NO3 uptake (μM)
    R_a : float
        Monod half-saturation constant for NH4 uptake (μM)
    k_p : float
        Monod half-saturation constant for P uptake (μM)
    k_h2s : float
        Monod half-saturation constant for H2S uptake (μM)
    """
    return min(Monod_nitrogen(NO3, NH4, R_no3, R_a), Monod(None, P, k_p), Monod(None, H2S, k_h2s))


def Monod_nitrogen(NO3, NH4, R_no3, R_a):
    """
    Forcing function for nitrogen limitation based on concentrations of NO3 and NH4.
    From Matano Paper

    Parameters
    ----------
    NO3 : float
        Nitrate concentration (μM)
    NH4 : float
        Ammonium concentration (μM)
    R_no3: float
        Monod half-saturation constant for NO3 uptake (μM)
    R_a : float
        Monod half-saturation constant for NH4 uptake (μM)
    """
    beta_NO3 = NO3 / (R_no3 + NO3)
    beta_a = NH4 / (R_a + NH4)
    return beta_NO3 + beta_a


def Monod(resp, S, k_S):
    """
    Forcing function for Monod limitation based on chemical species (S) concentration.

    Parameters
    ----------
    resp : NoneType
        Placeholder for NutMEG forcing function interface
    S : float
        Concentration of the limiting species (µM)
    k_S : float
        Monod half-saturation constant for the species (µM)
    """
    return S / (k_S + S)


def temperature_modifier(t, k_t):
    return np.exp(k_t * (t - 20))


def light_opnf(resp, I, I_opt):
    """
    Forcing function for light limitation in nitrogen-fixing oxygenic phototrophs.
    Matano paper page 41, equation 3

    Parameters
    ----------
    resp : NoneType
        Placeholder for NutMEG forcing function interface
    I : float
        Irradiance (µmol photons m⁻² s⁻¹)
    I_opt : float
        Optimal irradiance (µmol photons m⁻² s⁻¹)
    """
    if I == 0:
        return 0.0
    return (I_opt / I) * np.exp(1 - (I_opt / I))


def inhibition(resp, S, a_inh, S_inh):
    """
    Forcing function for growth inhibition based on chemical species (S) concentration.
    From Matano Paper

    Parameters
    ----------
    S : float
        Concentration of the inhibiting species (µM)
    a_inh : float
        Inhibition constant
    S_inh : float
        Inhibition constant for the species
    """
    return 0.5 * (1 - np.tanh(a_inh * (S - S_inh)))


def Haldane(resp, S, K_s, K_i):
    """
    Forcing function for limitation + inhibition.
    From Gemerden 1974

    Parameters
    ----------
    resp : NoneType
        Placeholder for NutMEG forcing function interface
    S : float
        Concentration of species (µM)
    K_s : float
        Half-saturation constant for species uptake (µM)
    K_i : float
        Inhibition constant for species (µM)
    """
    return S / (((K_s + S)) * (1 + (S / K_i)))