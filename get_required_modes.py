import numpy as np
import scipy.special as sp

def get_required_modes(radius, freq_hz, cutoff_factor=4.0, m=1):
    """
    Sélectionne automatiquement les modes TE_mn et TM_mn requis pour un rayon donné.
    
    Parameters:
        radius        : float - Rayon de la section du guide d'onde (m)
        freq_hz       : float - Fréquence de travail f0 (Hz)
        cutoff_factor : float - Facteur multiplicatif pour fc (ex. 3.0 à 5.0)
        m             : int   - Indice azimutal (m=1 par défaut pour TE11)
        
    Returns:
        modes : list of dict - Liste des modes retenus avec leurs caractéristiques
    """
    c = 299792458.0
    f_cutoff_max = cutoff_factor * freq_hz
    
    # Valeur maximale de chi pour respecter fc <= cutoff_factor * f0
    chi_max = (2 * np.pi * radius * f_cutoff_max) / c
    
    # Estimation dynamique du nombre de racines à tester
    n_search = max(10, int(chi_max + 10))
    
    # Calcul des racines : J'_m(x) = 0 (TE) et J_m(x) = 0 (TM)
    chi_te_all = sp.jnp_zeros(m, n_search)
    chi_tm_all = sp.jn_zeros(m, n_search)
    
    modes = []

    # 1. Extraction des modes TE
    for n, chi in enumerate(chi_te_all, start=1):
        fc = (chi * c) / (2 * np.pi * radius)
        if fc <= f_cutoff_max:
            modes.append({
                'type': 'TE', 'm': m, 'n': n,
                'chi': chi, 'fc_ghz': fc / 1e9,
                'propagative': fc <= freq_hz
            })
        else:
            break

    # 2. Extraction des modes TM
    for n, chi in enumerate(chi_tm_all, start=1):
        fc = (chi * c) / (2 * np.pi * radius)
        if fc <= f_cutoff_max:
            modes.append({
                'type': 'TM', 'm': m, 'n': n,
                'chi': chi, 'fc_ghz': fc / 1e9,
                'propagative': fc <= freq_hz
            })
        else:
            break

    return modes


def analyze_horn_profile_modes(r_profile, freq_hz, cutoff_factor=4.0):
    """
    Analyse le profil r(z) et affiche le nombre de modes au col et à l'ouverture.
    """
    r_min = np.min(r_profile)
    r_max = np.max(r_profile)
    
    modes_col = get_required_modes(r_min, freq_hz, cutoff_factor)
    modes_ouverture = get_required_modes(r_max, freq_hz, cutoff_factor)
    
    print(f"=== ANALYSE MODALE (f0 = {freq_hz/1e9:.1f} GHz, Facteur coupure = {cutoff_factor}) ===")
    print(f"1. COL (r = {r_min*1000:.2f} mm) :")
    print(f"   -> {len(modes_col)} modes au total ({sum(1 for m in modes_col if m['type']=='TE')} TE + {sum(1 for m in modes_col if m['type']=='TM')} TM)")
    print(f"   -> Propagatifs : {sum(1 for m in modes_col if m['propagative'])}")
    
    print(f"\n2. OUVERTURE (r = {r_max*1000:.2f} mm) :")
    print(f"   -> {len(modes_ouverture)} modes au total ({sum(1 for m in modes_ouverture if m['type']=='TE')} TE + {sum(1 for m in modes_ouverture if m['type']=='TM')} TM)")
    print(f"   -> Propagatifs : {sum(1 for m in modes_ouverture if m['propagative'])}")
    
    return modes_col, modes_ouverture


# ==============================================================================
# EXEMPLE DE TEST À 30 GHz
# ==============================================================================
if __name__ == "__main__":
    freq = 30e9  # 30 GHz
    
    # Profil d'un cornet (col: 4.2 mm, ouverture: 15 mm)
    z_pts = np.linspace(0, 0.05, 100)
    r_pts = 0.0042 + (0.015 - 0.0042) * (z_pts / 0.05)**1.5

    # Analyse modale au col et à l'ouverture avec un facteur 4.0
    modes_col, modes_out = analyze_horn_profile_modes(r_pts, freq_hz=freq, cutoff_factor=4.0)

    # Détail des modes retenus à l'ouverture
    print("\n--- DÉTAIL DES MODES À L'OUVERTURE ---")
    for m in modes_out:
        status = "Propagatif" if m['propagative'] else "Évanescent"
        print(f"Mode {m['type']}1,{m['n']} | fc = {m['fc_ghz']:.2f} GHz | {status}")