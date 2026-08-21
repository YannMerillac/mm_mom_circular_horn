import numpy as np
import scipy.special as sp
import scipy.integrate as integ
import vtk
from vtk.util.numpy_support import numpy_to_vtk

c = 299792458.0
mu0 = 4 * np.pi * 1e-7
eps0 = 1.0 / (c**2 * mu0)

class CircularMode:
    """Représente un mode TE ou TM dans un guide d'onde circulaire."""
    def __init__(self, mode_type, m, n, radius, freq_hz):
        self.type = mode_type  # 'TE' ou 'TM'
        self.m = m
        self.n = n
        self.R = radius
        
        self.omega = 2 * np.pi * freq_hz
        self.k0 = self.omega / c
        
        # Racine du mode (J'_m pour TE, J_m pour TM)
        if mode_type == 'TE':
            self.chi = sp.jnp_zeros(m, n)[n - 1]
        else:
            self.chi = sp.jn_zeros(m, n)[n - 1]
            
        self.kc = self.chi / self.R
        
        # Constante de propagation beta (complexe pour modes evanescent)
        if self.k0 >= self.kc:
            self.beta = np.sqrt(self.k0**2 - self.kc**2)
        else:
            self.beta = -1j * np.sqrt(self.kc**2 - self.k0**2)
            
        # Admittance modale Y
        if mode_type == 'TE':
            self.Y = self.beta / (self.omega * mu0)
        else:
            self.Y = (self.omega * eps0) / self.beta
            
        # Normalisation orthonormale du champ transverse
        self.norm = self._compute_norm()

    def field_components(self, r):
        """Composantes radiales et azimutales du champ electrique transverse."""
        u = self.kc * r
        m = self.m
        
        if self.type == 'TE':
            fr = (m / r) * sp.jv(m, u) if r > 1e-12 else (0.5 * self.kc if m == 1 else 0.0)
            fphi = -self.kc * sp.jvp(m, u)
        else:  # TM
            fr = -self.kc * sp.jvp(m, u)
            fphi = (m / r) * sp.jv(m, u) if r > 1e-12 else (0.5 * self.kc if m == 1 else 0.0)
            
        return self.norm * fr, self.norm * fphi

    def _compute_norm(self):
        """Calcule la constante N telle que pi * int_0^R (|fr|^2 + |fphi|^2) r dr = 1."""
        def integrand(r):
            fr, fphi = self.field_components_raw(r)
            return (fr**2 + fphi**2) * r

        val, _ = integ.quad(integrand, 0, self.R)
        return 1.0 / np.sqrt(np.pi * val)

    def field_components_raw(self, r):
        u = self.kc * r
        m = self.m
        if self.type == 'TE':
            fr = (m / r) * sp.jv(m, u) if r > 1e-12 else (0.5 * self.kc if m == 1 else 0.0)
            fphi = -self.kc * sp.jvp(m, u)
        else:
            fr = -self.kc * sp.jvp(m, u)
            fphi = (m / r) * sp.jv(m, u) if r > 1e-12 else (0.5 * self.kc if m == 1 else 0.0)
        return fr, fphi

    def field_rtz(self, r, theta, z, e0=1.):
        u = self.kc * r
        cos_n_theta_exp_beta_z = e0 * np.cos(self.m * theta) * np.exp(-1j * self.beta * z)
        sin_n_theta_exp_beta_z = e0 * np.sin(self.m * theta) * np.exp(-1j * self.beta * z)
        jv_m_u = sp.jv(self.m, u)
        jvp_m_u = sp.jvp(self.m, u)
        if self.type == 'TM':
            e_z = jv_m_u * cos_n_theta_exp_beta_z
            e_r = -1j * self.beta / self.kc * jvp_m_u * cos_n_theta_exp_beta_z
            e_theta = 1j * self.beta * self.m / (self.kc ** 2 * r) * jv_m_u * sin_n_theta_exp_beta_z
            h_r = -1j * self.omega * eps0 * self.m / (self.kc ** 2 * r) * jv_m_u * sin_n_theta_exp_beta_z
            h_theta = -1j * self.omega * eps0 / self.kc * jvp_m_u * cos_n_theta_exp_beta_z
            h_z = np.zeros_like(h_r)
        else:
            e_r = 1j * self.omega * mu0 * self.m / (self.kc ** 2 * r) * jv_m_u * sin_n_theta_exp_beta_z
            e_theta = 1j * self.omega * mu0 / self.kc * jvp_m_u * cos_n_theta_exp_beta_z
            e_z = np.zeros_like(e_r)
            h_z = jv_m_u * cos_n_theta_exp_beta_z
            h_r = -1j * self.beta / self.kc * jvp_m_u * cos_n_theta_exp_beta_z
            h_theta = 1j * self.beta * self.m / (self.kc ** 2 * r) * jv_m_u * sin_n_theta_exp_beta_z
        return np.array([e_r, e_theta, e_z]).T, np.array([h_r, h_theta, h_z]).T
    
    def field_xyz(self, r, theta, z, e0=1.):
        E_rtz, H_rtz = self.field_rtz(r, theta, z, e0)
        Er, Ephi = E_rtz[0], E_rtz[1]
        Hr, Hphi = H_rtz[0], H_rtz[1]
        Ex = Er * np.cos(theta) - Ephi * np.sin(theta)
        Ey = Er * np.sin(theta) + Ephi * np.cos(theta)

        Hx = Hr * np.cos(theta) - Hphi * np.sin(theta)
        Hy = Hr * np.sin(theta) + Hphi * np.cos(theta)
        E_rtz[0] = Ex
        E_rtz[1] = Ey
        H_rtz[0] = Hx
        H_rtz[1] = Hy
        return E_rtz, H_rtz


    def export_to_vtk(self, z_min=0., z_max=None, n_r=21, n_theta=51, n_z = 100, filename=None):
        if z_max is None:
            z_max = z_min + self.R * 10.
        # Maillage cylindrique (r commence à 1e-5 pour éviter la division par zéro à l'origine)
        r = np.linspace(1e-5 * self.R, self.R, n_r)
        phi = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        z = np.linspace(z_min, z_max, n_z)

        R, PHI, Z = np.meshgrid(r, phi, z, indexing='ij')
        R = R.ravel(order="F")
        PHI = PHI.ravel(order="F")
        Z = Z.ravel(order="F")

        E_rtz ,H_rtz = self.field_rtz(R, PHI, Z)
        
        # Coordonnées cartésiennes
        X = R * np.cos(PHI)
        Y = R * np.sin(PHI)

        Er = np.real(E_rtz[:, 0])
        Ephi = np.real(E_rtz[:, 1])
        Ez = np.real(E_rtz[:, 2])

        Hr = np.real(H_rtz[:, 0])
        Hphi = np.real(H_rtz[:, 1])
        Hz = np.real(H_rtz[:, 2])

        # Projection des composantes vectorielles en coordonnées cartésiennes
        Ex = Er * np.cos(PHI) - Ephi * np.sin(PHI)
        Ey = Er * np.sin(PHI) + Ephi * np.cos(PHI)

        Hx = Hr * np.cos(PHI) - Hphi * np.sin(PHI)
        Hy = Hr * np.sin(PHI) + Hphi * np.cos(PHI)

        # Formatage des données en listes 1D pour VTK
        coords = np.column_stack((X, Y, Z))
        E_vec = np.column_stack((Ex, Ey, Ez))
        H_vec = np.column_stack((Hx, Hy, Hz))

        E_mag = np.linalg.norm(E_vec, axis=1)
        H_mag = np.linalg.norm(H_vec, axis=1)

        # Création du vtkStructuredGrid
        sgrid = vtk.vtkStructuredGrid()
        sgrid.SetDimensions(n_r, n_theta, n_z)

        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(coords, deep=True))
        sgrid.SetPoints(vtk_points)

        # Ajout des tableaux de données aux points
        vtk_E = numpy_to_vtk(E_vec, deep=True)
        vtk_E.SetName("E_field")

        vtk_H = numpy_to_vtk(H_vec, deep=True)
        vtk_H.SetName("H_field")

        vtk_E_mag = numpy_to_vtk(E_mag, deep=True)
        vtk_E_mag.SetName("E_magnitude")

        vtk_H_mag = numpy_to_vtk(H_mag, deep=True)
        vtk_H_mag.SetName("H_magnitude")

        point_data = sgrid.GetPointData()
        point_data.AddArray(vtk_E)
        point_data.AddArray(vtk_H)
        point_data.AddArray(vtk_E_mag)
        point_data.AddArray(vtk_H_mag)
        point_data.SetActiveVectors("E_field")
        point_data.SetActiveScalars("E_magnitude")

        # Écriture du fichier .vts
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(sgrid)
        writer.Write()

        print(f"Export VTS réussi : '{filename}' (Grille 3D {n_r}x{n_theta}x{n_z}).")

            



def compute_coupling_matrix(modes1, modes2, a):
    """
    Calcule la matrice M_ij = <e1_i, e2_j> sur la surface du petit guide (rayon a).
    """
    N1, N2 = len(modes1), len(modes2)
    M = np.zeros((N1, N2))
    
    for i in range(N1):
        for j in range(N2):
            m1, m2 = modes1[i], modes2[j]
            
            # Couplage nul entre modes TE et TM si orthogonalite parfaite,
            # mais l'integration numerique traite directement tous les cas.
            def integrand(r):
                fr1, fphi1 = m1.field_components(r)
                fr2, fphi2 = m2.field_components(r)
                return (fr1 * fr2 + fphi1 * fphi2) * r
            
            val, _ = integ.quad(integrand, 0, a)
            M[i, j] = np.pi * val
            
    return M


def compute_step_gsm(modes1, modes2, M):
    """
    Calcule la matrice GSM globale S de la jonction a partir de M et des admittances Y.
    """
    N1, N2 = len(modes1), len(modes2)
    
    Y1 = np.diag([m.Y for m in modes1])
    Y2 = np.diag([m.Y for m in modes2])
    
    M_T = M.T
    
    # Resolution des equations de continuite aux limites
    A = M_T @ Y1 @ M + Y2
    A_inv = np.linalg.inv(A)
    
    S21 = 2.0 * A_inv @ (M_T @ Y1)
    S22 = A_inv @ (Y2 - M_T @ Y1 @ M)
    S11 = M @ S21 - np.eye(N1)
    S12 = M @ (S22 + np.eye(N2))
    
    # Matrice S globale (N1+N2 x N1+N2)
    S = np.block([[S11, S12],
                  [S21, S22]])
    
    return S, S11, S12, S21, S22


# ==============================================================================
# EXEMPLE : MARCHE DE RAYON 10 mm -> 15 mm A 20 GHz (Indice azimutal m=1)
# ==============================================================================
if __name__ == "__main__":
    freq = 20e9      # 20 GHz
    a = 0.010        # Petit guide : 10 mm
    b = 0.015        # Grand guide : 15 mm
    m_index = 1      # Famille azimutale m=1 (TE11, TM11, TE12, etc.)

    # Definition des modes de la base (ex. 3 TE + 3 TM dans chaque guide)
    modes_guide1 = [
        CircularMode('TE', m_index, 1, a, freq),
        CircularMode('TE', m_index, 2, a, freq),
        CircularMode('TM', m_index, 1, a, freq),
        CircularMode('TM', m_index, 2, a, freq),
    ]

    modes_guide2 = [
        CircularMode('TE', m_index, 1, b, freq),
        CircularMode('TE', m_index, 2, b, freq),
        CircularMode('TM', m_index, 1, b, freq),
        CircularMode('TM', m_index, 2, b, freq),
    ]

    # 1. Calcul de la matrice de couplage M
    M = compute_coupling_matrix(modes_guide1, modes_guide2, a)

    # 2. Calcul de la GSM
    S_full, S11, S12, S21, S22 = compute_step_gsm(modes_guide1, modes_guide2, M)

    # 3. Affichage du coefficient de réflexion du mode fondamental TE11
    S11_TE11_dB = 20 * np.log10(np.abs(S11[0, 0]))
    S21_TE11_dB = 20 * np.log10(np.abs(S21[0, 0]))

    print(f"--- MATRICE DE COUPLAGE M ({len(modes_guide1)}x{len(modes_guide2)}) ---")
    print(np.round(M, 4))
    print("\n--- RÉSULTATS GSM POUR LE MODE TE11 ---")
    print(f"S11 (TE11 -> TE11) : {S11_TE11_dB:.2f} dB  (Amplitude: {np.abs(S11[0,0]):.4f})")
    print(f"S21 (TE11 -> TE11) : {S21_TE11_dB:.2f} dB  (Amplitude: {np.abs(S21[0,0]):.4f})")
