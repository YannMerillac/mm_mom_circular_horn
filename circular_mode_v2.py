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

    def _compute_norm(self):
        """Calcule la constante N telle que pi * int_0^R (|fr|^2 + |fphi|^2) r dr = 1."""
        def integrand(r):
            e_r, _ = self.field_r(r)
            return (e_r[0]**2 + e_r[1]**2) * r

        val, _ = integ.quad(integrand, 0, self.R)
        return 1.0 / np.sqrt(np.pi * val)

    def field_r(self, r):
        u = self.kc * r
        jv_m_u = sp.jv(self.m, u)
        jvp_m_u = sp.jvp(self.m, u)
        fr1 = self.kc * jvp_m_u
        if isinstance(r, np.ndarray):
            if self.m == 1:
                fr2 = 0.5 * self.kc * np.ones_like(u)
            else:
                fr2 = np.zeros_like(u)
            i_filter = np.where(r > 1e-12)[0]
            fr2[i_filter] = (self.m / r[i_filter]) * jv_m_u[i_filter]
        else:
            fr2 = (self.m / r) * jv_m_u if r > 1e-12 else (0.5 * self.kc if self.m == 1 else 0.0)
        if self.type == "TE":
            ez = np.zeros_like(u)
            field_e_r = np.array([fr2, fr1, ez]).T
            field_h_r = np.array([-fr1, fr2, jv_m_u]).T
        else:
            hz = np.zeros_like(u)
            field_e_r = np.array([-fr1, fr2, jv_m_u]).T
            field_h_r = np.array([-fr2, -fr1, hz]).T
        return field_e_r, field_h_r

    def field_theta(self, theta):
        cos_m_theta = np.cos(self.m * theta)
        sin_m_theta = np.sin(self.m * theta)
        array_zero = np.zeros_like(theta)
        if self.type == "TE":
            field_e_t = np.array([sin_m_theta, cos_m_theta, array_zero]).T
            field_h_t = np.array([cos_m_theta, sin_m_theta, cos_m_theta]).T
        else:
            field_e_t = np.array([cos_m_theta, sin_m_theta, cos_m_theta]).T
            field_h_t = np.array([sin_m_theta, cos_m_theta, array_zero]).T
        return field_e_t, field_h_t
    
    def field_z(self, z, dir_z):
        beta = self.beta * dir_z
        exp_beta_z = np.exp(-1j * beta * z)
        if self.type == "TE":
            coeff_e_t = 1j * mu0 * self.omega / self.kc **2
            coeff_h_t = 1j * beta / self.kc ** 2
        else:
            coeff_e_t = 1j * beta / self.kc ** 2
            coeff_h_t = 1j * eps0 * self.omega / self.kc ** 2
        e_z = np.array([coeff_e_t * exp_beta_z, coeff_e_t * exp_beta_z, exp_beta_z]).T
        h_z = np.array([coeff_h_t * exp_beta_z, coeff_h_t * exp_beta_z, exp_beta_z]).T
        return e_z, h_z


    def field_rtz(self, r, theta, z, e0=1., dir_z=1.):
        e_r, h_r = self.field_r(r)
        e_theta, h_theta = self.field_theta(theta)
        e_z, h_z = self.field_z(z, dir_z)
        return e0 * e_r* e_theta * e_z, e0 * h_r * h_theta * h_z
    

    def field_tangent(self, r, theta, z=0.0, dir_z=1.0):
        e_r, h_r = self.field_r(r)
        e_theta, h_theta = self.field_theta(theta)
        beta = self.beta * dir_z
        exp_beta_z = np.exp(1j * beta * z)
        e = e_r * e_theta * exp_beta_z
        h = dir_z * self.Y * h_r * h_theta * exp_beta_z
        return e[:2], h[:2]


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

            def integrand_tr(t):
                et1, ht1 = m1.field_theta(t)
                et2, ht2 = m2.field_theta(t)
                return et1[0] * et2[0]
            
            coeff_r, _ = integ.quad(integrand_tr, 0, 2*np.pi)

            def integrand_tt(t):
                et1, ht1 = m1.field_theta(t)
                et2, ht2 = m2.field_theta(t)
                return et1[1] * et2[1]
            
            coeff_t, _ = integ.quad(integrand_tt, 0, 2*np.pi)
            
            # Couplage nul entre modes TE et TM si orthogonalite parfaite,
            # mais l'integration numerique traite directement tous les cas.
            def integrand_r(r):
                er1, hr1 = m1.field_r(r)
                er2, hr2 = m2.field_r(r)
                return (coeff_t * er1[0]*er2[0] + coeff_t * er1[1] * er2[1]) * r
            
            val, _ = integ.quad(integrand_r, 0, a)
            M[i, j] = val * m2.norm ** 2
            
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
    A = M @ Y2 @ M_T + Y1
    A_inv = np.linalg.inv(A)
    
    S11 = A_inv @ (Y1 - M @ Y2 @ M_T)
    S12 = 2.0 * A_inv @ (M @ Y2)
    S21 = M_T @ (S11 + np.eye(N1))
    S22 = M_T @ S12 - np.eye(N2)
    #S21 = 2.0 * M_T @ (A_inv @ Y1)
    #S22 = 2.0 * M_T @ (A_inv @ (M @ Y2)) - np.eye(N2)

    
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
