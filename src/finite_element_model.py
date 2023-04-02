import numpy as np


def lateral_pile_analysis(pile, soil, load):

    pile = calculate_pile_parameters(pile)
    K_p = pile_stiffness_matrix(pile)

    u = np.zeros(len(K_p))
    f_int = np.zeros(len(K_p))
    f_ext = external_force_vector(load, pile)

    res = 1
    iter = 0

    while res > 1e-3:
        K_s = soil_stiffness_matrix(u, pile, soil)
        K = K_p + K_s

        d_u = np.linalg.solve(K, (f_ext - f_int))
        u = u + d_u
        f_int = internal_force_vector(u, pile, soil)

        res = np.max(np.abs(f_ext - f_int))
        iter += 1

        if iter == 20:
            print('maximum number of iterations reached')
            print(res)
            break

    results_interpolated = interpolate_results(u, pile, soil)

    return results_interpolated


def interpolate_results(u, pile, soil):

    z_nodes = pile['z_nodes']
    n_elements = pile['n_elements']
    n_interp = 5
    z_e_interp = np.linspace(0, 1, n_interp)

    z_interp =  np.zeros(n_elements * n_interp)
    v_interp = np.zeros(n_elements * n_interp)
    phi_interp = np.zeros(n_elements * n_interp)
    M_interp = np.zeros(n_elements * n_interp)
    V_interp = np.zeros(n_elements * n_interp)
    p_interp = np.zeros(n_elements * n_interp)
    m_interp = np.zeros(n_elements * n_interp)

    for i in range(n_elements):
        l_e = z_nodes[i + 1] - z_nodes[i]
        phi = 12 * pile['EI'] / (pile['GAk'] * l_e ** 2)
        u_e = u[2*i:2*i+4]

        v_e_interp = [np.matmul(shape_functions(xi, l_e, phi), u_e) for xi in z_e_interp]
        phi_e_interp = [np.matmul(cross_section_rotation_functions(xi, l_e, phi), u_e) for xi in z_e_interp]
        M_e_interp = [pile['EI'] * np.matmul(bending_moment_functions(xi, l_e, phi), u_e) for xi in z_e_interp]
        V_e_interp = [pile['GAk'] * np.matmul(shear_force_functions(xi, l_e, phi), u_e) for xi in z_e_interp]
        p_e_interp = [lateral_soil_reaction(np.matmul(shape_functions(xi, l_e, phi), u_e), z_nodes[i] + xi * l_e, soil, pile) for xi in z_e_interp]
        m_e_interp = [distributed_moment_soil_reaction(np.matmul(cross_section_rotation_functions(xi, l_e, phi), u_e), pi, z_nodes[i] + xi * l_e, soil, pile) for xi, pi in zip(z_e_interp, p_e_interp)]

        z_interp[n_interp*i:n_interp*(i+1)] = z_nodes[i] + l_e * z_e_interp
        v_interp[n_interp * i:n_interp * (i + 1)] = v_e_interp
        phi_interp[n_interp * i:n_interp * (i + 1)] = phi_e_interp
        M_interp[n_interp * i:n_interp * (i + 1)] = M_e_interp
        V_interp[n_interp * i:n_interp * (i + 1)] = V_e_interp
        p_interp[n_interp * i:n_interp * (i + 1)] = p_e_interp
        m_interp[n_interp * i:n_interp * (i + 1)] = m_e_interp

    results_interpolated = {
        'z_interp': z_interp,
        'v_interp': v_interp,
        'phi_interp': phi_interp,
        'M_interp': M_interp,
        'V_interp': V_interp,
        'p_interp': p_interp,
        'm_interp': m_interp,
    }

    return results_interpolated


def external_force_vector(load, pile):

    n_dof = 2 * (pile['n_elements'] + 1)
    f = np.zeros(n_dof)

    f[0] = load['H_max']
    f[1] = load['h'] * load['H_max']

    return f


def calculate_pile_parameters(pile):

    pile['E'] = 210e9
    pile['G'] = 80e9
    pile['k'] = 0.5
    pile['A'] = np.pi / 4 * (pile['D']**2 - (pile['D'] - 2 * pile['t'])**2)
    pile['I'] = np.pi / 64 * (pile['D']**4 - (pile['D'] - 2 * pile['t'])**4)
    pile['EI'] = pile['E'] * pile['I']
    pile['GAk'] = pile['G'] * pile['A'] * pile['k']
    pile['z_nodes'] = np. linspace(0, pile['L'], pile['n_elements'] + 1)

    return pile


def internal_force_vector(u, pile, soil):

    n_elements = pile['n_elements']
    z_nodes = pile['z_nodes']
    L = pile['L']

    xi, wi = gauss_points()

    K_pile = pile_stiffness_matrix(pile)

    f_int_pile = np.matmul(K_pile, u)
    f_int_soil = np.zeros(len(u))

    for i in range(n_elements):
        f_e_soil = np.zeros(4)
        l_e = z_nodes[i + 1] - z_nodes[i]
        u_e = u[2*i:2*i+4]
        phi = 12 * pile['EI'] / (pile['GAk'] * l_e ** 2)

        for x, w in zip(xi, wi):
            z = z_nodes[i] + x * l_e

            N = np.array(shape_functions(x, l_e, phi))
            v = np.matmul(N, u_e)
            p = lateral_soil_reaction(v, z, soil, pile)

            dN = np.array(cross_section_rotation_functions(x, l_e, phi))
            psi = np.matmul(dN, u_e)
            m = distributed_moment_soil_reaction(psi, p, z, soil, pile)

            f_e_soil = f_e_soil + w * l_e * (p * N + m * dN)

        f_int_soil[2*i:2*i+4] = f_int_soil[2*i:2*i+4] + f_e_soil

    f_int_soil[-2] = f_int_soil[-2] + base_shear_soil_reaction(u[-2], L, soil, pile)
    f_int_soil[-1] = f_int_soil[-1] + base_moment_soil_reaction(u[-1], L, soil, pile)

    f_int = f_int_pile + f_int_soil

    return f_int


def pile_stiffness_matrix(pile):

    n_elements = pile['n_elements']
    z_nodes = pile['z_nodes']

    n_dof = 2 * (n_elements + 1)
    K = np.zeros((n_dof, n_dof))

    for i in range(n_elements):
        l_e = z_nodes[i+1] - z_nodes[i]
        K_e = beam_element_stiffness_matrix(l_e, pile['EI'], pile['GAk'])

        K[2*i:2*i+4, 2*i:2*i+4] = K[2*i:2*i+4, 2*i:2*i+4] + K_e

    return K


def soil_stiffness_matrix(u, pile, soil):

    n_elements = pile['n_elements']
    z_nodes = pile['z_nodes']
    EI = pile['EI']
    GAk = pile['GAk']
    L = pile['L']

    n_dof = 2 * (n_elements + 1)
    K = np.zeros((n_dof, n_dof))
    xi, wi = gauss_points()

    for i in range(n_elements):
        l_e = z_nodes[i + 1] - z_nodes[i]
        phi = 12 * EI / (GAk * l_e ** 2)
        u_e = u[2 * i:2 * i + 4]

        K_e = np.zeros((4, 4))
        for x, w in zip(xi, wi):

            v = np.matmul(shape_functions(x, l_e, phi), u_e)
            psi = np.matmul(cross_section_rotation_functions(x, l_e, phi), u_e)

            K_ls = local_soil_stiffness_matrix(v, psi, z_nodes[i] + x * l_e, soil, pile)

            N = shape_functions(x, l_e, phi)
            dN_b = cross_section_rotation_functions(x, l_e, phi)

            K_e = K_e + w * l_e * np.matmul(np.transpose([N, dN_b]), np.matmul(K_ls, [N, dN_b]))

        K[2 * i:2 * i + 4, 2 * i:2 * i + 4] = K[2 * i:2 * i + 4, 2 * i:2 * i + 4] + K_e

    d_v = 1e-6
    d_psi = 1e-7

    v_base = u[-2]
    psi_base = u[-1]

    base_shear_stiffness = (base_shear_soil_reaction(v_base + d_v / 2, L, soil, pile) - base_shear_soil_reaction(v_base - d_v / 2, L, soil, pile)) / d_v
    base_moment_stiffness = (base_moment_soil_reaction(psi_base + d_psi / 2, L, soil, pile) - base_moment_soil_reaction(psi_base - d_psi / 2, L, soil, pile)) / d_psi

    K[n_dof - 2, n_dof - 2] = K[n_dof - 2, n_dof - 2] + base_shear_stiffness
    K[n_dof - 1, n_dof - 1] = K[n_dof - 1, n_dof - 1] + base_moment_stiffness

    return K


def local_soil_stiffness_matrix(v, psi, z, soil, pile):

    d_v = 1e-6
    k_vv = (lateral_soil_reaction(v + d_v / 2, z, soil, pile) - lateral_soil_reaction(v - d_v / 2, z, soil, pile)) / d_v

    d_psi = 1e-7
    p = lateral_soil_reaction(v, z, soil, pile)
    k_pp = (distributed_moment_soil_reaction(psi + d_psi / 2, p, z, soil, pile) - distributed_moment_soil_reaction(psi - d_psi / 2, p, z, soil, pile)) / d_psi

    K = [[k_vv, 0], [0, k_pp]]

    return K


def gauss_points():

    # Gauss-Legendre quadrature with 4 points (suitable for Timoshenko beam theory)
    xi = (np.array([-.861136, -.339981, .339981, .861136]) + 1)/2
    wi = np.array([.347855, .652145, .652145, .347855]) / 2

    return xi, wi


def beam_element_stiffness_matrix(l, EI, GAk):

    phi = 12 * EI / (GAk * l**2)

    K = EI / (l**3 * (1 + phi)) * np.array([
        [12, 6 * l, -12, 6 * l],
        [6 * l, (4 + phi) * l**2, -6 * l, (2-phi) * l**2],  # Check this last term
        [-12, -6 * l, 12, -6 * l],
        [6 * l, (2 - phi) * l**2, -6 * l, (4 + phi) * l**2]  # Check the second term
    ])

    return K

def shape_functions(xi, l, phi):
    # https://www.slideshare.net/GhassanAlhamdany/timoshenko-beamelement

    N_1b = 1 / (1 + phi) * (1 - 3 * xi**2 + 2 * xi**3)
    N_1s = phi / (1 + phi) * (1 - xi)
    N_2b = l / (1 + phi) * (xi - 2 * xi**2 + xi**3 + 1 / 2 * (2 * xi - xi**2) * phi)
    N_2s = - l * phi / (1 + phi) * (1 / 2 * xi)
    N_3b = 1 / (1 + phi) * (3 * xi**2 - 2 * xi**3)
    N_3s = phi / (1 + phi) * xi
    N_4b = l / (1 + phi) * (-xi**2 + xi**3 + 1 / 2 * (xi**2) * phi)
    N_4s = - l * phi / (1 + phi) * (1 / 2 * xi)

    N_1 = N_1b + N_1s
    N_2 = N_2b + N_2s
    N_3 = N_3b + N_3s
    N_4 = N_4b + N_4s

    N = [N_1, N_2, N_3, N_4]

    return N


def cross_section_rotation_functions(xi, l, phi):

    dN_1b = 1 / (1 + phi) * (- 6 * xi + 6 * xi**2)
    dN_2b = l / (1 + phi) * (1 - 4 * xi + 3 * xi**2 + (1 - xi) * phi)
    dN_3b = 1 / (1 + phi) * (6 * xi - 6 * xi**2)
    dN_4b = l / (1 + phi) * (-2 * xi + 3 * xi**2 + xi * phi)

    dN_b = [dN_1b, dN_2b, dN_3b, dN_4b] / l

    return dN_b


def bending_moment_functions(xi, l, phi):

    d2N_1b = 1 / (1 + phi) * (- 6 + 12 * xi)
    d2N_2b = l / (1 + phi) * (- 4 + 6 * xi - phi)
    d2N_3b = 1 / (1 + phi) * (6 - 12 * xi)
    d2N_4b = l / (1 + phi) * (-2 + 6 * xi + phi)

    d2N_b = [d2N_1b, d2N_2b, d2N_3b, d2N_4b] / l**2

    return d2N_b


def shear_force_functions(xi, l, phi):

    dN_1s = -phi / (1 + phi)
    dN_2s = -phi * l / (1 + phi) / 2
    dN_3s = phi / (1 + phi)
    dN_4s = -phi * l / (1 + phi) / 2

    dN_s = [dN_1s, dN_2s, dN_3s, dN_4s] / l

    return dN_s


def lateral_soil_reaction(v, z, soil, pile):

    if soil['type'] == 'linear':

        p = soil['k_lateral'] * v

    elif soil['type'] == 'clay':

        D = pile['D']
        G = np.interp(z, soil['z'], soil['G'])
        su = np.interp(z, soil['z'], soil['su'])

        v_norm = v / D * G / su

        vu = 241.4
        k = 10.6 - 1.65 * z / D
        n = 0.939 - 0.03345 * z / D
        pu = 10.7 - 7.101 * np.exp(-0.3085 * z / D)

        p_norm = conic_function(v_norm, k, vu, pu, n)

        p = p_norm * su * D

    elif soil['type'] == 'sand':

        D = pile['D']
        L = pile['L']
        G = np.interp(z, soil['z'], soil['G'])
        sigma_v = np.interp(z, soil['z'], soil['sigma_v'])
        Dr = soil['Dr']

        if sigma_v == 0:
            return 0

        v_norm = v / D * G / sigma_v

        vu = 146.1 - 92.11 * Dr
        k = 8.731 - 0.6892 * Dr - 0.9178 * z / D
        n = 0.917 + 0.06193 * Dr
        pu = 0.3667 + 25.89 * Dr + (0.3375 - 8.9 * Dr) * z / L

        p_norm = conic_function(v_norm, k, vu, pu, n)

        p = p_norm * sigma_v * D

    return p


def distributed_moment_soil_reaction(psi, p, z, soil, pile):

    if soil['type'] == 'linear':

        m = soil['k_distributed_moment'] * psi

    elif soil['type'] == 'clay':

        D = pile['D']
        G = np.interp(z, soil['z'], soil['G'])
        su = np.interp(z, soil['z'], soil['su'])

        psi_norm = psi * G / su

        psiu = np.nan
        k = 1.420 - 0.09643 * z / D
        n = 0
        mu = 0.2899 - 0.04775 * z / D

        m_norm = conic_function(psi_norm, k, psiu, mu, n)

        m = m_norm * su * D**2

    elif soil['type'] == 'sand':

        D = pile['D']
        L = pile['L']
        G = np.interp(z, soil['z'], soil['G'])
        sigma_v = np.interp(z, soil['z'], soil['sigma_v'])
        Dr = soil['Dr']

        if sigma_v == 0:
            return 0

        psi_norm = psi * G / sigma_v

        psiu = np.nan
        k = 17
        n = 0
        mu = 0.2605 + (-0.1989 + 0.2019 * Dr) * z / L

        m_norm = conic_function(psi_norm, k, psiu, mu, n)

        m = m_norm * np.abs(p) * D

    return m


def base_shear_soil_reaction(v, z, soil, pile):

    if soil['type'] == 'linear':

        H = soil['k_base_shear'] * v

    elif soil['type'] == 'clay':

        D = pile['D']
        G = np.interp(z, soil['z'], soil['G'])
        su = np.interp(z, soil['z'], soil['su'])

        v_norm = v / D * G / su

        vu = 235.7
        k = 2.717 - 0.3575 * z / D
        n = 0.8793 - 0.0315 * z / D
        Hu = 0.4038 + 0.04812 * z / D

        H_norm = conic_function(v_norm, k, vu, Hu, n)

        H = H_norm * su * D**2

    elif soil['type'] == 'sand':

        D = pile['D']
        G = np.interp(z, soil['z'], soil['G'])
        sigma_v = np.interp(z, soil['z'], soil['sigma_v'])
        Dr = soil['Dr']

        if sigma_v == 0:
            return 0

        v_norm = v / D * G / sigma_v

        vu = 0.5150 + 2.883 * Dr + (0.1695 - 0.7018 * Dr) * z / D
        k = 6.505 - 2.985 * Dr + (-0.007969 - 0.4299 * Dr) * z / D
        n = 0.09978 + 0.7974 * Dr + (0.004994 - 0.07005 * Dr) * z / D
        Hu = 0.09952 + 0.7996 * Dr + (0.03988 - 0.1606 * Dr) * z / D

        H_norm = conic_function(v_norm, k, vu, Hu, n)

        H = H_norm * sigma_v * D**2

    return H


def base_moment_soil_reaction(psi, z, soil, pile):

    if soil['type'] == 'linear':

        M = soil['k_base_moment'] * psi

    elif soil['type'] == 'clay':

        D = pile['D']
        G = np.interp(z, soil['z'], soil['G'])
        su = np.interp(z, soil['z'], soil['su'])

        psi_norm = psi * G / su

        psiu = 173.1
        k = 0.2164 - 0.002132 * z / D
        n = 1.079 - 0.1087 * z / D
        Mu = 0.8192 - 0.08588 * z / D

        M_norm = conic_function(psi_norm, k, psiu, Mu, n)

        M = M_norm * su * D**3

    elif soil['type'] == 'sand':

        D = pile['D']
        G = np.interp(z, soil['z'], soil['G'])
        sigma_v = np.interp(z, soil['z'], soil['sigma_v'])
        Dr = soil['Dr']

        if sigma_v == 0:
            return 0

        psi_norm = psi * G / sigma_v

        psiu = 44.89
        k = 0.3515
        n = 0.3 + 0.4986 * Dr
        Mu = (0.09981 + 0.371 * Dr) + (0.01998 - 0.09041 * Dr) * z / D

        M_norm = conic_function(psi_norm, k, psiu, Mu, n)

        M = M_norm * sigma_v * D**3

    return M


def conic_function(x, k, xu, yu, n):

    sign_x = np.sign(x)
    abs_x = np.abs(x)

    if n == 0:
        abs_y = np.minimum(k * abs_x, yu)
    elif abs_x >= xu:
        abs_y = yu
    else:
        a = 1 - 2 * n
        b = 2 * n * abs_x / xu - (1 - n) * (1 + abs_x * k / yu)
        c = abs_x * k / yu * (1 - n) - n * (abs_x / xu)**2

        abs_y = yu * 2 * c / (-b + np.sqrt(b**2 - 4 * a * c))

    y = sign_x * abs_y

    return y


if __name__ == '__main__':

    pile = {
        'D': 2,
        't': 0.04,
        'L': 10.5,
        'n_elements': 20,
    }

    soil = {
        'type': 'linear',
        'k_lateral': 1e7,
        'z': [0, 2, 4, 5, 8, 13, 30],
        'su': np.array([60, 160, 120, 105, 115, 130, 185]) * 1e3,
        'G': np.array([2, 35, 70,  85, 130, 160, 330]) * 1e6
    }

    load = {
        'H_max': 20e6,
        'h': 10,
        'n_steps': 10
    }

    results = lateral_pile_analysis(pile, soil, load)
