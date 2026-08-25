import numpy as np

def explicit_Euler(y, h, F, dF=None):  # Euler Explícito y_n+1=y_n+h*f(y_n)
    m = y.size // 4
    p = y[0:2 * m]
    q = y[2 * m:4 * m]
    p_new = p + h * F(q)
    q_new = q + h * p
    return np.concatenate([p_new, q_new])


# Para métodos implícitos, necesitamos resolver sistemas de ecuaciones NO lineales por Newton-Raphson
def Newton(xn, G,JG, tol=1e-12, maxit=100):
    x = xn.copy()
    for k in range(maxit):
        Gx = G(x)
        if np.max(np.abs(Gx)) < tol:
            return x
        x = x - np.linalg.solve(JG(x), Gx)
    raise RuntimeError(f"Newton no convergió en {maxit} iteraciones (||G||={np.max(np.abs(Gx)):.2e})")


def implicit_Euler(y, h,F, dF):
    m = y.size // 4
    p, q = y[0:2 * m], y[2 * m:4 * m]
    Id = np.eye(2 * m)
    c = q + h * p
    G = lambda x: x - h ** 2 * F(x) - c
    JG = lambda x: Id - h ** 2 * dF(x)

    q_new = Newton(q, G, JG)
    p_new = p + h * F(q_new)
    return np.concatenate([p_new, q_new])


def implicit_midpoint(y, h,F,dF):
    m = y.size // 4
    p, q = y[0:2 * m], y[2 * m:4 * m]
    c = q + 0.5 * h * p
    Id = np.eye(2 * m)

    G = lambda x: x - h ** 2 / 4 * F(x) - c
    JG = lambda x: Id - h ** 2 / 4 * dF(x)

    q_m = Newton(q, G, JG)
    q_new = 2 * q_m - q
    p_new = p + h * F(q_m)

    return np.concatenate([p_new, q_new])


def symplectic_euler(y, h,F,dF=None):
    m = y.size // 4
    p, q = y[0:2 * m], y[2 * m:4 * m]
    p_new = p + h * F(q)
    q_new = q + h * p_new
    return np.concatenate([p_new, q_new])


def symplectic_euler_adjoint(y, h,F, dF=None):
    m = y.size // 4
    p, q = y[0:2 * m], y[2 * m:4 * m]
    q_new = q + h * p
    p_new = p + h * F(q_new)
    return np.concatenate([p_new, q_new])


# Implementación de Stormer-Verlet como composición de dos Euler Simplécticos

def stormer_verlet_composition(y, h, F,dF=None):
    # Euler Simpléctico
    y_half = symplectic_euler(y, h / 2.0, F, dF)
    # Euler Simpléctico Adjunto
    return symplectic_euler_adjoint(y_half, h / 2.0, F,dF)


# Implementación directa de Stormer-Verlet

def stormer_verlet(y, h, F, dF=None):
    m = y.size // 4
    p, q = y[0:2 * m], y[2 * m:4 * m]
    p_half = p + (h / 2.0) * F(q)
    q_new = q + h * p_half
    p_new = p_half + (h / 2.0) * F(q_new)

    return np.concatenate([p_new, q_new])

#METODOS DE PROYECCION

def proyecta(y_tilde, invariantes, grad_invariantes, valores0, tol=1e-12, maxit=10):
    # invariantes: función y -> vector g0(y) de valores (H, L, ...)
    # grad_invariantes: y -> matriz g'(y) de tamaño (m, 4)
    # valores0: (H0, L0, ...) fijados por la condición inicial
    lam = np.zeros(len(valores0))
    G = grad_invariantes(y_tilde)          # g' evaluado en y_tilde (fijo)
    GGT = G @ G.T                          # matriz m x m (aquí 2x2)
    for _ in range(maxit):
        y = y_tilde + G.T @ lam
        r = invariantes(y) - valores0      # residuo g(y)
        if np.max(np.abs(r)) < tol:
            return y
        lam = lam - np.linalg.solve(GGT, r)
    return y_tilde + G.T @ lam

def paso_proyectado(y, h, metodo_base, F, dF, invariantes, grad_inv, valores0):
    y_tilde = metodo_base(y, h, F, dF)
    return proyecta(y_tilde, invariantes, grad_inv, valores0)