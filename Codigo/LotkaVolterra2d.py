import numpy as np
import matplotlib.pyplot as plt



# --- 1. Definición del Integrador ---
def stormer_verlet_lv(y, h):
    p, q = y[0], y[1]
    p_half = p + h / 2 * (np.exp(q) - 2)
    q_new = q + h * (1 - np.exp(p_half))
    p_new = p_half + h / 2 * (np.exp(q_new) - 2)
    return np.array([p_new, q_new])

def symplectic_euler_lv(y, h, F=None, dF=None):
    p = y[0]
    q = y[1]
    p_new = p + h * (np.exp(q) - 2)
    q_new = q + h * (1 - np.exp(p_new))
    return np.array([p_new, q_new])

# --- 2. Funciones de Energía Exacta ---
def I_exacta(u, v):
    return np.log(u) - u + 2 * np.log(v) - v
def I_m(u,v,h):
    return (
                np.log(u)
                - u
                + 2 * np.log(v)
                - v
                - h / 2 * (1 - u) * (2 - v)
                - h ** 2 / 12 * (
                        u * (2 - v) ** 2
                        + v * (1 - u) ** 2
                )
        )

def H_exacta(p, q):
    return p - np.exp(p) + 2 * q - np.exp(q)
def H_m(p,q,h):
    return p-np.exp(p) + 2 * q - np.exp(q)-h/2*(1-np.exp(p))*(2-np.exp(q))-h**2/12*(np.exp(p)*(2-np.exp(q))**2+np.exp(q)*(1-np.exp(p))**2)

print("Iniciando simulación...")

# Parámetros de la simulación
CI = [[1.1, 2.1], [1.6, 2.6], [2.1, 3.1], [2.8, 3.8]]
h = 0.1
pasos = 400

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

v_grid = np.linspace(0.1, 6, 400)
u_grid = np.linspace(0.1, 4, 400)
V, U = np.meshgrid(v_grid, u_grid)
I_malla = I_exacta(U, V)
Im_malla = I_m(U, V,h)

q_grid = np.linspace(-2.5, 2.5, 400)
p_grid = np.linspace(-2.5, 2.0, 400)
Q, P = np.meshgrid(q_grid, p_grid)
H_malla = H_exacta(P, Q)
Hm_malla=H_m(P,Q,h)


#ax1.contour(V, U, I_malla, levels=15, colors='lightgray', alpha=0.4)
#ax2.contour(Q, P, H_malla, levels=15, colors='lightgray', alpha=0.4)
#ax2.contour(Q, P, Hm_malla, levels=15, colors='blue', alpha=0.4)

colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, x in enumerate(CI):
    I_0 = I_exacta(x[0], x[1])
    H_m_0 = H_m(np.log(x[0]), np.log(x[1]),h)
    ax1.contour(V, U, I_malla, levels=[I_0], colors='blue', linestyles='dashed', linewidths=2.5, alpha=0.5, zorder=3)
    ax1.contour(V, U, Im_malla, levels=[H_m_0], colors='black', linestyles='dashed', linewidths=2.5, alpha=0.5,zorder=3)

    ax2.contour(Q, P, H_malla, levels=[I_0], colors='blue', linestyles='dashed', linewidths=2.5, alpha=0.5, zorder=3)
    ax2.contour(Q, P, Hm_malla, levels=[H_m_0], colors='black', linestyles='dashed', linewidths=2.5, alpha=0.5,zorder=3)
    trayectoria = np.zeros((pasos, 2))
    p, q = np.log(x[0]), np.log(x[1])
    y_actual = np.array([p, q])

    for i in range(pasos):
        trayectoria[i] = y_actual
        y_actual = symplectic_euler_lv(y_actual, h)

    p_t = trayectoria[:, 0]
    q_t = trayectoria[:, 1]
    u_t = np.exp(p_t)
    v_t = np.exp(q_t)

    ax1.plot(v_t, u_t, color=colores[idx], linewidth=2.5, label=f"Numérico CI={x}",zorder=2)
    ax2.plot(q_t, p_t, color=colores[idx], linewidth=2.5,zorder=2)

ax1.plot([], [], color='blue', linestyle='--', linewidth=2.5, alpha=0.5, label='Flujo Exacto')
ax1.plot([], [], color='black', linestyle='--', linewidth=2.5, alpha=0.5, label='Flujo Modificado')
ax1.set_title("Espacio Biológico (Presas vs Depredadores)")
ax1.set_xlabel("Presas ($v$)");
ax1.set_ylabel("Depredadores ($u$)")
ax1.plot(2, 1, 'kx', markersize=8, label="Equilibrio")
ax1.legend(fontsize=8)

ax2.set_title("Espacio Canónico ($q$ vs $p$)")
ax2.set_xlabel("$q$")
ax2.set_ylabel("Momento $p$")
ax2.plot(np.log(2), np.log(1), 'kx', markersize=8)

plt.tight_layout()
plt.show()