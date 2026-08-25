import os
import numpy as np
import matplotlib.pyplot as plt
import integradores

os.makedirs("../Figuras", exist_ok=True)

mu = 0.005
h = 0.03
T = 200.0

e = 0.6
y0 = np.array([0.0, np.sqrt((1 + e) / (1 - e)), 1 - e, 0.0])

def F(q):
    r2 = q[0] ** 2 + q[1] ** 2;
    r = np.sqrt(r2)
    r3 = r2 * r;
    r5 = r3 * r2
    tf = (1.0 / r3) + (1.5 * mu / r5)
    return np.array([-tf * q[0], -tf * q[1]])


def dF(q):
    r2 = q[0] ** 2 + q[1] ** 2;
    r = np.sqrt(r2)
    r3 = r2 * r;
    r5 = r3 * r2;
    r7 = r5 * r2
    tf = (1.0 / r3) + (1.5 * mu / r5)
    tj = (3.0 / r5) + (7.5 * mu / r7)
    JF = np.zeros((2, 2))
    JF[0, 0] = tj * q[0] ** 2 - tf;
    JF[0, 1] = tj * q[0] * q[1]
    JF[1, 0] = JF[0, 1];
    JF[1, 1] = tj * q[1] ** 2 - tf
    return JF


def r_of(q): return np.sqrt(q[0] ** 2 + q[1] ** 2)


def V(q):    r = r_of(q); return -1.0 / r - mu / (2 * r ** 3)


def H(y):    p, q = y[0:2], y[2:4]; return 0.5 * np.sum(p ** 2) + V(q)


def L(y):    p, q = y[0:2], y[2:4]; return q[0] * p[1] - q[1] * p[0]


def gradH(y): p, q = y[0:2], y[2:4]; return np.concatenate([p, -F(q)])


def gradL(y): p, q = y[0:2], y[2:4]; return np.array([-q[1], q[0], p[1], -p[0]])


def arr_H(y):  return np.array([H(y)])


def gr_H(y):   return np.array([gradH(y)])


def arr_HL(y): return np.array([H(y), L(y)])


def gr_HL(y):  return np.array([gradH(y), gradL(y)])

def run_simulation(metodo, proj=None):
    y = y0.copy()
    H0, L0 = H(y), L(y)
    pasos = int(round(T / h))

    traj = np.zeros((pasos + 1, 4))
    traj[0] = y

    n_ok = pasos
    fallo = None

    for i in range(1, pasos + 1):
        try:
            if proj is None:
                y = metodo(y, h, F, dF)
            elif proj == 'H':
                y = integradores.paso_proyectado(y, h, metodo, F, dF, arr_H, gr_H, np.array([H0]))
            elif proj == 'HL':
                y = integradores.paso_proyectado(y, h, metodo, F, dF, arr_HL, gr_HL, np.array([H0, L0]))
        except RuntimeError as err:
            n_ok = i - 1;
            fallo = str(err);
            break

        if not np.all(np.isfinite(y)):
            n_ok = i - 1;
            fallo = "Overflow/NaN";
            break

        traj[i] = y

    return traj[:n_ok + 1], n_ok, fallo


metodos = {
    "SymplecticEuler": integradores.symplectic_euler,
    "ImplicitEuler": integradores.implicit_Euler,
    "ExplicitEuler": integradores.explicit_Euler,
    "ImplicitMidpointRule": integradores.implicit_midpoint,
    "StörmerVerlet": integradores.stormer_verlet,
}

proys = [None, 'H', 'HL']
titproj = {
    None: "Sin proyección",
    'H': "Proyección sobre $H$",
    'HL': "Proyección sobre $H$ y $L$"
}

print("Iniciando simulación de Kepler Perturbado...")

for nombre, metodo in metodos.items():
    print(f"Calculando y graficando método: {nombre}...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Método: {nombre}", fontsize=14, fontweight='bold')

    for j, proj in enumerate(proys):
        ax = axes[j]
        traj, n_ok, fallo = run_simulation(metodo, proj)

        q1, q2 = traj[:, 2], traj[:, 3]

        ax.plot(q1, q2, lw=0.6, color="navy", label="Trayectoria")
        ax.scatter(0, 0, color="orange", s=60, label="Sol", zorder=3)

        estado = titproj[proj]
        if fallo:
            estado += f"\n(FALLO en t={n_ok * h:.1f})"
            print(f"  -> {nombre} ({proj}) falló: {fallo}")

        ax.set_title(estado, fontsize=11)
        ax.set_xlabel(r"$q_1$")
        ax.set_ylabel(r"$q_2$")
        ax.set_aspect('equal')
        ax.set_xlim(-2, 1.6)
        ax.set_ylim(-1.8, 1.8)
        ax.grid(True, alpha=0.5)
        if j == 0:
            ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(f"../Figuras/PerturbedKepler-{nombre}.png", dpi=300, bbox_inches="tight")
    plt.close()

print("Simulación finalizada. Gráficas guardadas en ../Figuras/")