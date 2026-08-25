import numpy as np
import integradores
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def V(q):
    return -1/np.sqrt(q[0]**2+q[1]**2)
def F(q):
    arr_F=np.zeros(2)
    arr_F[0]=-q[0]/(q[0]**2+q[1]**2)**1.5
    arr_F[1]=-q[1]/(q[0]**2+q[1]**2)**1.5
    return arr_F

def dF(q):
    q = np.asarray(q, dtype=float)
    r = np.linalg.norm(q)

    I = np.eye(2)
    qqT = np.outer(q, q)

    return -I / r ** 3 + 3 * qqT / r ** 5


def H(p,q):
    return 0.5*np.sum(p**2)+V(q)

# PÁRAMETROS DE LA SIMULACIÓN
h=0.05
pasos=int(np.round(200/h))

#FUNCIONES PARA LOS INTEGRADORES
F_lambda= lambda q:F(q)
dF_lambda = lambda q:dF(q)


e=0.6
q10 = 1 - e
q20 = 0
p10 = 0
p20 = np.sqrt((1 + e) / (1 - e))
y0 = np.array([p10, p20, q10, q20])




metodos = {
    "Symplectic Euler": integradores.symplectic_euler,
    "Implicit Euler": integradores.implicit_Euler,
    "Explicit Euler": integradores.explicit_Euler,
    "Implicit Midpoint Rule": integradores.implicit_midpoint,
    "Störmer-Verlet Scheme": integradores.stormer_verlet,
}

print("Iniciando simulación...")

for nombre, metodo in metodos.items():
    print("Método: ", nombre)
    y = y0.copy()
    trayectoria = np.zeros((pasos, 4))
    pasos_calculados=pasos
    try:
        for i in range(pasos):
            trayectoria[i] = y
            y = metodo(y, h, F_lambda, dF_lambda)
    except RuntimeError as err:
        print(f"  -> {nombre} falló (probablemente colapsó en el Sol): {err}")
        pasos_calculados = i
    P = trayectoria[:pasos_calculados, 0:2]
    Q = trayectoria[:pasos_calculados, 2:4]
    plt.figure(figsize=(6,6))
    plt.plot(Q[:,0], Q[:,1], label=nombre)
    plt.scatter(0, 0, color='orange', s=80, label='Sol')

    plt.title(nombre+"; h="+str(h))
    plt.xlabel(r"$q_1$")
    plt.ylabel(r"$q_2$")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"../Figuras/KeplerProblem-{nombre}-h={h}.png", dpi=300, bbox_inches="tight")
    plt.close()
print("Simulación finalizada")
#plt.show()