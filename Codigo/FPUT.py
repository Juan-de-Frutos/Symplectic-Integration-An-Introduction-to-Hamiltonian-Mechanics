import numpy as np
import integradores
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def V(q, w):
    m=q.size//2
    Q=np.zeros(2*m+2)
    Q[1:2*m+1]=q # Q[k]=q_k, con Q[0]=Q[2m+1]=0

    #Resortes Rígidos Lineales
    a1= Q[1:2*m:2]  #q_1, q_3, ... , q_2m-1
    b1= Q[2:2*m+1:2] #q_2, q_4, ... , q_2m
    V_stiff= (w**2/4.0)*np.sum((b1-a1)**2)

    #Resortes Blandos No Lineales

    a2 = Q[0:2 * m+1:2]  # q_0, q_2, ... , q_2m
    b2 = Q[1:2 * m + 2:2]  # q_1, q_3, ... , q_2m+1
    V_soft = np.sum((b2 - a2) ** 4)

    return V_soft+V_stiff

def F(q,w):
    m=q.size//2
    Q=np.zeros(2*m+2)
    Q[1:2*m+1]=q
    arr_F=np.zeros(2*m+2)

    #Resortes Rígidos
    a1=np.arange(1,2*m,2)
    b1=a1+1
    F1=(w**2/2.0)*(Q[b1] - Q[a1])
    np.add.at(arr_F,a1,F1)
    np.add.at(arr_F,b1,-F1)

    #Resortes Blandos
    a2=np.arange(0,2*m+1,2)
    b2=a2+1
    F2=4.0*(Q[b2]-Q[a2])**3
    np.add.at(arr_F,a2,F2)
    np.add.at(arr_F,b2,-F2)
    return arr_F[1:2*m+1]

#Para aplicar Newton-Raphson para métodos implícitos, se necesita dF/dq=JF

def dF(q,w):
    m=q.size//2
    Q=np.zeros(2*m+2)
    Q[1:2*m+1]=q
    JF=np.zeros((2*m+2,2*m+2))

    #Resortes Rígidos
    d1=w**2 / 2.0
    a1 = np.arange(1, 2 * m, 2);
    b1 = a1 + 1
    np.add.at(JF, (a1, a1), -d1);
    np.add.at(JF, (a1, b1), d1)
    np.add.at(JF, (b1, a1), d1);
    np.add.at(JF, (b1, b1), -d1)

    #Resortes Blandos:
    a2 = np.arange(0, 2 * m + 1, 2);
    b2 = a2 + 1
    d2 = 12.0 * (Q[b2] - Q[a2]) ** 2
    np.add.at(JF, (a2, a2), -d2);
    np.add.at(JF, (a2, b2), d2)
    np.add.at(JF, (b2, a2), d2);
    np.add.at(JF, (b2, b2), -d2)

    return JF[1:2*m+1,1:2*m+1]


def H(p,q,w):
    return 0.5*np.sum(p**2)+V(q,w)

"""
def f(p,q,w):
    return np.concatenate([F(q,w),p])
    """
# PÁRAMETROS DE LA SIMULACIÓN
w_sim=50
m=3
h=0.03
pasos=int(np.round(225/h))

#FUNCIONES PARA LOS INTEGRADORES
F_lambda= lambda q:F(q,w=w_sim)
dF_lambda = lambda q:dF(q,w=w_sim)


#SIMULACIÓN
#Condiciones iniciales
"""
# 1. Excitación del Modo Fundamental (k=1)
k = 1
A = 1.0
j = np.arange(1, 2*m + 1)
L = 2*m + 1
"""

q0 = np.zeros(2*m)
p0 = np.zeros(2*m)

# Condiciones iniciales de la pág 22
# x0_1 = 1, y0_1 = 1, x1_1 = 1/w, y1_1 = 1
q0[0] = (1.0 - 1.0/w_sim) / np.sqrt(2.0)
q0[1] = (1.0 + 1.0/w_sim) / np.sqrt(2.0)

p0[0] = (1.0 - 1.0) / np.sqrt(2.0)
p0[1] = (1.0 + 1.0) / np.sqrt(2.0)

y0 = np.concatenate([p0, q0])
trayectoria=np.zeros((pasos, y0.size))
trayectoria[0]=y0
y_actual=y0.copy()

for i in range(pasos):
    y_new=integradores.stormer_verlet(y_actual,h,F_lambda,dF_lambda)
    trayectoria[i]=y_new
    y_actual=y_new

P_t = trayectoria[:, 0:2*m]
Q_t = trayectoria[:, 2*m:4*m]
H_t = np.array([H(P_t[k], Q_t[k], w_sim) for k in range(pasos)])

tiempo = np.arange(pasos) * h
plt.figure(figsize=(12, 5))

#GRAFICAS DE RESULTADOS, VALOR ABSOLUTO DE H Y ERROR RELATIVO

P_t = trayectoria[:, 0:2 * m]
Q_t = trayectoria[:, 2 * m:4 * m]

H_t = np.array([H(P_t[k], Q_t[k], w_sim) for k in range(pasos)])
tiempo = np.arange(pasos) * h

# Gráfica 1: Valor absoluto de H
plt.subplot(1, 2, 1)
plt.plot(tiempo, H_t, label="H(t)", color="navy", linewidth=0.5)
plt.title("Evolución del Hamiltoniano Total")
plt.xlabel("Tiempo")
plt.ylabel("Energía H")

H_media = np.mean(H_t)
margen = 0.10 * np.abs(H_media) if H_media != 0 else 1.0
plt.ylim(H_media - margen, H_media + margen)

plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

# Gráfica 2: Error relativo de H
if H_t[0] != 0:
    error_relativo = (H_t - H_t[0]) / H_t[0]
    plt.subplot(1, 2, 2)
    plt.plot(tiempo, error_relativo, label="ΔH / H₀", color="crimson", linewidth=0.5, alpha=0.8)
    plt.title("Error Relativo de la Energía")
    plt.xlabel("Tiempo")
    plt.ylabel("(H(t) - H₀) / H₀")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

#Análisis de las energías oscilatorias de cada oscilador Ij
#Cálculo de x e y
P_t = trayectoria[:, 0:2*m]
Q_t = trayectoria[:, 2*m:4*m]
q_odd = Q_t[:, 0::2]
q_even = Q_t[:, 1::2]

p_odd = P_t[:, 0::2]
p_even = P_t[:, 1::2]

# Ecuación 5.1
x1 = (q_even - q_odd) / np.sqrt(2.0)
y1 = (p_even - p_odd) / np.sqrt(2.0)

# Ecuación 5.3: Energía oscilatoria de cada resorte rígido j
I_j = 0.5 * (y1**2 + (w_sim**2) * x1**2)

# Energía oscilatoria total (suma de los m resortes)
I_total = np.sum(I_j, axis=1)

# GRÁFICA DEL INTERCAMBIO DE ENERGÍA OSCILATORIA

# ==========================================
# CÁLCULO DE ENERGÍAS
# ==========================================
# Evaluamos el Hamiltoniano exacto en cada instante de tiempo
H_t = np.array([H(P_t[k], Q_t[k], w_sim) for k in range(pasos)])
H_t-=0.8
plt.figure(figsize=(12, 6))

#plt.plot(tiempo, H_t, label="$H(t)-0.8$", color="crimson", linewidth=2.5, zorder=6)
plt.plot(tiempo, I_total, label="$I(t)$", color="black", linestyle="--", linewidth=2.5, zorder=5)
colores = cm.tab20(np.linspace(0, 1, m))
for j in range(m):
    plt.plot(tiempo, I_j[:, j], label=f"$I_{{{j+1}}}(t)$", color=colores[j], linewidth=1.0, alpha=0.8)

titulo = (f"FPUT - Störmer-Verlet\n"
          f"$h = {h}$")
plt.title(titulo, fontsize=14, pad=15)

plt.xlabel("Time", fontsize=12)
plt.ylabel("Energy", fontsize=12)

# Ajustamos el límite Y dinámicamente para que quepan tanto H_t como I_total
max_y = max(np.max(H_t), np.max(I_total))
plt.ylim(0, max_y * 1.1)

plt.grid(True, linestyle="--", alpha=0.6)

# Colocamos la leyenda fuera de la gráfica para no tapar los datos
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=10, ncol=2)

plt.tight_layout()
plt.show()

