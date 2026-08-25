import numpy as np
import integradores
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def V(q):
    return 0.5*(q[0]**2+q[1]**2)+q[0]**2*q[1]-q[1]**3/3.0
def F(q):
    arr_F=np.zeros(2)
    arr_F[0]=-q[0]-2*q[1]*q[0]
    arr_F[1]=-q[1]-q[0]**2+q[1]**2
    return arr_F

#Para aplicar Newton-Raphson para métodos implícitos, se necesita dF/dq=JF

def dF(q):
    JF=np.zeros((2,2))
    JF[0][0]=-1-2*q[1]
    JF[0][1]=-2*q[0]
    JF[1][0]=-2*q[0]
    JF[1][1]=-1+2*q[1]
    return JF


def H(p,q):
    return 0.5*np.sum(p**2)+V(q)

# PÁRAMETROS DE LA SIMULACIÓN
h=1e-5
pasos=int(np.round(225/h))

#FUNCIONES PARA LOS INTEGRADORES
F_lambda= lambda q:F(q)
dF_lambda = lambda q:dF(q)


#SIMULACIÓN
#Condiciones iniciales
H0=1.0/12.0
q10=0
q20,p20=0,0
U0=V([q10,q20])
p10=np.sqrt(2*H0-2*U0-p20**2)
p0=[p10,p20]
q0=[q10,q20]
y0 = np.concatenate([p0, q0])
y_actual=y0.copy()
Poincare=[]
#COMO NOS INTERESA SOLO LOS POINCARE CUTS, VAMOS A ESTUDIAR CUANDO CRUZA EL PLANO Q1=0 EN SENTIDO ASCENDENTE
#Para ello, basta ver que q1 pase de negativo a positivo e interpolar mediante interpolación lineal el punto de corte
"""
for i in range(pasos):
    y_new=integradores.stormer_verlet(y_actual,h,F_lambda,dF_lambda)
    q1_old=y_actual[2]
    q1_new=y_new[2]
    if q1_old < 0.0 and q1_new >= 0.0:
        #0=q1_old+(q1_new-q1_old)*s
        s = -q1_old / (q1_new - q1_old)
        q2c = y_actual[3] + s * (y_new[3] - y_actual[3])
        p2c = y_actual[1] + s * (y_new[1] - y_actual[1])
        Poincare.append([q2c, p2c])
    y_actual=y_new

Poincare=np.array(Poincare)

q2=Poincare[:,0]
p2=Poincare[:,1]
"""

def poincare(q20, p20, H0, h=1e-3, n_cuts=400):
    U0 = V([0.0, q20])
    arg = 2*H0 - 2*U0 - p20**2
    if arg < 0:
        return None            # <-- condición inicial NO permitida a esta energía
    p10 = np.sqrt(arg)
    y_actual = np.array([p10, p20, 0.0, q20])
    Poincare = []
    while len(Poincare) < n_cuts:
        y_new = integradores.implicit_Euler(y_actual, h, F_lambda, dF_lambda)
        q1_old = y_actual[2]
        q1_new = y_new[2]
        if q1_old < 0.0 and q1_new >= 0.0:
            # 0=q1_old+(q1_new-q1_old)*s
            s = -q1_old / (q1_new - q1_old)
            q2c = y_actual[3] + s * (y_new[3] - y_actual[3])
            p2c = y_actual[1] + s * (y_new[1] - y_actual[1])
            Poincare.append([q2c, p2c])
        y_actual = y_new
    return np.array(Poincare)


condiciones = [(0.0, 0.0), (0, -0.15), (0, 0.405),
               (0.0, 0.12), (0.0, 0.15), (0, 0.32)]   # prueba y ajusta

plt.figure(figsize=(7, 7))
for (q20, p20) in condiciones:
    cortes = poincare(q20, p20, 1/8)
    if cortes is not None:
        plt.plot(cortes[:, 0], cortes[:, 1], '.', ms=1.5)   # un color por órbita
plt.xlabel("$q_2$"); plt.ylabel("$p_2$")
plt.title("Poincaré cuts - Hénon-Heiles Model - Implicit Euler. H=1/8 (Fig 3.3)")
plt.gca().set_aspect('equal')        # que la curva no salga deformada
plt.grid(alpha=0.3)
plt.show()



