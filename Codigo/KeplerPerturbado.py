import numpy as np
import matplotlib.pyplot as plt
import integradores

mu = 0.005

def F(q):
    r2 = q[0]**2 + q[1]**2; r = np.sqrt(r2)
    r3 = r2*r; r5 = r3*r2
    tf = (1.0/r3) + (1.5*mu/r5)
    return np.array([-tf*q[0], -tf*q[1]])

def dF(q):
    r2 = q[0]**2 + q[1]**2; r = np.sqrt(r2)
    r3 = r2*r; r5 = r3*r2; r7 = r5*r2
    tf = (1.0/r3) + (1.5*mu/r5)
    tj = (3.0/r5) + (7.5*mu/r7)
    JF = np.zeros((2,2))
    JF[0,0] = tj*q[0]**2 - tf; JF[0,1] = tj*q[0]*q[1]
    JF[1,0] = JF[0,1];         JF[1,1] = tj*q[1]**2 - tf
    return JF

def r_of(q): return np.sqrt(q[0]**2 + q[1]**2)
def V(q):    r = r_of(q); return -1.0/r - mu/(2*r**3)
def H(y):    p,q = y[0:2],y[2:4]; return 0.5*np.sum(p**2)+V(q)
def L(y):    p,q = y[0:2],y[2:4]; return q[0]*p[1]-q[1]*p[0]
def gradH(y):p,q = y[0:2],y[2:4]; return np.concatenate([p, -F(q)])
def gradL(y):p,q = y[0:2],y[2:4]; return np.array([-q[1],q[0],p[1],-p[0]])
def arr_H(y):  return np.array([H(y)])
def gr_H(y):   return np.array([gradH(y)])
def arr_HL(y): return np.array([H(y), L(y)])
def gr_HL(y):  return np.array([gradH(y), gradL(y)])

# CONDICIONES INICIALES
e = 0.6
y0 = np.array([0.0, np.sqrt((1+e)/(1-e)), 1-e, 0.0])

def run(metodo, proj=None, h=0.03, T=400.0):
    y = y0.copy(); H0, L0 = H(y), L(y)
    pasos = int(round(T/h))
    traj = np.zeros((pasos+1, 4)); traj[0] = y
    dH = np.zeros(pasos+1); dL = np.zeros(pasos+1)
    n_ok = pasos; fallo = None
    for i in range(1, pasos+1):
        try:
            if   proj is None: y = metodo(y, h, F, dF)
            elif proj == 'H':  y = integradores.paso_proyectado(y, h, metodo, F, dF, arr_H,  gr_H,  np.array([H0]))
            elif proj == 'HL': y = integradores.paso_proyectado(y, h, metodo, F, dF, arr_HL, gr_HL, np.array([H0, L0]))
        except RuntimeError as err:                       # Newton no converge
            n_ok = i-1; fallo = str(err); break
        if not np.all(np.isfinite(y)):                    # overflow
            n_ok = i-1; fallo = "valores no finitos (overflow)"; break
        traj[i] = y; dH[i] = H(y)-H0; dL[i] = L(y)-L0
    traj, dH, dL = traj[:n_ok+1], dH[:n_ok+1], dL[:n_ok+1]
    return dict(traj=traj, dH=dH, dL=dL, H0=H0, L0=L0, h=h, T=T,
                n_ok=n_ok, pasos=pasos, fallo=fallo)

metodos = {
    "Symplectic Euler":       integradores.symplectic_euler,
    "Implicit Euler":         integradores.implicit_Euler,
    "Explicit Euler":         integradores.explicit_Euler,
    "Implicit Midpoint Rule": integradores.implicit_midpoint,
    "Stormer-Verlet":         integradores.stormer_verlet,
}
proys   = [None, 'H', 'HL']
titproj = {None:"sin proyeccion", 'H':"proy. sobre H", 'HL':"proy. sobre H y L"}

print(f"{'metodo':24s} {'proj':5s} {'max|dH|':>11s} {'max|dL|':>11s}  estado")
for nombre, metodo in metodos.items():
    for proj in proys:
        r = run(metodo, proj)
        estado = "ok" if r['fallo'] is None else f"FALLO en t={r['n_ok']*r['h']:.2f}"
        print(f"{nombre:24s} {str(proj):5s} "
              f"{np.max(np.abs(r['dH'])):11.3e} {np.max(np.abs(r['dL'])):11.3e}  {estado}")

nf = len(metodos)
fig, axes = plt.subplots(nf, 3, figsize=(10, 3.1*nf))
for i,(nom,met) in enumerate(metodos.items()):
    for j,proj in enumerate(proys):
        r = run(met, proj)
        q1, q2 = r['traj'][:,2], r['traj'][:,3]
        ax = axes[i,j]
        ax.plot(q1, q2, lw=0.4, color="navy")
        ax.plot(0, 0, 'o', color="orange", ms=5)
        sub = titproj[proj] if r['fallo'] is None else f"{titproj[proj]}\nFALLO t={r['n_ok']*r['h']:.1f}"
        ax.set_title(f"{nom} - {sub}", fontsize=8)
        ax.set_aspect('equal')
        ax.set_xlim(-2.3, 1.3); ax.set_ylim(-1.8, 1.8)
        ax.tick_params(labelsize=6)
fig.suptitle("Kepler perturbado - orbitas $(q_1,q_2)$, h=0.03, T=200", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.985])

fig2, (a1,a2) = plt.subplots(1, 2, figsize=(13, 4.5))
for nom, met in metodos.items():
    r = run(met, None)
    t = np.arange(len(r['dH'])) * r['h']
    a1.semilogy(t, np.abs(r['dH'])+1e-16, lw=0.8, label=nom)
    a2.semilogy(t, np.abs(r['dL'])+1e-16, lw=0.8, label=nom)
a1.set_title("$|H(t)-H_0|$  (sin proyeccion)"); a1.set_xlabel("t"); a1.legend(fontsize=8)
a2.set_title("$|L(t)-L_0|$  (sin proyeccion)"); a2.set_xlabel("t"); a2.legend(fontsize=8)
for a in (a1,a2): a.grid(alpha=0.3, which='both')
fig2.suptitle("Conservacion de invariantes por metodo", fontsize=12)
fig2.tight_layout(rect=[0,0,1,0.95])

plt.show()