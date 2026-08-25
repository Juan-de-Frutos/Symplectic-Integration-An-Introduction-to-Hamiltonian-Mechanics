import numpy as np

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

def f(p,q,w):
    return np.concatenate([F(q,w),p])

def explicit_Euler(y,w,h): #Euler Explícito y_n+1=y_n+h*f(y_n)
        m=y.size//4
        p=y[0:2*m]
        q=y[2*m:4*m]
        p_new=p+h*F(q,w)
        q_new=q+h*p
        return np.concatenate([p_new,q_new])

#Para métodos implícitos, necesitamos resolver sistemas de ecuaciones NO lineales por Newton-Raphson
def Newton(xn,JG,G,tol,maxit):
    x = xn.copy()
    for k in range(maxit):
        Gx = G(x)
        if np.max(np.abs(Gx)) < tol:
            return x
        x = x - np.linalg.solve(JG(x), Gx)
    raise RuntimeError(f"Newton no convergió en {maxit} iteraciones (||G||={np.max(np.abs(Gx)):.2e})")

def implicit_Euler(y,w,h,tol=1e-12,maxit=50):
    m=y.size//4
    p,q=y[0:2*m],y[2*m:4*m]
    Id=np.eye(2*m)
    c=q+h*p
    G= lambda x: x-h**2*F(x,w)-c
    JG= lambda x: Id-h**2*dF(x,w)

    q_new=Newton(q,G,JG,tol,maxit)
    p_new=p+h*F(q_new,w)
    return np.concatenate([p_new,q_new])

def implicit_midpoint(y,w,h,tol=1e-12,maxit=50):
    m=y.size//4
    p,q=y[0:2*m],y[2*m:4*m]
    c=q+0.5*h*p
    Id=np.eye(2*m)

    G= lambda x: x-h**2/4*F(x,w)-c
    JG= lambda x: Id-h**2/4*dF(x,w)

    q_m=Newton(q,G,JG,tol,maxit)
    q_new=2*q_m-q
    p_new=p+h*F(q_m,w)

    return np.concatenate([p_new,q_new])

def symplectic_euler(y,w,h):
    m=y.size//4
    p,q=y[0:2*m],y[2*m:4*m]
    p_new=p+h*F(q,w)
    q_new=q+h*p_new
    return np.concatenate([p_new,q_new])

def symplectic_euler_adjoint(y,w,h):
    m=y.size//4
    p,q=y[0:2*m],y[2*m:4*m]
    q_new=q+h*p
    p_new=p+h*F(q_new,w)
    return np.concatenate([p_new,q_new])

# Implementación de Stormer-Verlet como composición de dos Euler Simplécticos

def stormer_verlet_composition(y, w, h):
    # Euler Simpléctico
    y_half = symplectic_euler(y, w, h/2.0)
    # Euler Simpléctico Adjunto
    return symplectic_euler_adjoint(y_half, w, h/2.0)

#Implementación directa de Stormer-Verlet

def stormer_verlet(y, w, h):
    m = y.size // 4
    p,q = y[0:2*m],y[2*m:4*m]
    p_half = p + (h / 2.0) * F(q, w)
    q_new = q + h * p_half
    p_new = p_half + (h / 2.0) * F(q_new, w)

    return np.concatenate([p_new, q_new])