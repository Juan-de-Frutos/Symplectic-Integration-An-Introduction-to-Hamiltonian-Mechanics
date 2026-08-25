import sympy as sp
import math

# 1. Definir los símbolos
y, h = sp.symbols('y h')

# 2. Definir la función original (y' = y^2) y el número de términos
fcn = y ** 2
nn = 6

# Lista para guardar los coeficientes f_j(y) de la ecuación modificada
# fcoe[1] corresponderá a h^0, fcoe[2] a h^1, etc.
fcoe = [0] * (nn + 1)
fcoe[1] = fcn

# 3. Bucle principal iterativo (equivalente al 'for n from 2 to nn' en Maple)
for n in range(2, nn + 1):
    # Suma de la ecuación modificada truncada calculada hasta ahora
    modeq = sum(h ** j * fcoe[j + 1] for j in range(n - 1))

    # Lista para las derivadas sucesivas
    diffy = [0] * (n + 1)
    diffy[0] = y

    # Calcular las derivadas temporales usando la regla de la cadena
    # y la ecuación diferencial modificada actual
    for i in range(1, n + 1):
        diffy[i] = sp.diff(diffy[i - 1], y) * modeq

    # Calcular la serie de Taylor de y(t+h)
    ytilde = sum((h ** k * diffy[k]) / math.factorial(k) for k in range(n + 1))

    # Calcular el residuo con respecto al método numérico (Euler explícito: y + h*fcn)
    res = ytilde - y - h * fcn

    # Extraer el coeficiente de h^n del residuo y cambiarle el signo
    res_expanded = sp.expand(res)
    tay = sp.series(res_expanded, h, 0, n + 1).removeO()
    fcoe[n] = -tay.coeff(h, n)

# 4. Imprimir la ecuación diferencial modificada resultante
ecuacion_modificada = sum(h ** j * fcoe[j + 1] for j in range(nn))

print("Ecuación modificada:")
sp.pprint(ecuacion_modificada)