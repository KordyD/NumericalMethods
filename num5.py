# tg(3x) + 0.4 - x^2
# -pi/6 +pi/6

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

xp = []
yp = []
n = 10
a = -np.pi / 6
b = +np.pi / 6
_t= (b - a) / n

for i in range(n + 1):
    xp.append(a + i * _t)
    #xp.append(0.5 * ((b - a) * np.cos((2 * i + 1)/(2 * n + 2) * np.pi) + b + a))
    #yp.append(np.tan(3 * xp[i]) + 0.4 - xp[i] ** 2)
    yp.append(np.abs(xp[i]) * (np.tan(3 * xp[i]) + 0.4 - xp[i] ** 2))

x = sp.Symbol('x')
y = 0
for i in range(n+1):
    p = 1
    for j in range(n+1):
        if j != i:
            p *= (x - xp[j]) / (xp[i] - xp[j])
    y += yp[i] * p
y = sp.simplify(y)
print(y)
xplt = np.linspace(xp[0], xp[-1])

# for xt in xplt:
#     yt = 0
#     for xi, yi, in zip(xp, yp):
#         yt += yi * np.prod((xt - xp[xp != xi]) / (xi - xp[xp != xi]))
#     yplt = np.append(yplt,yt)

#plt.plot(xplt, np.tan(3 * xplt) + 0.4 - np.power(xplt, 2))
plt.plot(xplt, np.abs(xplt) * (np.tan(3 * xplt) + 0.4 - np.power(xplt, 2)))
plt.plot(xp, yp, 'ro')
plt.show()