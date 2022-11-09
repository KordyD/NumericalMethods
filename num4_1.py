import numpy as np
import matplotlib.pyplot as plt

# N = 16
# E = 10^-6
# f(x, y, z) = 2x^2 + (3 + 0.1 * 16)y^2 + (4 + 0.1 * 16)z^2 + xy - yz + xz + x - 2y + 3z + 16


def main():
    A = np.array([[4, 1, 1], [1, 9.2, -1], [1, -1, 11.2]])
    b = np.array([1., -2., 3.])
    x_old = np.array([1., 1., 1.])
    x_new = np.array([1., 1., 1.])
    eps = 10**(-6)
    first_pass = True
    n = 0
    while first_pass or np.abs(np.linalg.norm(x_new) - np.linalg.norm(x_old)) > eps:
        x_old = x_new
        _t = np.matmul(A, x_old) + b
        q = np.array([0, 0, 0])
        q[n%3] = 1
        _q = q[:, None]
        mu = - (np.matmul(q, _t)) / np.linalg.multi_dot([q, A, _q])
        x_new = x_old + q * mu 
        first_pass = False
        n += 1
    print(x_new, 
    2 * x_new[0] ** 2 + (3 + 0.1 * 16) * x_new[1] ** 2 + (4 + 0.1 * 16) * x_new[2] ** 2 + x_new[0] * x_new[1] - x_new[1] * x_new[2] + x_new[0] * x_new[2] + x_new[0] - 2 * x_new[1] + 3 * x_new[2] + 16,
    n)

    x = np.linspace(-50, 50, 200)
    plt.plot(x, 2 * x ** 2 + (3 + 0.1 * 16) * x_new[1] ** 2 + (4 + 0.1 * 16) * x_new[2] ** 2 + x * x_new[1] - x_new[1] * x_new[2] + x * x_new[2] + x - 2 * x_new[1] + 3 * x_new[2] + 16)
    y = np.linspace(-50, 50, 200)
    plt.plot(y, 2 * x_new[0] ** 2 + (3 + 0.1 * 16) * y ** 2 + (4 + 0.1 * 16) * x_new[2] ** 2 + x_new[0] * y - y * x_new[2] + x_new[0] * x_new[2] + x_new[0] - 2 * y + 3 * x_new[2] + 16)
    z = np.linspace(-50, 50, 200)
    plt.plot(z, 2 * x_new[0] ** 2 + (3 + 0.1 * 16) * x_new[1] ** 2 + (4 + 0.1 * 16) * z ** 2 + x_new[0] * x_new[1] - x_new[1] * z + x_new[0] * z + x_new[0] - 2 * x_new[1] + 3 * z + 16)
    
    plt.scatter(x_new[0], 
    2 * x_new[0] ** 2 + (3 + 0.1 * 16) * x_new[1] ** 2 + (4 + 0.1 * 16) * x_new[2] ** 2 + x_new[0] * x_new[1] - x_new[1] * x_new[2] + x_new[0] * x_new[2] + x_new[0] - 2 * x_new[1] + 3 * x_new[2] + 16)
    plt.scatter(x_new[1], 
    2 * x_new[0] ** 2 + (3 + 0.1 * 16) * x_new[1] ** 2 + (4 + 0.1 * 16) * x_new[2] ** 2 + x_new[0] * x_new[1] - x_new[1] * x_new[2] + x_new[0] * x_new[2] + x_new[0] - 2 * x_new[1] + 3 * x_new[2] + 16)
    plt.scatter(x_new[2], 
    2 * x_new[0] ** 2 + (3 + 0.1 * 16) * x_new[1] ** 2 + (4 + 0.1 * 16) * x_new[2] ** 2 + x_new[0] * x_new[1] - x_new[1] * x_new[2] + x_new[0] * x_new[2] + x_new[0] - 2 * x_new[1] + 3 * x_new[2] + 16)
    plt.show()
    return


if __name__ == '__main__':
    main()