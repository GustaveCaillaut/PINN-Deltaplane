import numpy as np
import matplotlib.pyplot as plt

def naca2412_points(n=200):
    m = 0.02
    p = 0.4
    t = 0.12

    # cosine spacing: plus de points près du bord d'attaque
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))

    yt = 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)

    mask = x < p
    yc[mask] = m / p**2 * (2*p*x[mask] - x[mask]**2)
    dyc[mask] = 2*m / p**2 * (p - x[mask])

    yc[~mask] = m / (1-p)**2 * ((1 - 2*p) + 2*p*x[~mask] - x[~mask]**2)
    dyc[~mask] = 2*m / (1-p)**2 * (p - x[~mask])

    theta = np.arctan(dyc)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)

    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # contour fermé : upper du bord d'attaque vers bord de fuite,
    # puis lower du bord de fuite vers bord d'attaque
    X = np.concatenate([xu, xl[::-1]])
    Y = np.concatenate([yu, yl[::-1]])

    # centre autour de x=0 pour plus pratique
    X = X - 0.5

    return X, Y

X, Y = naca2412_points()

plt.figure(figsize=(8, 3))
plt.plot(X, Y, "-o", markersize=2)
plt.axis("equal")
plt.grid()
plt.show()