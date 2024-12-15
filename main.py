import numpy as np
import matplotlib.pyplot as plt

clusters = 4
# x = np.array([[9, 1, -3, -14, -15, 10, -1, 10, 0, 7, -11],
#              [-2, 14, 15, 7, 6, -2, 14, -3, 12, -1, 8]]).T
x = np.array([
        (2, -3),
        (1, 5),
        (18, -11),
        (-10, -7),
        (11, -6),
        (19, -11),
        (-19, -16),
        (18, -10),
        (4, -5),
        (9, -3),
        (-8, -5),
        (12, -2),
        (-9, -7),
        (-8, -4),
        (9, -2),
        (-19, -18),
        (17, -10)
    ])


w = np.random.rand(clusters, 2)
w /= np.linalg.norm(w, axis=1)[:, np.newaxis]
init_coef = 0.7

plt.subplot(1, 2, 1)
plt.scatter(x.T[0], x.T[1], color='blue')
plt.scatter(w.T[0], w.T[1], color='red')

print(w)
for iter in range(100):
    coef = init_coef / ((iter + 1)/2)
    # coef = init_coef
    for xi in x:
        distances = np.linalg.norm(w - xi, axis=1)
        winner_index = np.argmin(distances)
        w[winner_index] += coef * (xi - w[winner_index])
        # w[winner_index] /= np.linalg.norm(w[winner_index])
print(f"Веса: {w}")

plt.subplot(1, 2, 2)
plt.scatter(x.T[0], x.T[1], color='blue')
plt.scatter(w.T[0], w.T[1], color='red')
for i in range(clusters):
    plt.quiver(0, 0, w[i][0], w[i][1], angles='xy', scale_units='xy', scale=1, color='black', headwidth=4, headlength=6)

plt.show()