# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def y_0(alpha, beta, x):
    return 1 / (1 + np.exp(-(alpha - beta) * x) + np.exp(-alpha * x))


x = np.linspace(-10, 10, 101)
alpha = 1
betas = np.linspace(0, 2, 11)

sns.set_theme()
plt.figure(figsize=(10, 6))

for beta in betas:
    plt.plot(x, y_0(alpha, beta, x))

plt.legend([f"$\\beta = {beta:.1f}$" for beta in betas])
plt.xlabel('$x$')
plt.ylabel('$\\tilde{y}_{1, \\beta}(x, 0)$')
plt.savefig('example.pdf', bbox_inches='tight')
plt.show()
plt.close()

# %%
