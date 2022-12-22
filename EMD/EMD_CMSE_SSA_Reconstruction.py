"""
### Группировка компонент
Алгоритм основан на [статье](https://link.springer.com/article/10.1007/s00034-018-0861-1). За исключением оптимального алгоритма перебора компонент: он реализован в лоб.
"""

import numpy as np
from PyEMD import EMD

def denoise(x_hat: np.ndarray, series: np.ndarray):
    """
    :param x_hat: SSA components
    :param series: intial series, that we are working with
    :return: reconstructed and denoised series
    """
    emd = EMD()
    imfs = emd(series)

    groups = dict()
    for i in range(len(x_hat)):
        best_corr = -np.inf
        i_best_corr = None

        for j in range(len(imfs)):
            corr = np.abs(np.dot(x_hat[i], imfs[j])) / (np.linalg.norm(x_hat[i]) * np.linalg.norm(imfs[j]))
            if corr > best_corr:
                best_corr = corr
                i_best_corr = j

        if i_best_corr in groups.keys():
            groups[i_best_corr] += x_hat[i]
        else:
            groups[i_best_corr] = x_hat[i]

    groups = np.array([groups[i] for i in sorted(list(groups.keys()))])

    norms = np.linalg.norm(groups, axis= 1) ** 2 / groups.shape[1]
    C = np.argmin(norms[:-2])

    reconstructed_series = groups[:-2][C:].sum(axis= 0) + groups[-2:].sum(axis= 0)

    return reconstructed_series