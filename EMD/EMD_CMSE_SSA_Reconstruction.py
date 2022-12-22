"""
### Группировка компонент
Алгоритм основан на [статье](https://link.springer.com/article/10.1007/s00034-018-0861-1). За исключением оптимального алгоритма перебора компонент: он реализован в лоб.
"""

import numpy as np
from PyEMD import EMD

def denoise(x_hat: np.ndarray, series: np.ndarray):
    #     """
    #     :param x_hat: SSA components
    #     :param series: intial series, that we are working with
    #     :return:
    #       1.  reconstructed and denoised series
    #       2. (IMFs, CMSEs)
    #     """
    emd = EMD()
    imfs = emd(series)

    groups = {i: [] for i in range(len(imfs))}

    for i in range(len(x_hat)):
        best_corr = -np.inf
        i_best_corr = None

        for j in range(len(imfs)):
            corr = np.abs(np.dot(x_hat[i], imfs[j])) / (np.linalg.norm(x_hat[i]) * np.linalg.norm(imfs[j]))
            if corr > best_corr:
                best_corr = corr
                i_best_corr = j

        if len(groups[i_best_corr]) != 0:
            groups[i_best_corr] += x_hat[i]
        else:
            groups[i_best_corr] = x_hat[i]

    groups = [groups[i] for i in sorted(list(groups.keys()))]

    # Determine C constant. Index of tendency change in power of IMFs.
    norms = np.linalg.norm(imfs, axis= 1) ** 2 / imfs.shape[1]
    C = np.argmin(norms[:-2])

    reconstructed_series = np.zeros(imfs.shape[1])

    for array in groups[C:]:
        if len(array) != 0:
            reconstructed_series += array

    return reconstructed_series, (imfs, norms)