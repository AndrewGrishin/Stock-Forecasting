import numpy as np
import matplotlib.pyplot as plt

class SSA:

    @staticmethod
    def x_to_Hankelian(x, L, K):
        """
        :param x: initial times series
        :param L: window size
        :param K: number of elements in rows
        :return: trajectory matrix (sometimes it is called Hankel's matrix of Hankelian)
        """
        X = np.zeros((L, K))
        for i in range(L):
            X[i] = x[i: i + K]
        return X

    @staticmethod
    def Hankelian_to_TS(X):
        """
        :param X: Gets matrix (L, K)
        :return: times series, computed by means of anti-diagonals
        """
        X = X[::-1]
        X = np.array([X.diagonal(i).mean() for i in range(-X.shape[0] + 1, X.shape[1])])
        return X

    @staticmethod
    def get_elementry_matrix(s, u, v):
        """
        :param s: j-th singular value
        :param u: j-th vector in U
        :param v: j-th vector in V
        :return: evelementary matrix
        """
        return s * np.outer(u, v)

    @staticmethod
    def get_contribution(Sigma):
        """
        :param Sigma: vector of singular values
        :return: (contribution of each eigen value = (singular value) ** 2, cumulative contribution)
        """
        Sigma = np.power(Sigma, 2)
        tmp = Sigma / Sigma.sum()
        return tmp, tmp.cumsum()

    @staticmethod
    def get_W_corr_matrix(x_hat, d, K, L):
        """
        :param x_hat: reconstructed parts of the time series
        :param d: rank of matrix X (number of non-zero singular values)
        :param K: N - L + 1
        :param L: window size
        :return: matrix of weighted correlation between reconstructed vectors.
            $${
                WCorr_{i, j} = \frac{\left(\hat{x}_i, \hat{x}_j \right)_w}{\sqrt{\lVert \hat{x}_j \rVert_w \cdot \lVert \hat{x}_i \rVert_w} }: \lVert \hat{x}_i \rVert_w =  \left(\hat{x}_i, \hat{x}_i \right)_w = \sum_{k = 1}^K w_k \cdot \hat{x}_{i,k} \cdot \hat{x}_{i,k}
            }$$
        """
        # vector of weights (number of which each element in initial TS x meets in X)
        w = np.array(list(np.arange(L) + 1) + [L] * (K - 1 - L) + list(np.arange(L) + 1)[::-1])
        def w_dot_product(a, b, w= w):
            return np.sum(w * a * b)

        W_corr = np.zeros((d, d))
        for i in range(L):
            for j in range(i, L):
                enumerator = w_dot_product(x_hat[i], x_hat[j], w)
                denominator = (w_dot_product(x_hat[i], x_hat[i], w) * w_dot_product(x_hat[j], x_hat[j], w))**(0.5)
                W_corr[i, j] =  enumerator / denominator
                W_corr[j, i] = W_corr[i, j]

        return W_corr

    @staticmethod
    def plot_W_corr(W_corr):
        """
        :param W_corr: correlation matrix of reconstructed components
        :return: Nothing. Plots color map of correlation.
        """
        plt.figure(figsize= (18, 9))
        axis = plt.imshow(W_corr)
        plt.colorbar(axis.colorbar);

    @staticmethod
    def plot_x_hat(x_hat, x):
        """
        :param x_hat: reconstructed TS in decomposed view
        :param x: initial TS
        :return: Nothing. Plots all x_hat and x
        """
        plt.figure(figsize= (18, 7))
        for i in range(x_hat.shape[0]):
            plt.plot(x_hat[i], label= f"Component: {i}")

        plt.plot(x, label= "True series", alpha= 0.5)
        plt.legend(loc= "best")
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_contribution(sigma_contribution):
        """
        :param sigma_contribution: (normed to 1 contribution of each singular value, cumsum of normed contribution)
        :return: Nothing. Plots 2 figures (Singular contribution, Cumsum of singular contribution)
        """
        fig, axis = plt.subplots(nrows= 1, ncols= 2, figsize= (18, 5))
        axis[0].set_title("Contribution")
        axis[0].plot(sigma_contribution[0])
        axis[0].scatter(list(np.arange(len(sigma_contribution[0]))), sigma_contribution[0])
        axis[0].grid(True)

        axis[1].set_title("Cumulative Contribution")
        axis[1].plot(sigma_contribution[1])
        axis[1].scatter(list(np.arange(len(sigma_contribution[1]))), sigma_contribution[1])
        axis[1].grid(True)

        plt.show()

    @staticmethod # Спросить о правильности!!!!!
    def forecast(Q, reconstructed_values, r, U, N, L):
        """
        :param Q: steps to forecast
        :param reconstructed_values: reconstruction of initial TS
        :param r: number of non-zero singular values
        :param U: orthonormal matrix in SVD decomposition (left side) U Sigma V^T
        :param N: length of initial TS
        :return: np.array() -> shape (Q,) - forecast
        """
        U_underline = U[:, 0:r]
        verticality_coef = np.power(U_underline[-1, :], 2).sum()
        R = np.sum(U_underline[-1, :] * U_underline[0:-1, :], axis= 1) / (1 - verticality_coef)
        R = R[::-1]

        for i in range(N, N + Q):
            s = 0
            for j in range(L - 1):
                s += (R[j] * reconstructed_values[i - j - 1])
            reconstructed_values.append(s)

        forecasted = np.array(reconstructed_values[N:])
        return forecasted

    @staticmethod
    def plot_prediction(reconstructed_values, forecasted, x):

        plt.figure(figsize= (18, 8))
        plt.plot(np.concatenate([np.array(reconstructed_values), forecasted], axis= 0), label= "Reconstruction + forecast", c= "orange")
        plt.plot(np.arange(len(x)),x, label= "Initial x")
        plt.legend(loc= "best")
        plt.grid(True)
        plt.show()