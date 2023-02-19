import numpy as np
import torch


class Metrics:
    def __init__(self):
        self.metrics_dict = {
            "MAPE": self.MAPE,
            "sMAPE": self.sMAPE,
            "WAPE": self.WAPE,
            "RMSE": self.RMSE,
        }

    def __getitem__(self, type):
        assert type in self.metrics_dict.keys(), "Unknown metric"
        return self.metrics_dict[type]()

    class sMAPE:
        def __init__(self):
            self.name = "sMAPE"

        def __call__(self, y_preds, y_true, eps=1e-7):
            y_pr = y_preds
            y_tr = y_true
            if (not isinstance(y_preds, np.ndarray)) and (not isinstance(y_true, np.ndarray)):
                y_pr = y_preds.detach().cpu().numpy()
                y_tr = y_true.detach().cpu().numpy()

            ans = 100 * np.mean(np.abs(y_pr - y_tr) / (eps + np.abs(y_pr + y_tr) / 2))
            return ans

        def __str__(self):
            return self.name

    class WAPE:
        def __init__(self):
            self.name = "WAPE"

        def __call__(self, y_preds, y_true, eps=1e-7):
            y_pr = y_preds
            y_tr = y_true
            if (not isinstance(y_preds, np.ndarray)) and (not isinstance(y_true, np.ndarray)):
                y_pr = y_preds.detach().cpu().numpy()
                y_tr = y_true.detach().cpu().numpy()
            ans = 100 * np.sum(np.abs(y_pr - y_tr)) / (np.sum(np.abs(y_tr)) + eps)

            return ans

        def __str__(self):
            return self.name

    class MAPE:
        def __init__(self):
            self.name = "MAPE"

        def __call__(self, y_preds, y_true, eps=1e-7):
            y_pr = y_preds
            y_tr = y_true
            if (not isinstance(y_preds, np.ndarray)) and (not isinstance(y_true, np.ndarray)):
                y_pr = y_preds.detach().cpu().numpy()
                y_tr = y_true.detach().cpu().numpy()

            ans = 100 * np.mean(np.abs((y_tr - y_pr) / (y_tr + eps)))

            return ans

        def __str__(self):
            return self.name


    class RMSE:
        def __init__(self):
            self.name = "RMSE"

        def __call__(self, y_preds, y_true):
            y_pr = y_preds
            y_tr = y_true
            if (not isinstance(y_preds, np.ndarray)) and (not isinstance(y_true, np.ndarray)):
                y_pr = y_preds.detach().cpu().numpy()
                y_tr = y_true.detach().cpu().numpy()

            ans = np.sqrt(np.mean((y_tr - y_pr) ** 2)) #100 * np.mean(np.abs((y_tr - y_pr) / (y_tr + eps)))

            return ans

        def __str__(self):
            return self.name