import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm, trange

class Train_Eval_Plot:

    @staticmethod
    def train(model, criterion, metric, optimizer, epochs, train_data, verbose= True, lag= 1, metric_to_max= False, device= "cpu"):

        model = model.to(device)
        train_log = []
        val_log = []

        train_metric_log = []
        val_metric_log = []

        best_model = None
        best_eval_metric_log = []
        best_eval_metric = torch.inf * ((-1) ** int(metric_to_max))
        best_iter = None

        train_loader = train_data["train"]
        val_loader = train_data["val"]

        # X_train = X_train.to(device)
        # X_val = X_val.to(device)
        #
        # y_train = y_train.to(device)
        # y_val = y_val.to(device)

        epochs_bar = tqdm(range(epochs)) if not verbose else range(epochs)

        for epoch in epochs_bar:

            avg_train_loss = 0
            avg_train_metric = 0
            train_len = 0
            standard_len = None


            # train_pbar = tqdm(enumerate(train_loader))
            model.train()
            for ind, (X_train_bt, y_train_bt) in enumerate(train_loader):
                # train_pbar.set_description(f"Train: {ind + 1}/{len(train_loader)}")
                optimizer.zero_grad()

                train_preds = model(X_train_bt.to(device), device= device)
                train_loss = criterion(train_preds, y_train_bt.to(device))
                train_metric = metric(train_preds, y_train_bt.to(device))

                # ingroup
                avg_train_loss += train_loss.item()
                avg_train_metric += train_metric

                if standard_len is None:
                    standard_len = X_train_bt.shape[0]

                train_len += (X_train_bt.shape[0] / standard_len)

                model.train()
                train_loss.backward()
                optimizer.step()

            # over group
            avg_train_loss /= train_len
            avg_train_metric /= train_len

            avg_val_loss = 0
            avg_val_metric = 0
            val_len = 0
            standard_len = None

            model.eval()
            with torch.inference_mode():
                # val_pbar = tqdm(enumerate(val_loader))
                for ind, (X_val_bt, y_val_bt) in enumerate(val_loader):
                    # val_pbar.set_description(f"Val: {ind + 1}/{len(val_loader)}")
                    val_preds = model(X_val_bt.to(device), device= device)
                    val_loss = criterion(val_preds, y_val_bt.to(device))
                    val_metric = metric(val_preds, y_val_bt.to(device))

                    # ingroup
                    avg_val_loss += val_loss.item()
                    avg_val_metric += val_metric

                    if standard_len is None:
                        standard_len = X_val_bt.shape[0]

                    val_len += (X_val_bt.shape[0] / standard_len)

            # over group
            avg_val_loss /= val_len
            avg_val_metric /= val_len

            ans = (avg_val_metric > best_eval_metric) if metric_to_max else (avg_val_metric < best_eval_metric)
            if ans:
                best_eval_metric = avg_val_metric
                best_iter = epoch
                best_model = model.state_dict()
                torch.save(best_model, "../best_model.pt")

            train_log.append(avg_train_loss)
            train_metric_log.append(avg_train_metric)
            val_log.append(avg_val_loss)
            val_metric_log.append(avg_val_metric)
            best_eval_metric_log.append(best_eval_metric)

            if verbose and epoch % lag == 0:
                print(f"Epoch: {epoch + 1}/{epochs}. Train loss: {train_log[-1]:.6f} Train metric: {train_metric_log[-1]:.2f}% Val loss: {val_log[-1]:.6f} Val metric: {val_metric_log[-1]:.2f}% Best val metric: {best_eval_metric_log[-1]:.2f}% (iter: {best_iter + 1})")
            elif not verbose:
                epochs_bar.set_description(f"Validation metrics: {val_metric_log[-1]:.2f}%. Best: {best_eval_metric_log[-1]:.2f}% (iter: {best_iter + 1})")

        res = {
            "best model": best_model,
            "train log": train_log,
            "val log": val_log,
            "train metric log": train_metric_log,
            "val metric log": val_metric_log,
            "best val metric log": best_eval_metric_log,
            "best iteration": best_iter,
            "metric name": metric.__str__()
        }

        return res

    @staticmethod
    def eval(model, test_loader, metric, criterion, device= "cpu"):
        # X_test = X_test.to(device)
        # y_test = y_test.to(device)

        model = model.to(device)

        answer = []
        avg_test_loss = 0
        avg_test_metric = 0
        test_len = 0
        standard_len = None

        model.eval()
        with torch.inference_mode():
            # test_pbar = tqdm(enumerate(test_loader))
            for ind, (X_test_bt, y_test_bt) in enumerate(test_loader):
                # test_pbar.set_description(f"Test: {ind + 1}/{len(test_loader)}")
                test_preds = model(X_test_bt.to(device), device= device)
                test_loss = criterion(test_preds, y_test_bt.to(device))
                test_metric = metric(test_preds, y_test_bt.to(device))

                answer += list(test_preds.cpu().detach().numpy())

                avg_test_loss += test_loss.item()
                avg_test_metric += test_metric

                if standard_len is None:
                    standard_len = X_test_bt.shape[0]

                test_len += (X_test_bt.shape[0] / standard_len)

        # over group
        avg_test_loss /= test_len
        avg_test_metric /= test_len


        res = {
            "test preds": np.array(answer),
            "test loss": avg_test_loss,
            "test metric": avg_test_metric,
            "metric name": metric.__str__()
        }

        return res

    @staticmethod
    def plot_eval_res(eval_res, y_test, scaler, dates, start= 0, end= None):
        test_preds = np.array(eval_res["test preds"]).reshape(-1)[start:end]
        y_ts = np.array(y_test).reshape(-1)[start:end]
        date = dates.reshape(-1)[start:end]

        test_loss = eval_res["test loss"]
        test_metric = eval_res["test metric"]
        metric_name = eval_res["metric name"]

        test_preds = scaler.inverse_transform(test_preds)
        y_ts = scaler.inverse_transform(y_ts)

        plt.figure(figsize= (20, 5))
        plt.title("Test Predictions")
        plt.plot(date, test_preds, label= "Test predictions")
        plt.grid(True)
        plt.legend(loc= "lower right")
        plt.show()

        plt.figure(figsize= (20, 5))
        plt.title("Test predictions/actual")
        plt.plot(date, test_preds, label= "Test predictions")
        plt.plot(date, y_ts, label= "Actual values")
        plt.grid(True)
        plt.legend(loc= "lower right")
        plt.text(date[0], max(y_ts) * 0.95, f"Test loss: {test_loss:.5f}", fontsize= 16)
        plt.text(date[0], 0.9 * max(y_ts), f"Test {metric_name}: {test_metric:.2f}%", fontsize= 16)
        plt.show()

    @staticmethod
    def plot_train_res(train_res, start= 0, end= None):
        train_log = train_res["train log"][start:end]
        val_log = train_res["val log"][start:end]
        train_metric_log = train_res["train metric log"][start:end]
        val_metric_log = train_res["val metric log"][start:end]
        best_eval_metric_log = train_res["best val metric log"][start:end]
        best_iter = train_res["best iteration"]

        fig, axis = plt.subplots(ncols= 2, nrows= 1, figsize= (20, 5))
        axis[0].set_title("Train/test loss")
        axis[0].plot(train_log, label= "Train loss")
        axis[0].plot(val_log, label= "Val loss")
        axis[0].axvline(x= best_iter, color= "purple", label= f"Best iter: {best_iter}")
        axis[0].grid(True)
        axis[0].legend(loc= "best")

        axis[1].set_title("Train/test metric (%)")
        axis[1].plot(train_metric_log, label= "Train metric")
        axis[1].plot(val_metric_log, label= "Val metric")
        axis[1].axvline(x= best_iter, color= "purple", label= f"Best iter: {best_iter}")
        axis[1].grid(True)
        axis[1].legend(loc= "best")
        plt.show()

        plt.figure(figsize= (20, 5))
        plt.title("Best validation metric (%)")
        plt.plot(best_eval_metric_log, label= "Best val metric")
        plt.axvline(x= best_iter, color= "purple", label= f"Best iter: {best_iter}")
        plt.legend(loc= "best")
        plt.grid(True)
        plt.show()