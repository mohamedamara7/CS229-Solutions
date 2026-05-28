import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    mse_values = []
    for tau in tau_values:
        # Fit a LWR model
        lwr = LocallyWeightedLinearRegression(tau)
        lwr.fit(x_train, y_train)

        # Load validation set
        x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

        # Get MSE value on the validation set
        predictions = lwr.predict(x_valid)
        mse = np.mean((predictions - y_valid) ** 2)
        mse_values.append(mse)
        print(f'Tau: {tau}, MSE: {mse}')
    # Fit a LWR model with the best tau value
    best_tau = tau_values[np.argmin(mse_values)]
    lwr = LocallyWeightedLinearRegression(best_tau)
    lwr.fit(x_train, y_train)
    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    test_predictions = lwr.predict(x_test)
    test_mse = np.mean((test_predictions - y_test) ** 2)
    print(f'Best Tau: {best_tau}, Test MSE: {test_mse}')
    # Save predictions to pred_path
    np.savetxt(pred_path, test_predictions, delimiter=',')
    # Plot data
    def plot(x, y_label, y_pred, title):
        plt.figure()
        plt.plot(x[:, -1], y_label, 'bx', label='label')
        plt.plot(x[:, -1], y_pred, 'ro', label='prediction')
        plt.suptitle(title, fontsize=12)
        plt.legend(loc='upper left')
    plot(x_train, y_train, lwr.predict(x_train), 'Training set predictions')
    plot(x_valid, y_valid, predictions, 'Validation set predictions')
    plot(x_test, y_test, test_predictions, 'Test set predictions')
    plt.show()
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='ds5_train.csv',
         valid_path='ds5_valid.csv',
         test_path='ds5_test.csv',
         pred_path='p05c_pred.txt')