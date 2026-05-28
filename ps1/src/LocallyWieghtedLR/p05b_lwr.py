import matplotlib.pyplot as plt
import numpy as np
import util



def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)    
    # *** START CODE HERE ***
    # Fit a LWR model
    LWR = LocallyWeightedLinearRegression(tau)
    LWR.fit(x_train, y_train)
    # Get MSE value on the validation set
    predictions = LWR.predict(x_eval)
    print(f'MSE: {np.mean((predictions-y_eval)**2)}')
    # Plot validation predictions on top of training set
    def plot(x, y_label, y_pred, title):
        plt.figure()
        plt.plot(x[:,-1], y_label, 'bx', label='label')
        plt.plot(x[:,-1], y_pred, 'ro', label='prediction')
        plt.suptitle(title, fontsize=12)
        plt.legend(loc='upper left')
    plot(x_train, y_train, LWR.predict(x_train), 'Training set predictions')
    plot(x_eval, y_eval, predictions, 'Validation set predictions')
    plt.show()
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression():
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        n, d = x.shape
        # Reshape the input x by adding an additional dimension so that it can broadcast
        w_vector = np.exp(- np.linalg.norm(self.x - np.reshape(x, (n, -1, d)), ord=2, axis=2)**2 / (2 * self.tau**2))
        
        # Turn the weights into diagonal matrices, each corresponds to a single input. Shape (n, m, m)
        w = np.apply_along_axis(np.diag, axis=1, arr=w_vector)
         # Compute theta for each input x^(i). Shape (l, n)
        theta = np.linalg.inv(self.x.T @ w @ self.x) @ self.x.T @ w @ self.y
        return np.einsum('ij,ij->i', x, theta)
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau=5e-1,
         train_path='ds5_train.csv',
         eval_path='ds5_valid.csv')