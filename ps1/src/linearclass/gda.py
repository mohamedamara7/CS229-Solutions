import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a GDA classifier
    clf = GDA()
    clf.fit(x_train, y_train)
    predictions=clf.predict(x_eval)
    print('LR Accuracy: %.2f' % np.mean((predictions > 0.5) == (y_eval == 1)))
    # Plot decision boundary on validation set
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_eval, y_eval, clf.theta, plot_path)
    # Use np.savetxt to save outputs from validation set to save_path
    np.savetxt(save_path, predictions)
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, d  = x.shape
        # Find phi, mu_0, mu_1, and sigma
        phi = np.mean(y)
        one_examples = np.sum(y==1)
        zero_examples = n - one_examples
        mu_0 = np.sum(x[y==0], axis=0) / zero_examples
        mu_1 = np.sum(x[y==1], axis=0) / one_examples

        diff = np.copy(x)
        diff[y==0] -= mu_0
        diff[y==1] -= mu_1

        sigma = (diff.T @ diff) / n
        # Write theta in terms of the parameters
        self.theta = np.zeros(d+1) # +1 for intercept term
        self.theta[0] = (mu_0.T @ np.linalg.inv(sigma) @ mu_0 - mu_1.T @ np.linalg.inv(sigma) @ mu_1)/2 - np.log((1-phi)/phi)
        self.theta[1:] = -np.linalg.inv(sigma)@(mu_0-mu_1)
        if self.verbose:
            print('Final theta (GDA): {}'.format(self.theta))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1/(1+np.exp(-x @ self.theta))
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
