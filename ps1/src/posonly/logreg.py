import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    # *** START CODE HERE ***
    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    predictions=clf.predict(x_eval)
    print('LR Accuracy: %.2f' % np.mean((predictions > 0.5) == (y_eval == 1)))
    # Plot decision boundary on top of validation set set
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_eval, y_eval, clf.theta, plot_path)
    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path, predictions)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, d = x.shape
        if self.theta is None:
            self.theta = np.zeros(d)
        for i in range(self.max_iter):
            probs=self.predict(x)
            grad=(x.T@(probs-y))/n

            # Compute the Hessian matrix
            hessian = (x.T @ np.diag(probs * (1 - probs)) @ x) / n

            prev_theta = np.copy(self.theta)
            self.theta -= self.step_size * np.linalg.inv(hessian) @ grad

            # Compute the loss (negative log likelihood)
            epsilon = 1e-12  # To prevent log(0)
            probs = np.clip(probs, epsilon, 1 - epsilon)  # Clip predictions to avoid numerical issues
            loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
            
            if self.verbose:
                print(f'Iter: {i}, Loss: {loss}')
            if np.linalg.norm(self.theta - prev_theta) < self.eps:
                break
        if self.verbose:
            print(f'Final Theta (LogReg): {self.theta}')
        # *** END CODE HERE ***
    
    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return self._sigmoid(x @ self.theta)
        # *** END CODE HERE ***
    
    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
        
if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
