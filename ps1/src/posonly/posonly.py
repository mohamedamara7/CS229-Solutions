import numpy as np
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!
from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***
    plot_path = save_path.replace('.txt', '.png')
    plot_path_true = plot_path.replace(WILDCARD, 'true')
    plot_path_naive = plot_path.replace(WILDCARD, 'naive')
    plot_path_adjusted = plot_path.replace(WILDCARD, 'adjusted')
    # Part (a): Train and test on true labels
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    x_train, y_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    eval_predictions=clf.predict(x_eval)
    print('Eval Accuracy: %.2f' % np.mean((eval_predictions > 0.5) == (y_eval == 1)))
    
    test_predictions=clf.predict(x_test)
    print('Test Accuracy: %.2f' % np.mean((test_predictions > 0.5) == (y_test == 1)))
    util.plot(x_test, y_test, clf.theta, plot_path_true)
    np.savetxt(output_path_true, test_predictions)

    # Part (b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    eval_predictions=clf.predict(x_eval)
    print('Eval Accuracy: %.2f' % np.mean((eval_predictions > 0.5) == (y_eval == 1)))
    
    test_predictions=clf.predict(x_test)
    print('Test Accuracy: %.2f' % np.mean((test_predictions > 0.5) == (y_test == 1)))
    util.plot(x_test, y_test, clf.theta, plot_path_naive)
    np.savetxt(output_path_naive, test_predictions)

    # Part (f): Apply correction factor using validation set and test on true labels
    x_eval, y_eval = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    x_eval = x_eval[y_eval == 1]
    predictions= clf.predict(x_eval)
    alpha = np.mean(predictions)
    print(f'Alpha: {alpha:.2f}')
    test_predictions = clf.predict(x_test) / alpha
    print(f'New Theta: {clf.theta / alpha}')
    print(f'Test Accuracy: {np.mean((test_predictions > 0.5) == (y_test == 1)):.2f}')

    # Plot and use np.savetxt to save outputs to output_path_adjusted
    util.plot(x_test, y_test, clf.theta, plot_path_adjusted, correction=alpha)
    np.savetxt(output_path_adjusted, test_predictions)

    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
