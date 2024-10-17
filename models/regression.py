from typing import Tuple, Dict

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


def fit_regression_models(
    X: pd.DataFrame, y: pd.Series, num_folds: int = 5, unbiased_variance: bool = True
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    X, y = shuffle(X, y, random_state=42)
    kf = KFold(n_splits=num_folds)

    log_likelihoods = []
    coefficients_list = []
    squared_errors_list = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

        model = sm.OLS(y_train, X_train).fit()
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        squared_errors = residuals**2
        squared_errors_list.append(squared_errors)
        if unbiased_variance:  # use ols variance estim
            variance = model.scale
        else:
            variance = np.var(residuals, ddof=0)

        log_likelihood = -0.5 * np.log(2 * np.pi * variance) - (residuals**2) / (
            2 * variance
        )
        log_likelihoods.append(log_likelihood)
        coefficients_list.append(model.params)

    avg_coefficients = np.mean(np.vstack(coefficients_list), axis=0)
    variables = X.columns.insert(0, "const").tolist()
    var_coefficients = dict(zip(variables, avg_coefficients))
    all_log_likelihoods = np.concatenate(log_likelihoods)
    all_squared_errors = np.concatenate(squared_errors_list)

    return var_coefficients, all_log_likelihoods, all_squared_errors
