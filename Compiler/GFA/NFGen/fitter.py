"""The pipeline for using polynomial to fit **any** non-linear functions.
"""
import numpy as np
import copy

from sympy import evaluate

from NFGen.profiler import *
from NFGen.funcs import calculate_decimal, calculate_zero, convert_function, evaluate_polynomial, solve_lse, find_residual_func, adaptive_sampling, FLPsimFXP, overflowR, underflowR


import warnings

DEBUG = False


def _check_input(x, y=None):
    """Check the input x and y before expand its dimensions.
    """
    if len(x.shape) < 2:
        x = x[:, np.newaxis]
    if y is None:
        return x
    if len(y.shape) < 2:
        y = y[:, np.newaxis]

    return x, y



def constrain_k(a, b, k_target, n, f):
    """Constraint the maximum order k in interval [a, b] with given number system <n, f>.
    """
    overflow_x = np.max(np.abs([a, b]))
    k_limit_overflow = ((n - f) /
                       (np.log2(overflow_x))) if (overflow_x > 1) else k_target
    if(a*b > 0):
        underflow_x = np.min(np.abs([a, b]))
        k_limit_underflow = (f / (-np.log2(underflow_x))) if (
            underflow_x < 1 and underflow_x > 0) else k_target
    else:
        k_limit_underflow = 3
    k_limit_overflow = max(k_limit_overflow, 1)
    k_limit_underflow = max(k_limit_underflow, 3)
    
    feasible_k = int(np.min([k_limit_overflow, k_limit_underflow, k_target]))
    return feasible_k, k_target - feasible_k 


def define_cheby_coefficients(a, b, k, func):
    """Compute the chebyshev polynomial coefficients.
    """
    map_ab = lambda t: 0.5*(b-a)*t + 0.5*(a+b)
    cheb_points = np.cos(np.arange(k + 1)*np.pi/(k))
    x = map_ab(cheb_points)
    y = func(x)
    polyF = PolynomialFeatures(k)
    X = polyF.fit_transform(x[:, np.newaxis]).squeeze()
    try:
        coeffs = np.linalg.solve(X, y)
    except:
        warnings.warn("slover error, try scipy ", RuntimeWarning)
        coeffs = solve_lse(X, y, 'scipy').squeeze()
    
    return coeffs[:, np.newaxis]


class PolyK():
    """self-defined k-order polynomial fitting methods.
    """

    def __init__(self, k):
        self.k = k
        self.coeff = None
        self.decimal = None
        return

    def fit(self, a, b, func, f, n, ms=1000, derivative_flag=True, method="OLS", zero_mask=None):
        """Fit function with each coefficient statisfying the data representation requirement.
        This fit is based on samples.
        """
        self.n = n
        self.f = f
        decimal = calculate_decimal(f)
        self.decimal = decimal

        def _scaleC(char_x, c, i, n, f):
            """Construct the scaler s for given c on computing c * x^i in <n,f> number system;
            """
            s = np.min([2**(n-f-2)/(np.abs(c) * char_x**i) if (np.abs(c) * char_x**i) != 0 else 2**(n-f-2), 2**(f//2)])
            s = np.max([s, 1])
            s = FLPsimFXP(1/s, n, f)
            scaled_c = FLPsimFXP(c/s, n, f)
            
            return scaled_c, s  

        def _get_residual_cheb(a, b, func, k, coeff, scale_list, k_target):
            """Compute the residual coefficients using chebyshev method.
            """
            k_less = k_target - k
            p_feasible = coeff * scale_list
            residual_func = lambda x : func(x) - evaluate_polynomial(x, p_feasible)
            mix_coeff = define_cheby_coefficients(a, b, k, residual_func).squeeze()
            
            return np.append(mix_coeff, [0] * k_less)[:, np.newaxis]  

        def _get_residual(x,
                          func,
                          k, 
                          coeff,
                          scale_list,
                          k_target,
                          derivative_flag=True):
            """Fit the new polynomial for residual functions.
            """
            k_less = k_target - k
            p_feasible = coeff * scale_list

            if derivative_flag:
                # Residual function for this item.
                residual_func = find_residual_func(func, p_feasible)
            else:
                raise NotImplementedError(
                    "Residual boosting not support for underivative functions")

            y = residual_func(x)
            X = copy.deepcopy(x)
            X, y = _check_input(X, y)
            polyFeatures = PolynomialFeatures(k)
            X = polyFeatures.fit_transform(X)

            try:
                mix_coeff = solve_lse(X, y, "solver")
            except:
                warnings.warn("Solver error, try scipy ", RuntimeWarning)
                mix_coeff = solve_lse(X, y, 'scipy')
            
            mix_coeff = np.append(mix_coeff, [0] * k_less)[:, np.newaxis]
            return mix_coeff


        def residual_boosting(target_k, coeff, scale_list, func, x, y, metric="mean"):
            """Residual boosting.
            """
            X = copy.deepcopy(x)
            X, y = _check_input(X, y)
            polyFeatures = PolynomialFeatures(target_k)
            X = polyFeatures.fit_transform(X)
            
            for k in range(target_k - 1, 0, -1):
                # mix_coeff FXP emulation
                if metric == "mean":
                    mix_coeff = _get_residual(x, func, k, coeff, scale_list,
                                            target_k)
                elif metric == "max":
                    mix_coeff = _get_residual_cheb(x[0], x[-1], func, k, coeff, scale_list, target_k)
                # ScalePoly
                mix_coeff = overflowR(mix_coeff, n, f)
                mix_coeff = (coeff * scale_list) + mix_coeff
                new_scale_list = np.ones(scale_list.shape)
                character_x = np.max(np.abs(x))
                for i in range(target_k+1):
                    sc, s = _scaleC(character_x, mix_coeff[i][0], i, n, f)
                    mix_coeff[i][0] = sc
                    new_scale_list[i][0] = s
                    
                # check benefit, if exist -> boost!
                if metric == "mean":
                    mse_previous = np.mean(
                        (np.dot(X * scale_list.T, coeff) - y)**2)
                    mse_tests = np.mean(
                        (np.dot(X * new_scale_list.T, mix_coeff) - y)**2)
                    if (mse_tests < mse_previous):
                        coeff = mix_coeff
                        scale_list = new_scale_list
                elif metric == "max":
                    max_previous = np.max(
                        (np.dot(X * scale_list.T, coeff) - y)**2)
                    max_tests = np.max(
                        (np.dot(X * new_scale_list.T, mix_coeff) - y)**2)
                    if (max_tests < max_previous):
                        coeff = mix_coeff
                        scale_list = new_scale_list
            return coeff, scale_list

        if derivative_flag:
            func_exec = convert_function(func)
        else:
            func_exec = func
        X, y = adaptive_sampling(a, b, func_exec, f, ms, zero_mask=zero_mask)
        sample_num = len(X)
        x = copy.deepcopy(X)
        target_k, k_less = constrain_k(a, b, self.k, n, f)

        if method == "OLS":
            X, y = _check_input(X, y)
            # Solve the optimal coefficients in FLP environment.
            if sample_num > target_k + 1:  # OLS method
                polyFeatures = PolynomialFeatures(target_k)
                X = polyFeatures.fit_transform(X)
                try:
                    coeff = solve_lse(X, y, "solver")
                except:
                    warnings.warn("slover error, try scipy ", RuntimeWarning)
                    coeff = solve_lse(X, y, 'scipy')
            else:  # LI method
                target_k = sample_num - 1  # update target_k.
                k_less = self.k - target_k
                polyFeatures = PolynomialFeatures(target_k)
                X = polyFeatures.fit_transform(X).squeeze()
                try:
                    coeff = np.linalg.solve(X,
                                            y)  # more precious solving function.
                except:
                    warnings.warn(
                        "can not directly solve the inverse, place it with pinv. ",
                        RuntimeWarning)
                    coeff = np.dot(np.linalg.pinv(X), y.squeeze())
                if (len(coeff.shape) < 2):
                    coeff = coeff[:, np.newaxis]

        elif method == "Cheby":
            if sample_num > target_k + 1:
                coeff = define_cheby_coefficients(a, b, target_k, func_exec)
            else:
                X, y = _check_input(X, y)
                target_k = sample_num - 1  # update target_k.
                k_less = self.k - target_k
                polyFeatures = PolynomialFeatures(target_k)
                X = polyFeatures.fit_transform(X).squeeze()
                try:
                    coeff = np.linalg.solve(X,
                                            y)  # more precious solving function.
                except:
                    warnings.warn(
                        "can not directly solve the inverse, place it with pinv. ",
                        RuntimeWarning)
                    coeff = np.dot(np.linalg.pinv(X), y.squeeze())
                if (len(coeff.shape) < 2):
                    coeff = coeff[:, np.newaxis]

        # Set scaling factor
        coeff = overflowR(coeff, n, f)
        character_x = np.max(np.abs(x))
        scale_list = np.ones((len(coeff), 1))
        # Scale poly
        for i in range(target_k+1):
            sc, s = _scaleC(character_x, coeff[i][0], i, n, f)
            coeff[i][0] = sc
            scale_list[i][0] = s

        if DEBUG:
            print(">>> scale list: ", scale_list)
            print(">>> scale_list shape: ", scale_list.shape)
            print("coeff shape: ", coeff.shape)
            print(">>>> X shape: ", X.shape)

        # Residual Boosting
        if derivative_flag:
            if method == "OLS":
                coeff, scale_list = residual_boosting(target_k, coeff, scale_list, func, x, y, "mean")
            elif method == "Cheby":
                coeff, scale_list = residual_boosting(target_k, coeff, scale_list, func_exec, x, y, "max")

        # Save constructed coefficients.
        self.coeff = np.append(coeff, [0] * k_less)[:, np.newaxis]
        self.scale = np.append(scale_list, [1] * k_less)[:, np.newaxis]


    def predict(self, X, n, f):
        """Predict the target values.
        """
        X = _check_input(X)
        polyFeatures = PolynomialFeatures(self.k)
        X = polyFeatures.fit_transform(X)
        res = FLPsimFXP(X * self.coeff.T, n, f)
        res = FLPsimFXP(res * self.scale.T, n, f)
        res = np.sum(res, axis=1)

        return FLPsimFXP(res, n, f)


def reconstruct_prediction(x, coeffA, breaks):
    """The reconstructed prediction method.
    """
    x_index = x >= breaks
    x_index = np.sum(x_index) - 1  # in cipher-text -> get_last_one.

    coeff = coeffA[x_index]  # select coeff
    tmp = np.zeros(len(x))
    for c in coeff[::-1]:  # sequential k muls
        tmp = tmp * x + c

    return tmp
