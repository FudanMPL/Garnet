from logging import raiseExceptions
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, fminbound
import sympy as sp
from sklearn.preprocessing import PolynomialFeatures

DEBUG = False
DEBUG_folder = './Record/'

def overflowR(x, n, f):
    """Round off the overflow bits.
    """
    if isinstance(x, np.ndarray):
        shape = x.shape
        x = x.flatten()
        sign = np.sign(x)
        overflow_flag = np.abs(x) > 2**(n - f - 1)
        x[overflow_flag] = (sign * 2**(n - f - 1))[overflow_flag]
        x = np.reshape(x, shape)
    elif isinstance(x, float):
        sign = np.sign(x)
        x = sign * np.min([np.abs(x), 2**(n - f - 1)])
    else:
        raise NotImplementedError("Not support type " + str(type(x)))

    return x


def underflowR(x, f):
    """Round off the underflow bits
    """
    if isinstance(x, np.ndarray):
        shape = x.shape
        x = x.flatten()
        sign = np.sign(x)
        underflow_flag = np.abs(x) < 2**(-f)
        x[underflow_flag] = (sign * 2**(-f))[underflow_flag]
        x = np.reshape(x, shape)
    elif isinstance(x, float):
        sign = np.sign(x)
        x = sign * np.max([np.abs(x), 2**(-f)])
    else:
        raise NotImplementedError("Not support type " + str(type(x)))

    return x


def FLPsimFXP(x, n, f):
    """FXP emulator.
    """
    decimal = calculate_decimal(f)
    x = overflowR(x, n, f)
    x = underflowR(x, f)
    x = np.round(x, decimals=decimal)

    return x

def calculate_zero(f):
    """Calculate **zero mask** using f.
    """
    decimal = int(np.log10(1 / 2**-f))
    if decimal >= 11:
        decimal = 11
    zero_mask = 10**(-decimal)

    return zero_mask


def calculate_decimal(f):
    """Calculate the decimal using f.
    """
    decimal = int(np.log10(1/2**-f))
    if decimal >= 11:
        decimal = 11
    return decimal


def solve_lse(A, b, strategy='inverse'):
    """Solve the unconstrained mse, using inverse.
    """

    def _func(x):
        return np.dot(A, x) - b.squeeze()

    x = None
    if strategy == 'inverse':
        x = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))

    elif strategy == 'solver':
        x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))

    elif strategy == 'scipy':
        x_init = np.random.normal(size=(A.shape[1]))
        res = least_squares(_func, x_init)
        if (res.success):
            x = least_squares(_func, x_init).x
            if (len(x.shape) < 2):
                x = x[:, np.newaxis]
        else:
            raise NameError("Scipy optimization failed.")
    return x


def analytic_error(func,
                   interval_coeff,
                   a,
                   b,
                   f,
                   zero_mask=None,
                   method="absolute_error"):
    """Error analysis in analytic way, the analytic error we only support the absolute error now.

    Here we calculate the maximum value for |func - ploy_func| in the interval [a, b].
    Care we only support absolute_error in the analytic way.
    """
    decimal = calculate_decimal(f)
    if zero_mask is None:
        zero_mask = calculate_zero(f)

    residual_func = analytic_residual_func(func, interval_coeff, method=method)

    x = sp.Symbol('x')
    residual_func = sp.lambdify(x, -(residual_func**2), 'numpy')
    x_min, max_error, _, _ = fminbound(residual_func, a, b, full_output=1)
    x_min = np.round(x_min, decimal)
    max_error = np.sqrt(-residual_func(x_min))

    return max_error if max_error >= zero_mask else 0


def sampled_error(y_true, y_pred, f, zero_mask=None, method="relative_error"):
    """error analysis using samples.

    We offerd three options, absolute_error, relative_error and significant figures.
    """
    if isinstance(y_true, np.float64):
        y_true = np.array([y_true])
    if isinstance(y_pred, np.float64):
        y_pred = np.array([y_pred])
    if(len(y_true.shape) > 1):
        y_true = y_true.squeeze()
        
    # Set the distinguish limit for cipher-text
    decimal = calculate_decimal(f)
    if zero_mask is None:
        zero_mask = calculate_zero(f)
    # Mask all the value <= zero_mask to zero.
    y_pred = np.round(y_pred, decimal)
    y_true = np.round(y_true, decimal)
    # y_true[np.abs(y_true) <= zero_mask] = 0
    # y_pred[np.abs(y_pred) <= zero_mask] = 0

    if method == "relative_error":
        abs_error = np.sqrt((y_pred - y_true)**2)
        relative_error = np.array([
            abs_error[i] /
            np.abs(y_true[i]) if y_true[i] >= zero_mask else abs_error[i]
            for i in range(len(abs_error))
        ])
        ignore_index = np.array(abs_error <= 10**(-decimal))
        relative_error[ignore_index] = 0
        return relative_error

    elif method == "absolute_error":
        return np.sqrt((y_pred - y_true)**2)

    else:
        raise Exception("Invilid method - ", method)


def calculate_kx(x, k):
    """calculate [1, x, x^2, ..., x^k], totally log(k) times vector mul.
    """
    target = [i for i in range(k)]
    in_set = {0, 1}
    items = np.transpose(np.tile(x, (k, 1)))
    items[:, 0] = 1
    now_index = [1] * (k - 1)
    now_index.insert(0, 0)

    while (max(in_set) < (k - 1)):
        index_less = [target[i] - now_index[i] for i in range(k)]
        index_less = [
            item if (item in in_set) else max(in_set) for item in index_less
        ]

        items = items * items[:, index_less]
        now_index = [now_index[i] + index_less[i] for i in range(k)]
        in_set.update(now_index)

    return items


def piece_prediction(x, coeffA, breaks, scale_list, f, n):
    """The reconstructed prediction method.
    """
    x_index = x >= breaks[:-1]
    x_index = np.sum(x_index, axis=1) - 1  # in cipher-text -> get_last_one.
    coeff = coeffA[x_index]  # select coeff.
    scale = scale_list[x_index]  # select the scale factor
    x = np.squeeze(x)
    x_items = FLPsimFXP(calculate_kx(x, len(coeff[0])) * coeff, n, f)
    tmp = FLPsimFXP(x_items * scale, n, f)
    tmp = np.sum(tmp, axis=1)

    return FLPsimFXP(tmp, n, f)


def plain_piece_prediction(x, coeffA, breaks):

    x_index = x >= breaks[:-1]
    x_index = np.sum(x_index, axis=1) - 1  # in cipher-text -> get_last_one.
    coeff = coeffA[x_index]  # select coeff.

    x = np.squeeze(x)
    tmp = np.zeros(len(x))
    for c in coeff.T[::-1]:  # sequential k muls
        tmp = tmp * x + c

    return tmp


def result_orgnize(breaks, coeffA, scaler_list):
    """Orgnize the result as the target form.
    """
    breaksf = [breaks[0][0]]
    breaksf.extend([breaks[i][1] for i in range(len(breaks))])
    coeffA = np.stack(coeffA).squeeze()
    if scaler_list != None:
        scaler_list = np.stack(scaler_list).squeeze()
        return breaksf, coeffA, scaler_list
    else:
        return breaksf, coeffA


def find_derivative(func, order=1):
    """Generate the derivative function
    """
    x = sp.Symbol('x')
    y = func(x)
    yp = sp.diff(y, x, order)
    return sp.lambdify(x, yp, 'numpy')


def convert_function(func):
    """Convert the function to callable func.
    """
    x = sp.Symbol('x')
    y = func(x)

    return sp.lambdify(x, y, 'numpy')


def coeff_reorgnize(coeff, c):
    """Refine the coeff of a0 + a1(x-c) + a2(x-c)^2 + ... + an(x-c)^n;
    to b0 + b1x + b2x^2 + ... + bnx^n;
    """
    res = [[1]]

    for i in range(len(coeff)):
        tmp = res[-1]
        tmp_reverse = [-k for k in tmp]
        tmp_append = tmp.copy()
        tmp_reverse.append(0)
        tmp_append.insert(0, 0)
        tmp = [tmp_reverse[k] + tmp_append[k] for k in range(len(tmp_reverse))]
        res.append(tmp)

    tmp_coeff = [0] * (len(coeff) + 1)
    res = res[1:]
    for i in range(len(res)):
        m_max = len(res[i]) - 1
        for k in range(len(res[i])):
            tmp_coeff[k] += coeff[i] * res[i][k] * (c**(m_max - k))

    tmp_coeff[0] += coeff[0]
    return tmp_coeff


def adaptive_sampling(start, end, func, f, ms, zero_mask=None):
    """Sampling samples from given func from [start, end].
    
    The **distinguishable** samples's condition includes the plain-text distinguishable and cipher-text distinguishable. Care that the precision of cipher-text is always 2**(-f) absolute points while plain-text follows the defination of effective number.
    """
    # input indistinguishable
    step = 2**(-f)
    input_indis = int((end - start) / step)
    decimal = calculate_decimal(f)
    if zero_mask is None:
        zero_mask = calculate_zero(f)

    # Sampling method
    ms = min(ms, input_indis)
    x = np.unique(np.linspace(start, end, ms).round(decimal))
    y = func(x).round(decimal)
    y[np.abs(y) <= zero_mask] = 0.0  # Mask value less then zero to zero.

    return x, y


def find_residual_func(func, feasible_coeff, analytic=False):
    """Generate the residual function for target functions.
    """
    feasible_coeff = feasible_coeff.squeeze()
    x = sp.Symbol('x')
    func_rterm = feasible_coeff[0]
    func = func(x)

    for i in range(1, len(feasible_coeff)):
        func_rterm += feasible_coeff[i] * (x**i)

    residual_func = func - func_rterm

    if analytic:
        return residual_func
    else:
        return sp.lambdify(x, residual_func, 'numpy')


def evaluate_polynomial(x, feasible_coeff):
    """Evaluate the polynomial for x with given coefficients.
    """
    k = feasible_coeff.shape[0]
    polyFeature = PolynomialFeatures(k-1)
    if(len(x.shape) < 2):
        x = x[:, np.newaxis]
    X = polyFeature.fit_transform(x)
    
    return np.dot(X, feasible_coeff).squeeze()
    

def analytic_residual_func(func, feasible_coeff, method="relative_error"):
    """Generate the expression of the residual function among func and polynomial approximation.
    """
    feasible_coeff = feasible_coeff.squeeze()
    x = sp.Symbol('x')
    func_rterm = feasible_coeff[0]
    func = func(x)

    for i in range(1, len(feasible_coeff)):
        func_rterm += feasible_coeff[i] * (x**i)
    residual_func = func - func_rterm

    if method == "relative_error":
        residual_func = residual_func / (func + 1e-8)

    return residual_func