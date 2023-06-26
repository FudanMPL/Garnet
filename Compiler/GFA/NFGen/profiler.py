"""The time prediction model.
PolyProfiler - contains polynimial models for each building blocks.
"""
import numpy as np
import copy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error


class SubPoly():
    """Polynomial model, for normal 1-d building blocks.
    """
    def __init__(self, degree=2, fit_intercept=True):
        self.degree = degree
        self.fit_intercept = fit_intercept
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=self.degree)),
            ('linear', LinearRegression(fit_intercept=self.fit_intercept))
        ])

    def fit(self, X, y):
        X = np.array(X)
        if(len(X.shape) < 2):
            X = X[:, np.newaxis]
        self.model.fit(X, y)

    def predict(self, x):
        if type(x) is np.ndarray:
            if(len(x.shape) < 2):
                x = x[:, np.newaxis]
        else:
            x = np.array(x)[:, np.newaxis]
        return self.model.predict(x)


    def print_coeff(self):
        """print the coefficients of the target models.
        """
        print(self.model.named_steps['linear'].coef_)
    
    
    def model_analysis(self, x, y):
        """Analyze the model's learning state.
        """
        pred = self.predict(x)
        relative_error = np.sqrt((pred - y)**2) / y
        
        mean_error = np.mean(relative_error)
        std_error = np.std(relative_error)
        max_error = np.max(relative_error)
        
        print("Error Analysis:\n>>>Mean: %.4g\n>>>Std: %.4g\n>>>Max: %.4g" %(mean_error, std_error, max_error))


class CrossPoly():
    """Construct the cross features for 2-d building blocks. 
    """
    def __init__(self, degree=1):
        self.degree = degree
        self.model = LinearRegression()
    
    def cross_feature_construct(self, X):
        """Construct the cross features x1*x2.
        """
        if X.shape[1] !=2 :
            print("For cross profiler, the input dimension must be 2.")
        x_cross = X[:, 0] * X[:, 1]
        X = np.hstack([X, x_cross[:, np.newaxis]])

        return X

    def fit(self, X, y):
        """
        """
        X = self.cross_feature_construct(X)
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        """
        X = self.cross_feature_construct(X)
        return self.model.predict(X)


class PolyProfiler():
    """Using different SubPoly for each building blocks.
    """
    def __init__(self, building_blocks, degree=2, fit_intercept=False):
        """Model dict construction.
        """
        self.keys_1d = building_blocks['1d']
        self.keys_2d = building_blocks['2d']

        self.model_dict_1d = {key: SubPoly(degree, fit_intercept) for key in building_blocks['1d']}

        self.model_dict_2d = ({key: CrossPoly() for key in building_blocks['2d']})


    def fit(self, X_dict, y_dict):
        """fit each SubPoly model for each building blocks.
        
        Data from pre-run profiling of each building blocks.
        X_dict: {'1d': data_list for 1-d buidling blocks, 
                '2d': data_list for 2d buidling blocks}

        y_dict: {'1d': {bb: data_list for all 1d bb}, 
                '2d': {bb: data_list for all 2d building blocks}}.
        """ 
        for key in self.keys_1d:
            self.model_dict_1d[key].fit(X_dict['1d'], y_dict['1d'][key])
        for key in self.keys_2d:
            self.model_dict_2d[key].fit(X_dict['2d'], y_dict['2d'][key])


    def predict_time_cost(self, d_dict, c_dict):
        """Using the fitted poly model to predict the time cost.
        
        d_dict: the data volumn for each building blocks.
        c_dict: the coefficient for each building blocks.
        """
        time_predict = 0
        for key in self.keys_1d:
            if (len(c_dict['1d'][key]) == 0):
                continue
            for i in range(len(d_dict['1d'][key])):
                time_predict += c_dict['1d'][key][i] * self.model_dict_1d[key].predict(
                    d_dict['1d'][key][i])

        for key in self.keys_2d:
            if (len(c_dict['2d'][key]) == 0):
                continue
            for i in range(len(d_dict['2d'][key])):
                time_predict += c_dict['2d'][key][i] * self.model_dict_2d[key].predict(
                    d_dict['2d'][key][i])

        return time_predict


def error_analysis(test_set, time_test, building_blocks, metric=mean_absolute_percentage_error):
    """Analyze each error.
    """
    error_dict = {key:[] for key in building_blocks}
    for key in building_blocks:
        error = metric(test_set['data'][key], time_test[key][0])
        error_dict[key] = error
        print("key: %s | Error: %.6f" %(key, error))
    return error_dict


def important_error_metric(error_dict):
    """Compute the error of main important building blocks, ignore the less important blocks.
    degree 10: get_last_one, normed_bits.
    degree 7: gt, eq, bit_decompose.
    degree 5: mul, trunc.
    degree 3: dot_product.
    """

    d10_list = ['get_last_one', 'normed_bits']
    d7_list = ['gt', 'eq', 'bit_decompose']
    d5_list = ['mul', 'trunc']
    d3_list = ['dot_product']

    final_error = 0
    for key in error_dict.keys():
        if key in d10_list:
            final_error += error_dict[key]*10
        if key in d7_list:
            final_error += error_dict[key]*7
        if key in d5_list:
            final_error += error_dict[key]*5
        if key in d3_list:
            final_error += error_dict[key]*3
    
    return final_error/10