from Compiler.types import Array 
from Compiler.types import sint
from Compiler.group_ops import GroupSum
from Compiler.types import sfix  
from Compiler.joint_statistics import mean

def square(self):
    """Calculate the mean of the array.
    
    :returns: Mean value of the array elements.
    """
    if not isinstance(self[0], sfix):
        raise TypeError("This function can only be executed for arrays of type sfix.")
    # Ensure the array is not empty
    if self.length == 0:
        raise ValueError("Cannot compute mean of an empty array.")
    
    # Get the vector representation of the array
    vector = self.get_vector()
    
    # 计算每个元素的平方
    squared_vector = [x * x for x in vector]

    squared_array = Array(self.length, sfix)

    for i in range(self.length):
        squared_array[i] = squared_vector[i]
    
    return squared_array

def sqrt_exp_log(x):
    """ Approximate square root using exp and log.

    :param x: Input value (sfix)
    :returns: Approximate square root of x
    """
    if not isinstance(x, sfix):
        raise TypeError("sqrt_exp_log can only be executed for sfix values.")

    return (x.log() * 0.5).exp()

# def sqrt_approx(x, iterations=8):
#     """Approximate square root using Newton's method.

#     :param x: Input value (sfix)
#     :param iterations: Number of iterations for refinement
#     :returns: Approximate square root of x
#     """
#     if not isinstance(x, sfix):
#         raise TypeError("sqrt_approx can only be executed for sfix values.")

#     # 选择一个初始值（避免除零错误）
#     y = x.v / 2 if x.v > 1 else 1  # 选取较好的初始猜测值

#     for _ in range(iterations):
#         y = (y + x.v / y) / 2  # 牛顿迭代公式
    
#     return sfix._new(y, k=x.k, f=x.f)  # 返回计算结果作为 sfix 类型



def sqrt_approx(x, iterations=8):
    """Approximate square root using Newton's method with encrypted comparisons.

    :param x: Input value (sfix)
    :param iterations: Number of iterations for refinement
    :returns: Approximate square root of x
    """
    if not isinstance(x, sfix):
        raise TypeError("sqrt_approx can only be executed for sfix values.")

    # 选择一个初始值（避免除零错误）
    one = sfix._new(1, k=x.k, f=x.f)
    two = sfix._new(2, k=x.k, f=x.f)
    
    # 使用加密的布尔比较来替代 x.v > 1
    is_greater_than_one = (x > one)
    
    # 根据布尔值选择初始猜测值
    y = two * x if is_greater_than_one else one  # 如果 x > 1，y = x / 2，否则 y = 1
    
    # 牛顿迭代法
    for _ in range(iterations):
        y = (y + x / y) / two  # 牛顿迭代公式
    
    return sfix._new(y.v, k=x.k, f=x.f)  # 返回计算结果作为 sfix 类型


def variance(self):
    """Calculate the variance of the array.
    
    :returns: Variance of the array elements.
    """
    if not isinstance(self[0], sfix):
        raise TypeError("This function can only be executed for arrays of type sfix.")
    if self.length == 0:
        raise ValueError("Cannot compute variance of an empty array.")
    
    # 计算 X 的平方的均值
    mean_of_squares = self.square().mean()
    
    # 计算 X 的均值
    mean_value = self.mean()
    
    # 计算均值的平方
    square_of_mean = mean_value * mean_value
    
    # 计算方差
    variance_value = mean_of_squares - square_of_mean
    
    return variance_value

def std_dev(self):
    """Calculate the standard deviation of the array.
    
    :returns: Standard deviation of the array elements.
    """
    if not isinstance(self[0], sfix):
        raise TypeError("This function can only be executed for arrays of type sfix.")
    if self.length == 0:
        raise ValueError("Cannot compute standard deviation of an empty array.")
    
    # 计算方差
    variance_value = self.variance()
    
    # 计算标准差（平方根）
    std_dev_value = variance_value.sqrt(variance_value)
    
    return std_dev_value


def sqrt(cls, y, iter=5):
    """ Secret fixed-point square root using Newton-Raphson method """
    assert isinstance(y, sfix)
    
    # 设定初始值
    x = cls.exp_fx(y, iter=5)
    
    # 进行迭代
    for _ in range(iter):
        x = 0.5 * (x + cls.newton_div(y, x))
    
    return x


sfix.sqrt_exp_log = sqrt_exp_log
sfix.sqrt_approx = sqrt_approx
sfix.sqrt = sqrt
Array.square = square
# 绑定方法到 Array
Array.variance = variance
# 绑定方法到 Array
Array.std_dev = std_dev
