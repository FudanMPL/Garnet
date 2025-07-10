from Compiler.types import Array 
from Compiler.types import sint
from Compiler.group_ops import GroupSum
from Compiler.types import sfix  

    # start point
def mean(arr):
    """Calculate the mean of the array.
    
    :returns: Mean value of the array elements.
    """
    if not isinstance(arr[0], sfix):
        raise TypeError("This function can only be executed for arrays of type sfix.")
    # Ensure the array is not empty
    if arr.length == 0:
        raise ValueError("Cannot compute mean of an empty array.")
    
    # Get the vector representation of the array
    vector = arr.get_vector()
    
    # Sum all elements
    total_sum = sum(vector)
    # Compute the mean
    mean_value = total_sum / arr.length
    
    return mean_value

def median(arr):
    """Calculate the median of the array.
    
    :returns: Median value of the array elements.
    """
    if not isinstance(arr[0], sfix):
        raise TypeError("This function can only be executed for arrays of type sfix.")
    if arr.length == 0:
        raise ValueError("Cannot compute median of an empty array.")
    arr.sort()
    # vector = self.get_vector()
    # sorted_vector = vector.sort()
    sorted_vector = arr.get_vector()
    mid_index = arr.length // 2
    
    if arr.length % 2 == 0:
        median_value = (sorted_vector[mid_index - 1] + sorted_vector[mid_index]) / 2
    else:
        median_value = sorted_vector[mid_index]
    
    return median_value

def mode(arr):

    if not isinstance(arr[0], sfix):
        raise TypeError("This function can only be executed for arrays of type sfix.")
    if arr.length == 0:
        raise ValueError("Cannot compute median of an empty array.")
    arr.sort()
    sorted_data = arr.get_vector()
    n = arr.length
    # 对self排完序后 从头开始遍历 若当前值与前一个值相同，计数加一 当前值与前一个值不同 如果当前计数大于最大计数，更新最大计数并重置众数列表 如果当前计数等于最大计数，将当前值加入众数列表 
    # Step 2: Initialize group indicator array
    g = Array(n, sint)
    g.assign_all(1)  # 初始值为1，表示每个元素都是一个新组的开始
    
    # Step 3: Create group indicator based on sorted data
    for i in range(1, n):
        g[i] = sorted_data[i] != sorted_data[i - 1]  # 如果当前元素与前一个元素不同，则为新组
    
    # Step 4: Calculate group sums to get frequency of each element
    group_sums = GroupSum(g, Array(n, sint).assign_all(1))
    
    # Step 5: Find the maximum frequency and corresponding element
    max_count = 0
    mode = sorted_data[0]
    
    # n轮的比较 
    for i in range(n):
        current_count = g[i] * group_sums[i]  # 仅在 g[i] == 1 时更新，否则为 0
        is_new_max = current_count > max_count
        max_count = is_new_max * current_count + (1 - is_new_max) * max_count
        mode = is_new_max * sorted_data[i] + (1 - is_new_max) * mode
    
    return mode

def square(arr):
    """Calculate the mean of the array.
    
    :returns: Mean value of the array elements.
    """
    if not isinstance(arr[0], sfix):
        raise TypeError("This function can only be executed for arrays of type sfix.")
    # Ensure the array is not empty
    if arr.length == 0:
        raise ValueError("Cannot compute mean of an empty array.")
    
    # Get the vector representation of the array
    vector = arr.get_vector()
    
    # 计算每个元素的平方
    squared_vector = [x * x for x in vector]

    squared_array = Array(arr.length, sfix)

    for i in range(arr.length):
        squared_array[i] = squared_vector[i]
    
    return squared_array



def variance(arr):
    """Calculate the variance of the array.
    
    :returns: Variance of the array elements.
    """
    if not isinstance(arr[0], sfix):
        raise TypeError("This function can only be executed for arrays of type sfix.")
    if arr.length == 0:
        raise ValueError("Cannot compute variance of an empty array.")
    
    # 计算 X 的平方的均值
    mean_of_squares = arr.square().mean()
    
    # 计算 X 的均值
    mean_value = arr.mean()
    
    # 计算均值的平方
    square_of_mean = mean_value * mean_value
    
    # 计算方差
    variance_value = mean_of_squares - square_of_mean
    
    return variance_value

def std_dev(arr):
    """Calculate the standard deviation of the array.
    
    :returns: Standard deviation of the array elements.
    """
    if not isinstance(arr[0], sfix):
        raise TypeError("This function can only be executed for arrays of type sfix.")
    if arr.length == 0:
        raise ValueError("Cannot compute standard deviation of an empty array.")
    
    # 计算方差
    variance_value = arr.variance()
    
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


Array.mean = mean
Array.median = median
Array.mode = mode
sfix.sqrt = sqrt
Array.square = square
Array.variance = variance
Array.std_dev = std_dev

