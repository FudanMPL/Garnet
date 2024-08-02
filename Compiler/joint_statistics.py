from Compiler.types import Array 
from Compiler.types import sint
from Compiler.group_ops import GroupSum
from Compiler.types import sfix  

    # start point
def mean(self):
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
    
    # Sum all elements
    total_sum = sum(vector)
    # Compute the mean
    mean_value = total_sum/self.length
    
    return mean_value

def median(self):
    """Calculate the median of the array.
    
    :returns: Median value of the array elements.
    """
    if not isinstance(self[0], sfix):
        raise TypeError("This function can only be executed for arrays of type sfix.")
    if self.length == 0:
        raise ValueError("Cannot compute median of an empty array.")
    self.sort()
    # vector = self.get_vector()
    # sorted_vector = vector.sort()
    sorted_vector = self.get_vector()
    mid_index = self.length // 2
    
    if self.length % 2 == 0:
        median_value = (sorted_vector[mid_index - 1] + sorted_vector[mid_index]) / 2
    else:
        median_value = sorted_vector[mid_index]
    
    return median_value

def mode(self):

    if not isinstance(self[0], sfix):
        raise TypeError("This function can only be executed for arrays of type sfix.")
    if self.length == 0:
        raise ValueError("Cannot compute median of an empty array.")
    self.sort()
    sorted_data = self.get_vector()
    n = self.length
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
Array.mean = mean
Array.median = median
Array.mode = mode

