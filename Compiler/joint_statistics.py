from Compiler.types import Array 
from Compiler.types import sint, regint, regint
import math
from Compiler import util
from Compiler.group_ops import GroupSum, GroupPrefixSum, PrefixSum


    # start point
def mean(self):
    """Calculate the mean of the array.
    
    :returns: Mean value of the array elements.
    """
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

# def mode(self):
#     if self.length == 0:
#         raise ValueError("Cannot compute mode of an empty array.")
    
#     self.sort()
#     sorted_vector = self.get_vector()
    
#     modes = []
#     max_count = 1
#     current_count = 1
#     current_value = sorted_vector[0]

#     for i in range(1, self.length):
#         is_same = int(sorted_vector[i] == current_value)
#         current_count = current_count + is_same
#         is_different = 1 - is_same
        
#         max_count_update = int(current_count > max_count)
#         max_count = max_count * (1 - max_count_update) + current_count * max_count_update
#         modes_update = max_count_update
#         modes_reset = modes_update * [current_value]
#         modes_extend = (int(current_count == max_count)) * [current_value] * (1 - max_count_update)
        
#         modes = modes_reset if max_count_update else (modes + modes_extend)
        
#         current_value = sorted_vector[i] * is_different + current_value * (1 - is_different)
#         current_count = current_count * (1 - is_different) + is_different
    
#     final_modes_update = int(current_count > max_count)
#     modes = (final_modes_update * [current_value]) + ((int(current_count == max_count)) * [current_value] * (1 - final_modes_update)) * (1 - final_modes_update) + modes * (1 - final_modes_update)

#     return modes
    
# 

# def mode(self):
#         """Calculate the mode of the array using a secure method.

#         :returns: Mode value of the array elements.
#         """
#         if self.length == 0:
#             raise ValueError("Cannot compute mode of an empty array.")
        
#         vector = self.get_vector()
#         n = self.length
        
#         # Phase 1: Find a candidate for the majority element using a secure method
#         candidate = self.value_type(0)
#         count = self.value_type(0)

#         for num in vector:
#             equal_count_zero = (count == 0)
#             equal_num_candidate = (num == candidate)
            
#             candidate = equal_count_zero.if_else(num, candidate)
#             count = equal_count_zero.if_else(
#                 self.value_type(1), 
#                 equal_num_candidate.if_else(count + 1, count - 1)
#             )
        
#         return candidate

# def count_elements(self):
#     # Initialize an empty array for element counts
#     element_counts = Array(len(self), sint, address=regint.Array(len(self)))  # adjust size and type as needed
#     count_index = regint(0)

#     # Iterate through the array elements
#     i = regint(0)
#     while_true = regint(1)  # 用于控制循环的逻辑变量
#     while while_true:
#         element = self[i]
#         found = regint(0)
#         j = regint(0)
#         while_true_inner = regint(1)
#         while while_true_inner:
#             # Use an MPC-compatible comparison
#             is_equal = (element_counts[j][0] == element)
#             element_counts[j][1] = element_counts[j][1] + is_equal
#             found = found + is_equal

#             # Update j and check the condition
#             j += 1
#             while_true_inner = (j < count_index).if_else(regint(1), regint(0))

#         # If not found, add a new element count
#         is_new_element = (found == 0)
#         element_counts[count_index][0] = is_new_element.if_else(element, element_counts[count_index][0])
#         element_counts[count_index][1] = is_new_element.if_else(1, element_counts[count_index][1])
#         count_index = count_index + is_new_element

#         # Update i and check the condition
#         i += 1
#         while_true = (i < len(self)).if_else(regint(1), regint(0))

#     return element_counts, count_index

# def mode(self):
#     element_counts, count_index = self.count_elements()
#     max_count = sint(0)
#     mode_element = sint(0)
    
#     i = regint(0)
#     while_true = regint(1)
#     while while_true:
#         count = element_counts[i][1]
#         is_max = (count > max_count)
#         max_count = is_max.if_else(count, max_count)
#         mode_element = is_max.if_else(element_counts[i][0], mode_element)

#         # Update i and check the condition
#         i += 1
#         while_true = (i < count_index).if_else(regint(1), regint(0))
    
#     return mode_element
# Existing methods...

# end point

Array.mean = mean
Array.median = median
Array.mode = mode
# Array.unique_elements = unique_elements
# Array.count_elements = count_elements
