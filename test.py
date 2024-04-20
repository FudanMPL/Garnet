def bit_decompose(a, k):
    return [a >> i & 1 for i in range(k)]

dec_list = bit_decompose(10186733665917103528, 64)
dec_list.reverse()
print(dec_list)
print(bin(10186733665917103528))
print( ((10186733665917103528-(10186733665917103528>>(64-2)<<(64-2)))>>16))