'''
Author: SkyTu 1336923451@qq.com
Date: 2023-11-01 10:45:39
LastEditors: SkyTu 1336923451@qq.com
LastEditTime: 2023-11-06 16:08:54
FilePath: /txy/Garnet/GPU/test_result.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''





res1 = "562ad57ee6bab17c5087da7998441014"
res2 = "0c042faa53f6482a97868fe63fb5433b"
s = "5a2efad4b54cf956c701559fa7f1532f"
print(bin(int(res1, 16)))
print(bin(int(res2, 16)))
print(bin(int(res1, 16) ^ int(res2, 16)))
print(bin(int(s, 16)))
print(int(res1, 16) ^ int(res2, 16))
print(int(s, 16))

