n = 1024
f0 = 7
f1 = 7
f = f0

base_path = "./Player-Data/PSI/"
pn = 2

import generate_fake_data as gd

# generate fake data
for i in range(pn):
    gd.generate_random_ids_to_file(n, base_path+'ID-P'+str(i))
    gd.generate_random_numbers_to_file(n, f, base_path+'F-P'+str(i))
    
import plain_psi as pp

fs = pp.psi(n,f0,f1)
# print(fs)

import os

# with open('test_psi.mpc','w') as file:
#     content = 'f0 = %d\nf1 = %d\nn = %d\nresult,num = PSI(n,f0,f1)\nprint_str("num:")\nprint_int(num)\nprint_str("\n")\nprint_ln("result=\%\s",result.reveal())'%(f0,f1,n)

print(os.system('bash Scripts/PSI-Test/run_psi.sh r'))

print(os.system('bash Scripts/PSI-Test/run_psi.sh t'))

with open('Scripts/PSI-Test/out.txt', 'r') as file:
    content = file.readlines()
    for line in content:
        if line[0:8] == "result=[":
            tmps = line[8:len(line)-2].split(", ")
            l = len(fs)
            # print(l)
            for i in range(0,l):
                assert int(tmps[i]) == fs[i]
            for j in range(l,len(tmps)):
                assert tmps[j] == '0'  
            break