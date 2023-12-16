import math


def plain_psi(filename1, filename2,n):
    ids1 = list()
    ids2 = list()
    lines1 = list()
    lines2 = list()
    with open(filename1, 'r') as file1:
        lines1.extend(file1.readlines()[:n])
        for line in lines1:
            ids1.append(int(line))


    with open(filename2, 'r') as file2:
        lines2.extend(file2.readlines()[:n])
        for line in lines2:
            ids2.append(int(line))

    
    ids = list(set(ids1).intersection(set(ids2)))
    ids.sort()
    print(ids)
    id1 = list()
    id2 = list()
    for id in ids:
        id1.append(ids1.index(id))
        id2.append(ids2.index(id))
    print(id1,id2)
    return id1,id2


def plain_merge_feature(id0,id1, f0, f1,n):
    lines0 = list()
    lines1 = list()
    with open(f0, 'r') as file0:
        lines0 = file0.readlines()[:n]
    with open(f1, 'r') as file1:
        lines1 = file1.readlines()[:n]   
    result = list()
    l = len(id0)
    for i in range(0,l):
        result.append([lines0[id0[i]],lines1[id1[i]]])
    return result

pn = 2

n = 6
f0_num = 7
f1_num = 7
base_path = "./Player-Data/PSI/"

id0,id1 = plain_psi(base_path+'ID-P0', base_path+'ID-P1',n)
assert len(id0)==len(id1)
fs = plain_merge_feature(id0,id1, base_path+'F-P0', base_path+'F-P1',n)
print(fs)