import math


def plain_psi(filename1, filename2):
    ids1 = set()
    ids2 = set()
    with open(filename1, 'r') as file1:
        lines1 = file1.readlines()
        for line in lines1:
            ids1.add(int(line))

    with open(filename2, 'r') as file2:
        lines2 = file2.readlines()
        for line in lines2:
            ids2.add(int(line))

    ids = list(ids1.intersection(ids2))
    return ids


def plain_merge_feature(ids, f1, f2):
    with open(f1, 'r') as file1:
        

m = 20
n = 7
base_path = "./Player-Data/PSI/"
pn = 2

ids = plain_psi(base_path+'ID_P0', base_path+'ID_P1')
fs = plain_merge_feature(ids, base_path+'F_P0', base_path+'F_P1')