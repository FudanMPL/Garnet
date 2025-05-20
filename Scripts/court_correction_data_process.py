import sys

import pandas as pd
from datetime import datetime
import re
import queue
import argparse





def id_process(id_str):
    l = 18 - len(id_str)
    for i in range(l):
        id_str = id_str + "\\x00"
    return id_str

def get_number_code(str):
    str = str[-13:-1]
    return int(str)

def process(file_path):
    data = pd.read_excel(file_path, engine='openpyxl')
    data = data[~data['证件号码'].isin(['无', '无身份证号码', 'NULL', None])]  # 删除没有身份证的
    data['证件号码'] = data['证件号码'].astype(str)
    data['证件号码'] = data['证件号码'].apply(id_process)
    data['部门受案号'] = data['部门受案号'].apply(get_number_code)
    data['部门受案号'] = data['部门受案号'].astype(int)
    result = data[['证件号码'] ]
    result = result.drop_duplicates(subset=['证件号码'])
    file = open("./Player-Data/Input-P1-0", 'w')
    for index, row in result.iterrows():
        file.write(" ".join(str(v) for v in row) + "\n")
    file.close()
    num_rows, num_cols = result.shape
    print(f"行数: {num_rows}")
    print(f"列数: {num_cols}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Court correction data process')
    parser.add_argument('--filepath', type=str, default="./Data/court.xlsx", help='The path of excel file')
    args = parser.parse_args()
    process(args.filepath)