import sys

import pandas as pd
from datetime import datetime
import re
import queue
import argparse


name_length = 8  # 名字的最长长度
id_length = 18  # 身份证号的最长长度
number_length = 25  # 处方单号的最长长度


def convert_to_seconds(date_str):
    date_format = "%Y-%m-%d %H:%M:%S"
    date_obj = datetime.strptime(date_str, date_format)
    return int(date_obj.timestamp())


def id_process(id_str):
    id_str = id_str[:min(len(id_str), id_length)]
    l = id_length - len(id_str)
    for i in range(l):
        id_str = id_str + "\\x00"
    return id_str

def date_process(date_str):
    # 提取日期部分并删除短划线
    date_only = date_str.split()[0].replace('-', '')
    # 转换为整数
    date_num = int(date_only)
    return date_num


def name_process(name_str):
    name_str = name_str[:min(len(name_str),name_length)]
    l = name_length - len(name_str)
    for i in range(l):
        name_str = name_str + "\\x00"
    return name_str


def number_process(number_str):
    number_str = number_str[:min(len(number_str),number_length)]
    l = number_length - len(number_str)
    for i in range(l):
        number_str = number_str + "\\x00"
    return number_str


def convert_to_inttime(date_str):
    # 提取日期部分并删除短划线
    date_only = date_str.split()[0].replace('-', '')
    # 转换为整数
    date_num = int(date_only)
    return date_num




def process_each_person(data):
    data = data.sort_values(by=['秒时间'], ascending=[True])
    data = data.reset_index(drop=True)
    q = queue.Queue()
    count = 0
    result = pd.DataFrame(columns=data.columns)
    for index, row in data.iterrows():
        while (not q.empty()) and (row['秒时间'] - q.queue[0]['秒时间'] > check_time):
            if count < threshold:
                ele = q.get()
                count = count - ele['倍数']
            else:
                count = 0
                start_date = q.queue[0]['开具时间']
                start_id = q.queue[0]['处方单号']
                while not q.empty():
                    ele = q.get()
                    count = count + ele['倍数']
                    ele['累积倍数'] = count
                    ele['起始日期'] = start_date
                    ele['起始单号'] = start_id
                    ele['结束日期'] = ele['开具时间']
                    ele['结束单号'] = ele['处方单号']
                    if count >= threshold:
                        result = result.append(ele, ignore_index=True)
                count = 0
        count = count + row['倍数']
        q.put(row)
    return result


def process(file_path):
    data = pd.read_excel(file_path, engine='openpyxl')
    data = data[data['是否领药'] == '是']  # 删除未领药字段
    data = data.drop_duplicates(subset=['处方单号'])  # 删除重复的字段
    data['患者姓名'] = data['患者姓名'].apply(name_process)
    data['倍数'] = data['标准开药量（mg）'] / (14 * data['标准用药量（mg）'])
    data['秒时间'] = data['开具时间'].apply(convert_to_seconds)
    data['开具时间'] = data['开具时间'].apply(date_process)
    data['处方单号'] = data['处方单号'].astype(str)
    data['处方单号'] = data['处方单号'].apply(number_process)
    data['患者证件号'] = data['患者证件号'].astype(str)
    data['患者证件号'] = data['患者证件号'].apply(id_process)
    data['起始单号'] = 0
    data['起始日期'] = 0
    data['结束单号'] = 0
    data['结束日期'] = 0
    data['累积倍数'] = 0
    ids = data['患者证件号'].unique()
    result = pd.DataFrame(columns=data.columns)
    for id in ids:
        personal_data = process_each_person(data[data['患者证件号'] == id])
        result = pd.concat([result, personal_data], ignore_index=True)
    result = result[['患者证件号', '患者姓名',  '累积倍数', '起始单号', '起始日期', '结束单号', '结束日期', ] ]
    result['累积倍数'] = result['累积倍数'].round(2)
    file = open("./Player-Data/Input-P1-0", 'w')
    for index, row in result.iterrows():
        file.write(" ".join(str(v) for v in row) + "\n")
    file.close()
    num_rows, num_cols = result.shape

    print(f"行数: {num_rows}")
    print(f"列数: {num_cols}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Drug data process')
    parser.add_argument('--filepath', type=str, default="./Data/drug.xlsx", help='The path of excel file')
    parser.add_argument('--checkdays', type=int, default=14,  help='The continue days to check')
    parser.add_argument('--threshold', type=float, default=3,  help='The threshold for judgement')

    args = parser.parse_args()
    check_time = args.checkdays * 24 * 60 * 60
    threshold = args.threshold
    process(args.filepath)