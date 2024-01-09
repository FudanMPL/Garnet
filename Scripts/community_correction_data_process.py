import math
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import re
import queue
import argparse


TimeInterval_XS_Zxtz = 20 * 24 * 3600  # 刑事-执行通知书距离判决时间报警阈值
TimeInterval_Rjbd = 10 * 24 * 3600  # 入矫时间距离执行通知书报警阈值


def id_process(date_str):
    result = re.sub(r'\D', '', date_str)
    return int(result)


def date_process(date):
    if isinstance(date, float):
        return 0
    date = date.replace("-", "")
    return int(date)

def convert_to_inttime(date_str):
    # 提取日期部分并删除短划线
    date_only = date_str.split()[0].replace('-', '')
    # 转换为整数
    date_num = int(date_only)
    return date_num

def convert_to_seconds(date_str):
    date_format = "%Y-%m-%d"
    date_obj = datetime.strptime(date_str, date_format)
    return int(date_obj.timestamp())


def duration_convert_to_seconds(s):
    # Regular expressions to match the patterns
    year_pattern = r'(\d+)年'
    month_pattern = r'(\d+)月'
    day_pattern = r'(\d+)日'

    # Extract years, months, and days
    years = int(re.search(year_pattern, s).group(1)) if re.search(year_pattern, s) else 0
    months = int(re.search(month_pattern, s).group(1)) if re.search(month_pattern, s) else 0
    days = int(re.search(day_pattern, s).group(1)) if re.search(day_pattern, s) else 0

    # Convert to seconds
    seconds = years * 31556952 + months * 2592000 + days * 86400

    return seconds


def process(file_path):
    data = pd.read_excel(file_path, engine='openpyxl')
    data = data[~data['证件号'].isin(['无', '无身份证号码', 'NULL', None, np.nan])]  # 删除没有身份证的
    data = data.dropna(subset=['证件号'])  # 删除没有身份证的
    data['证件号'] = data['证件号'].astype(str)
    data['证件号'] = data['证件号'].apply(id_process)  # 删除字母身份证号中的字母
    data['证件号'] = data['证件号'].astype(int)
    data['提前结束天数'] = 0
    data['间隔天数'] = 0
    result = pd.DataFrame(columns=data.columns)

    for index, ele in data.iterrows():
        if ele['矫正级别名称'] == '暂予监外执行':
            continue
        if ele['判决时间'] in ['NULL', "", np.nan] or ele['入矫日期'] in ['NULL', "", np.nan]:
            continue
        ele['间隔天数'] = (convert_to_seconds(ele['入矫日期']) - convert_to_seconds(ele['判决时间'])) // 24 // 3600
        if convert_to_seconds(ele['入矫日期']) > convert_to_seconds(ele['判决时间']) \
                and convert_to_seconds(ele['入矫日期']) - convert_to_seconds(ele['判决时间']) > (TimeInterval_XS_Zxtz + TimeInterval_Rjbd):
            # if ele['矫正期限'] not in ['NULL', "", np.nan] and ele['终止日期'] not in ['NULL', "", np.nan] and ele['矫正级别名称'] != '初期矫正':
            #     duration = duration_convert_to_seconds(ele['矫正期限'])
            #     ele['提前结束天数'] = (duration - convert_to_seconds(ele['终止日期']) + convert_to_seconds(ele['判决时间'])) // 24 // 3600
            result = result.append(ele, ignore_index=True)
            continue
        if ele['矫正期限'] in ['NULL', "", np.nan] or ele['终止日期'] in ['NULL', "", np.nan] or ele['矫正级别名称'] == '初期矫正':
            continue
        duration = duration_convert_to_seconds(ele['矫正期限'])
        ele['提前结束天数'] = (duration - convert_to_seconds(ele['终止日期']) + convert_to_seconds(ele['判决时间'])) // 24 // 3600
        if convert_to_seconds(ele['终止日期']) > convert_to_seconds(ele['判决时间'])\
                and convert_to_seconds(ele['终止日期']) - convert_to_seconds(ele['判决时间']) < duration:
            result = result.append(ele, ignore_index=True)
            continue
    result['矫正期限'] = result['矫正期限'].apply(duration_convert_to_seconds) // 24 // 3600

    result['判决时间'] = result['判决时间'].apply(date_process)  # 将时间转化为数字
    result['终止日期'] = result['终止日期'].apply(date_process)  # 将时间转化为数字
    result['入矫日期'] = result['入矫日期'].apply(date_process)  # 将时间转化为数字

    result = result[['证件号', '判决时间', '矫正期限', '入矫日期', '终止日期']]
    result = result.drop_duplicates()
    result = result.drop_duplicates(subset=['证件号'])
    file = open("./Player-Data/Input-P2-0", 'w')
    for index, row in result.iterrows():
        file.write(" ".join(str(v) for v in row) + "\n")
    file.close()
    num_rows, num_cols = result.shape
    print(f"行数: {num_rows}")
    print(f"列数: {num_cols}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Court correction data process')
    parser.add_argument('--filepath', type=str, default="./Data/community.xlsx", help='The path of excel file')
    args = parser.parse_args()
    process(args.filepath)