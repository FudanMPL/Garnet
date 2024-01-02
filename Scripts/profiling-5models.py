import re

def loadfile(file_path):
    # 读取文件内容
    with open(file_path, 'r') as file:
        file_content = file.read()

    # 定义正则表达式模式
    profiling_time_pattern = r'profiling time: (\d+\.\d+)'
    compiling_time_pattern = r'compiling time: (\d+\.\d+)'

    # 在文件内容中查找匹配的数字
    profiling_match = re.search(profiling_time_pattern, file_content)
    compiling_match = re.search(compiling_time_pattern, file_content)

    # 提取数字
    if profiling_match:
        profiling_time = float(profiling_match.group(1))
        

    if compiling_match:
        compiling_time = float(compiling_match.group(1))
    
    return profiling_time, compiling_time

models = ['mpcformer', 'densenet', 'mobilenet', 'resnet', 'shufflenet']
for m in models:
    T1, T2 = [], []
    for i in range(1,6):
        t1, t2 = loadfile('./' + m+str(i)+'.txt')
        T1.append(t1)
        T2.append(t2)
    print(m)
    print(*T1, sep=' ')
    print(*T2, sep=' ')