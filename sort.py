import re

# 文件路径
file_path = 'test.log'

# 用于存储数据的列表
data_list = []

# 读取文件内容
with open(file_path, 'r') as file:
    lines = file.readlines()
    for i in range(len(lines)):
        if "Compiling with:" in lines[i]:
            # 提取 m_max 和 n_max
            m_max = int(re.search(r'm_max\s*=\s*(\d+)', lines[i]).group(1))
            n_max = int(re.search(r'n_max\s*=\s*(\d+)', lines[i]).group(1))
            
            # 查找 t/s 值
            for j in range(i + 1, len(lines)):
                if '|' in lines[j] and 't/s' in lines[j]:
                    # 下一行是实际的 t/s 值所在行
                    ts_line = lines[j + 2]
                    value_str = re.findall(r'\s*(\d+\.\d+\s*±\s*\d+\.\d+)\s*', ts_line)[0]
                    print(value_str)
                    ts_value = float(re.findall(r'\|\s*(\d+\.\d+)\s*±\s*\d+\.\d+\s*\|', ts_line)[0])
                    data_list.append((ts_value, value_str, m_max, n_max))
                    break

# 找到 t/s 最大的那条数据
max_data = max(data_list, key=lambda x: x[0])

# 输出结果
print(f"Max t/s value: {max_data[1]}, m_max: {max_data[2]}, n_max: {max_data[3]}")
