import csv  
from collections import defaultdict  
  
def read_csv_and_list_headers_and_unique_values(file_path):  
    headers = []  
    unique_values = defaultdict(set)  
  
    with open(file_path, 'r', encoding='utf-8') as file:  
        reader = csv.reader(file)  
  
        # 读取头部  
        headers = next(reader, None)  
        if headers is None:  
            print("CSV文件为空或格式不正确。")  
            return  
  
        # 读取所有行并收集唯一值  
        for row in reader:  
            for header, value in zip(headers, row):  
                unique_values[header].add(value)  
  
    # 打印头部和对应的所有可能值  
    print("头部:")  
    print(headers)  
  
    print("\n对应的所有可能值:")  
    for header, values in unique_values.items():  
        print(f"{header}: {list(values)}")  # 将values转换为列表并打印  
  
# 使用示例  
file_path = 'RTA Dataset.csv'  # 替换为你的CSV文件路径  
read_csv_and_list_headers_and_unique_values(file_path)