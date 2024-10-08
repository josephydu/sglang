import re

with open("a.txt", mode="r") as f:
    text = f.read()

# 使用正则表达式匹配所有数字（包括负数和小数）
# numbers = re.findall(r'else time = (-?\d*\.?\d+(?:[eE][-+]?\d+)?)', text)
# numbers = re.findall(r"send radix time = (-?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
# numbers = re.findall(r'real dispatch time = (-?\d*\.?\d+(?:[eE][-+]?\d+)?)', text)
# numbers = re.findall(r'find max idx = (-?\d*\.?\d+(?:[eE][-+]?\d+)?)', text)
# numbers = re.findall(r'find max idx = (-?\d*\.?\d+(?:[eE][-+]?\d+)?)', text)

numbers = re.findall(r"scheduler time = (-?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)


# 将匹配到的字符串转换为浮点数
numbers = [float(num) for num in numbers]

# 计算总和
total_sum = sum(numbers)

print("匹配到的数字:", len(numbers))
print("总和:", total_sum)

# # import re

# # # 示例字符串
# # data = """
# # scheduler time = 0.0001246929168701172
# # match time = 3.0994415283203125e-06
# # find max idx = 1.1920928955078125e-06
# # len two = 1.239776611328125e-05
# # real dispatch time = 7.43865966796875e-05
# # scheduler time = 9.965896606445312e-05
# # """

# # # 使用正则表达式匹配所有scheduler time的数字，包括科学计数法
# # scheduler_times = re.findall(r"scheduler time\s*=\s*([-+]?\d*\.\d+|\d+)([eE][-+]?\d+)?", data)

# # # 提取数字并将其转换为浮点数
# # total_sum = sum(float(f"{num[0]}{num[1] if num[1] else ''}") for num in scheduler_times)

# # print("匹配到的scheduler time总和为:", total_sum)
# # print(0.0001246929168701172 + 9.965896606445312e-05)


# import re

# # 假设你的数据存储在一个文本文件中，文件名为 'gpu_logs.txt'
# file_path = 'g.txt'

# # 初始化一个字典来存储每个GPU的统计信息
# gpu_stats = {}

# # 正则表达式模式
# pattern = r'\[gpu=(\d+)\] .*? #running-req: (\d+), .*? #queue-req: (\d+)'

# # 读取文件并处理每一行
# with open(file_path, 'r') as file:
#     for line in file:
#         match = re.search(pattern, line)
#         if match:
#             gpu_id = match.group(1)
#             running_req = int(match.group(2))
#             queue_req = int(match.group(3))

#             # 初始化GPU统计信息
#             if gpu_id not in gpu_stats:
#                 gpu_stats[gpu_id] = {'running_req': 0, 'queue_req': 0}

#             # 更新统计信息
#             gpu_stats[gpu_id]['running_req'] += running_req
#             gpu_stats[gpu_id]['queue_req'] += queue_req

# # 打印结果
# for gpu_id, stats in gpu_stats.items():
#     print(f"GPU {gpu_id}: #running-req: {stats['running_req']}, #queue-req: {stats['queue_req']}")


# import re

# # 示例字符串
# with open('a.txt', mode='r') as f:
#     data = f.read()

# # 正则表达式模式，匹配中括号内的数字
# pattern = r'choose1==>\[(\d+),(\d+)\]'

# # 找到所有匹配的数字对
# matches = re.findall(pattern, data)

# # 统计大于500的数量
# count_greater_than_500 = 0
# count_less_than_500 = 0
# thresold = 3000
# for match in matches:
#     # 将匹配的字符串转换为整数
#     num1, num2 = int(match[0]), int(match[1])

#     # 检查是否大于500
#     if num2 > thresold:
#         count_greater_than_500 += 1
#     else:
#         count_less_than_500 += 1

# # 输出结果
# print(f"大于{thresold}的数量: {count_greater_than_500}")
# print(f"小于{thresold}的数量: {count_less_than_500}")


# import re
# from collections import defaultdict

# # 用于存储每个 req 对应的 index 统计
# req_index_count = defaultdict(lambda: defaultdict(int))

# # 正则表达式匹配
# pattern = r'index = (\d+), req = (\d+)'

# index_count_dict = defaultdict(int)
# # 读取文件
# with open('a.txt', 'r') as file:  # 替换 'your_file.txt' 为你的文件名
#     for line in file:
#         match = re.search(pattern, line)
#         if match:
#             index = int(match.group(1))  # 提取 index
#             req = int(match.group(2))     # 提取 req

#             index_count_dict[index] += 1
#             # 直接更新计数
#             req_index_count[req][index] += 1  # 统计 index 的出现次数

# # 输出统计结果
# for req, index_counts in req_index_count.items():
#     print(f"req = {req}:")
#     for index, count in index_counts.items():
#         print(f"  index {index}: {count} times")

# # 将结果转换为普通字典
# result_dict = {req: dict(index_counts) for req, index_counts in req_index_count.items()}
# import json
# # 将结果保存为 JSON 文件
# with open('result.json', 'w') as json_file:  # 替换 'result.json' 为你想要的文件名
#     json.dump(result_dict, json_file, indent=4)

# print(index_count_dict)
# # 输出结果字典
# print("结果已保存为 result.json")
