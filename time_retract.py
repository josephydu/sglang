import re

txt = """
[build_tree_kernel time] = 0.09763813018798828
[build_tree_kernel time] = 0.0008273124694824219
[build_tree_kernel time] = 0.0007874965667724609
[build_tree_kernel time] = 0.0007970333099365234
[build_tree_kernel time] = 0.0007903575897216797
[build_tree_kernel time] = 0.0007984638214111328
[build_tree_kernel time] = 0.0007832050323486328
[build_tree_kernel time] = 0.0008039474487304688
[build_tree_kernel time] = 0.00078582763671875
[build_tree_kernel time] = 0.0008006095886230469
[build_tree_kernel time] = 0.0007951259613037109
[build_tree_kernel time] = 0.0008156299591064453
[build_tree_kernel time] = 0.0008256435394287109
[build_tree_kernel time] = 0.0008144378662109375
[build_tree_kernel time] = 0.0007944107055664062
[build_tree_kernel time] = 0.0008013248443603516
[build_tree_kernel time] = 0.0008156299591064453
[build_tree_kernel time] = 0.0008115768432617188
[build_tree_kernel time] = 0.0008177757263183594
[build_tree_kernel time] = 0.0007796287536621094
"""


def calculate_total_time_from_string(data):
    total_time = 0.0

    # 按行分割字符串
    lines = data.split("\n")

    for line in lines:
        # 提取时间值
        if "[build_tree_kernel time] =" in line:
            time_str = line.split("=")[1].strip()
            time_value = float(time_str)
            total_time += time_value

    return total_time


if __name__ == "__main__":
    total_time = calculate_total_time_from_string(txt)
    print(f"Total build_tree_kernel time: {total_time} seconds")
