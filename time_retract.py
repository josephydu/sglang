def calculate_total_time(file_path):
    total_time = 0.0

    with open(file_path, "r") as file:
        for line in file:
            # 提取时间值
            if "[build_tree_kernel time] =" in line:
                time_str = line.split("=")[1].strip()
                time_value = float(time_str)
                total_time += time_value

    return total_time


if __name__ == "__main__":
    file_path = "times.txt"  # 假设你的数据保存在 times.txt 文件中
    total_time = calculate_total_time(file_path)
    print(f"Total build_tree_kernel time: {total_time} seconds")
