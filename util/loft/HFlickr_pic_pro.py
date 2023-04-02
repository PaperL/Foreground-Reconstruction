import os
import random
import shutil

# 指定数据文件夹和输出文件夹
original_data_folder = "/data/tanlinfeng/IHD/HFlickr"
data_folder = "/data/smb_shared/@Dataset/IHD_FR/HFlickr/rerun"
backup_data_folder = "/data/smb_shared/@Dataset/IHD_FR/HFlickr/foreground_images_fix"
output_folder = "/data/smb_shared/@Dataset/IHD_FR/HFlickr/rerun_examples"

# 定义参数1的范围
ranges = [(0, 150), (150, 250), (250, 350)]

# 定义每个范围要挑选的数据点数量
num_points_per_range = 15

# 读取数据文件
with open("HFlickr_fMSE.csv", "r") as f:
    lines = f.readlines()

# 去掉第一行（表头）

# 将数据点按参数1的值从小到大排序
lines.sort(key=lambda x: float(x.split(",")[1]))

# 初始化一个字典，用于存储每个范围已经挑选了多少个数据点
num_points_selected = {r: 0 for r in ranges}

# 初始化一个列表，用于存储挑选出来的数据点名称
selected_points = {r: [] for r in ranges}

# 遍历每个数据点
for line in lines:
    # 获取参数1的值
    param1 = float(line.split(",")[1])
    
    # 遍历每个范围
    for r in ranges:
        # 如果参数1在当前范围内，并且当前范围还没有挑选够数量的数据点
        if r[0] <= param1 < r[1] and num_points_selected[r] < num_points_per_range + 10000:
            # 将该数据点名称加入到 selected_points 列表中，并将该范围已经挑选的数量加一
            selected_points[r].append(line.split(",")[0])
            num_points_selected[r] += 1
            
            # 跳出当前循环，继续处理下一个数据点
            break

print([len(x) for x in selected_points.values()])

# 遍历每个范围，将对应范围内挑选出来的数据点复制到输出文件夹中
for i, r in enumerate(ranges):
    # 创建对应范围的输出文件夹
    folder_name = f'range_{i}_{r[0]}-{r[1]}'
    os.makedirs(os.path.join(output_folder, folder_name), exist_ok=True)
    
    # 随机选择 num_points_per_range 个数据点，并将它们复制到对应范围的输出文件夹中
    for j in range(num_points_per_range):
        point_name = random.choice(selected_points[r])
        selected_points[r].remove(point_name)
        name_parts = point_name.split('_')
        try:
            name_parts = point_name.split('_')
            shutil.copy(os.path.join(data_folder, f"{point_name}.jpg"), os.path.join(output_folder, folder_name, f"{point_name}_fore.jpg"))
        except FileNotFoundError:
            shutil.copy(os.path.join(backup_data_folder, f"{point_name}.jpg"), os.path.join(output_folder, folder_name, f"{point_name}_fore.jpg"))
        shutil.copy(os.path.join(original_data_folder, f"composite_images/{point_name}.jpg"), os.path.join(output_folder, folder_name, f"{point_name}_comp.jpg"))
        shutil.copy(os.path.join(original_data_folder, f"masks/{'_'.join(name_parts[:2])}.png"), os.path.join(output_folder, folder_name, f"{point_name}_mask.png"))
        shutil.copy(os.path.join(original_data_folder, f"real_images/{'_'.join(name_parts[:1])}.jpg"), os.path.join(output_folder, folder_name, f"{point_name}_real.jpg"))