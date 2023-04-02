import pandas as pd

# 读取 txt 文件
txt_file = pd.read_csv('./HCOCO_info.txt', sep=' ', header=None, names=['name', 'param1', 'param2', 'param3'])
txt_file['name'] = txt_file['name'].str.replace('.jpg', '')

# 读取 csv 文件
csv_file = pd.read_csv('./HCOCO_fMSE.csv', header=None, names=['name', 'param1', 'param2', 'param3'])

# 合并文件
merged_file = pd.concat([csv_file, txt_file], axis=0, ignore_index=True)
merged_file.drop_duplicates(subset=['name'], keep='first', inplace=True)

# 保存文件
merged_file.to_csv('./HCOCO_merged.csv', index=False, header=False)