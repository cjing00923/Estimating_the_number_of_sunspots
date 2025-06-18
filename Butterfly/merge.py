import pandas as pd

# 读取两个CSV文件
df1 = pd.read_csv('sunspot_latitudes.csv')  # 第一个文件
df2 = pd.read_csv('boxmengban_all_area.csv')  # 第二个文件

# 合并两个DataFrame，按照"FITS File"这一列进行合并
merged_df = pd.merge(df1, df2, on="FITS File")

# 保存合并后的数据到新文件
merged_df.to_csv('merged_file.csv', index=False)

# 打印合并后的数据
print(merged_df)
