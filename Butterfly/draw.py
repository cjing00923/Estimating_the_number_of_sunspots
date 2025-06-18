



import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

# 加载CSV文件数据
csv_file = 'merged_file.csv'  # 请替换为您的CSV文件路径
data = pd.read_csv(csv_file)

# 提取日期，假设 'FITS File' 格式为 'YYYYMMDD_x'
data['date'] = pd.to_datetime(data['FITS File'].str.slice(0, 8), format='%Y%m%d')

# 确保日期从2011年开始
data = data[data['date'].dt.year >= 2010]

# 获取Area (MSH)列的最大值和最小值
min_area = data['Area (MSH)'].min()
max_area = data['Area (MSH)'].max()

# 自定义面积范围
custom_bins = [0, 50, 200, float('inf')]  # 手动设置的面积范围
labels = ['Small', 'Medium', 'Large']  # 对应的标签

# 根据自定义范围将数据划分为三个类别
data['Area_Category'] = pd.cut(data['Area (MSH)'], bins=custom_bins, labels=labels, include_lowest=True)

# 设置颜色字典，每个范围分配一种固定颜色
color_dict = {'Small': 'black', 'Medium': 'red', 'Large': 'yellow'}

# 映射颜色，处理缺失值（默认灰色）
data['Color'] = data['Area_Category'].map(color_dict)

# 添加 'gray' 到类别中以便处理缺失值
data['Color'] = data['Color'].astype('category')  # 确保列为类别类型
if 'gray' not in data['Color'].cat.categories:
    data['Color'] = data['Color'].cat.add_categories('gray')

# 填充缺失值为 'gray'
data['Color'] = data['Color'].fillna('gray')

# 检查 Color 列
print(data['Color'].value_counts())

# 创建绘图
plt.figure(figsize=(10, 10))

# 绘制散点图
plt.scatter(
    data['date'],  # 横坐标为日期
    data['Latitude'],  # 纵坐标为纬度
    c=data['Color'],  # 使用颜色列作为散点图颜色
    s=2,  # 方形点的大小
    alpha=0.7,  # 透明度
    marker='o',  # 使用方形标记
)

# 设置坐标轴标签和标题的字号
plt.xlabel('Date', fontsize=20)
plt.ylabel('Latitude', fontsize=20)

# 设置纵坐标的范围和刻度
plt.ylim(-45, 45)


plt.xticks(fontsize=18)
plt.yticks(range(-40, 45, 10),fontsize=18)

# 设置横坐标的日期格式，每年显示一次
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # 每年显示一次
plt.gcf().autofmt_xdate()  # 自动旋转日期

# 添加图例，显示每个范围的面积区间，并设置字体大小
small_patch = mpatches.Patch(color='black', label='area>0')  # 黑色方块
medium_patch = mpatches.Patch(color='red', label='area>50')  # 红色方块
large_patch = mpatches.Patch(color='yellow', label='area>200')  # 黄色方块
plt.legend(
    handles=[small_patch, medium_patch, large_patch],
    title='Area (MSH) Categories',
    loc='upper center',
    fontsize=14,            # 图例项字体大小
    title_fontsize=14       # 图例标题字体大小
)


# # 设置标题并调整字体大小
# plt.title('Butterfly Plot: Latitude vs Date', fontsize=20)

# 显示图像
plt.tight_layout()
plt.savefig("a.png",dpi=300)
plt.show()