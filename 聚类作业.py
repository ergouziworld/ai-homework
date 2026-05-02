import numpy as np  # 导入 NumPy 库，并使用别名 np 方便后续数组运算
import pandas as pd  # 导入 pandas 库，并使用别名 pd 方便后续数据处理
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，并使用别名 plt 进行绘图
from sklearn.preprocessing import StandardScaler  # 导入标准化类，用于对数据进行均值方差标准化
from sklearn.cluster import KMeans  # 导入 KMeans 聚类算法类
from sklearn.metrics import silhouette_score  # 导入 silhouette_score 函数，用于计算轮廓系数

# 生成50名学生的模拟成绩，模拟数据符合真实成绩分布
np.random.seed(42)  # 设置随机种子为 42，保证每次生成的随机数相同，结果可复现

data = {
    '语文': np.random.randint(60, 95, 50),  # 生成 50 个 60 到 94 之间的整数作为语文成绩
    '数学': np.random.randint(55, 100, 50),  # 生成 50 个 55 到 99 之间的整数作为数学成绩
    '英语': np.random.randint(62, 96, 50)  # 生成 50 个 62 到 95 之间的整数作为英语成绩
}

df = pd.DataFrame(data)  # 将生成的字典转换为 pandas DataFrame，便于数据分析和可视化

# 查看数据集前五行，检查数据是否生成正确
print(df.head())  # 输出前 5 行数据

# 数据预处理：将原始成绩进行标准化处理，消除不同科目间量纲差异
X = df.values  # 将 DataFrame 转换为 NumPy 数组，用于 sklearn 的输入格式
scaler = StandardScaler()  # 创建 StandardScaler 对象，用于后续标准化
X_scaled = scaler.fit_transform(X)  # 先拟合数据再转换数据，得到标准化后的数组

# 使用肘部法则和轮廓系数确定最佳聚类数
inertia = []  # 用于保存不同 k 值对应的簇内误差平方和
silhouette_scores = []  # 用于保存不同 k 值对应的轮廓系数
K = range(1, 11)  # 聚类数 k 的取值范围从 1 到 10
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)  # 创建 KMeans 对象，设置聚类数和随机种子
    labels = kmeans.fit_predict(X_scaled)  # 对标准化数据进行聚类，并返回聚类标签
    inertia.append(kmeans.inertia_)  # 将聚类模型的簇内误差平方和追加到 inertia 列表
    if k == 1:
        silhouette_scores.append(np.nan)  # k=1 时无法计算轮廓系数，使用 NaN 占位
    else:
        silhouette_scores.append(silhouette_score(X_scaled, labels))  # 计算并保存轮廓系数

# 设置 matplotlib 的中文字体为 SimHei，避免中文标签显示乱码
plt.rcParams['font.sans-serif'] = ['SimHei']

# 绘制肘部法则曲线，观察不同聚类数下的误差抑制情况
plt.figure(figsize=(8, 5))  # 创建一个 8x5 英寸的图形窗口
plt.plot(K, inertia, marker='o')  # 绘制聚类数 k 与 inertia 的折线图，标记点为圆圈
plt.xlabel('聚类数K')  # 设置 x 轴标签为“聚类数K”
plt.ylabel('簇内误差平方和')  # 设置 y 轴标签为“簇内误差平方和”
plt.title('肘部法则确定最佳聚类数')  # 设置图表标题
plt.xticks(K)  # 将 x 轴刻度设置为 1 到 10 的整数
plt.grid(True)  # 显示网格，便于观察曲线趋势
plt.show()  # 显示图形

# 绘制轮廓系数曲线，观察不同聚类数下的聚类效果
plt.figure(figsize=(8, 5))  # 创建一个新的 8x5 英寸图形窗口
plt.plot(K, silhouette_scores, marker='o')  # 绘制聚类数 k 与轮廓系数的折线图
plt.xlabel('聚类数K')  # 设置 x 轴标签
plt.ylabel('轮廓系数')  # 设置 y 轴标签
plt.title('轮廓系数确定最佳聚类数')  # 设置图表标题
plt.xticks(K)  # 设置 x 轴刻度为 1 到 10
plt.grid(True)  # 显示网格
plt.show()  # 显示轮廓系数图形

# 按轮廓系数选择最佳聚类数，跳过 k=1 的 NaN 值
valid_scores = silhouette_scores[1:]  # 取从 k=2 开始的轮廓系数值
optimal_k = int(np.nanargmax(valid_scores)) + 2  # 找到最大轮廓系数对应的索引，并加 2 得到实际 k 值
print(f"最佳聚类数（依据轮廓系数）: {optimal_k}")  # 输出最佳聚类数

# 使用最佳聚类数重新训练 KMeans，并生成最终聚类标签
kmeans = KMeans(n_clusters=optimal_k, random_state=42)  # 创建使用最佳 k 的 KMeans 模型
labels = kmeans.fit_predict(X_scaled)  # 对标准化数据进行聚类预测，获取最终标签

# 将聚类结果添加回原始 DataFrame，方便后续分析和可视化
df['聚类类别'] = labels  # 新增一列“聚类类别”保存每个样本的聚类标签

print(df.head())  # 输出带有聚类类别的前 5 行数据，检查结果是否正确

# 可视化聚类结果，使用数学成绩和英语成绩作为坐标
plt.figure(figsize=(8, 5))  # 创建一个 8x5 英寸的图形窗口
plt.scatter(
    df['数学'],  # x 轴数据为数学成绩
    df['英语'],  # y 轴数据为英语成绩
    c=df['聚类类别'],  # 按聚类类别着色
    cmap='viridis',  # 使用 viridis 颜色映射
    s=60,  # 点的大小为 60
    edgecolor='k'  # 点边缘颜色为黑色
)
plt.xlabel('数学成绩')  # 设置 x 轴标签
plt.ylabel('英语成绩')  # 设置 y 轴标签
plt.title(f'学生成绩聚类结果 (K={optimal_k})')  # 设置图表标题，显示最佳聚类数
plt.colorbar(label='聚类类别')  # 添加颜色条并标注为聚类类别
plt.grid(True)  # 显示网格线
plt.show()  # 显示最终聚类结果图形
#根据上述代码，写一下上述代码中可能会遇到的问题，算是学习总结，直接给200字
#在上述代码中，可能会遇到以下问题：首先，生成的模拟数据可能不完全符合真实成绩分布，导致聚类效果不佳。其次，在使用肘部法则和轮廓系数确定最佳聚类数时，可能会出现多个 k 值具有相似的指标值，难以明确选择最佳 k。此外，k=1 时无法计算轮廓系数，需要特别处理。最后，在可视化聚类结果时，如果数据点较多或颜色映射不清晰，可能会导致图表难以解读。因此，在实际应用中，需要结合领域知识和多种评估指标综合判断最佳聚类数，并注意数据质量和可视化效果。
#其次，在使用 KMeans 聚类算法时，可能会遇到局部最优解的问题，即算法可能会收敛到一个非全局最优的聚类结果。为了解决这个问题，可以尝试多次运行 KMeans，并选择具有最低簇内误差平方和的结果。此外，KMeans 对异常值较为敏感，如果数据中存在异常值，可能会严重影响聚类结果。因此，在进行聚类分析之前，建议先进行数据清洗和异常值处理，以提高聚类的准确性和稳定性。
#接下来，在数据预处理阶段，标准化处理可能会导致某些特征的分布发生变化，尤其是当原始数据存在极端值时，标准化可能会放大这些极端值的影响。因此，在进行标准化之前，建议先对数据进行探索性分析，了解数据的分布情况，并考虑是否需要进行异常值处理或使用其他更适合的数据变换方法。此外，在选择聚类数时，肘部法则和轮廓系数虽然提供了参考，但并不是绝对的标准，最终的聚类数还需要结合实际业务需求和领域知识进行综合判断。
