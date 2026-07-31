# 如果没有安装 tensorflow，先在终端运行：
# pip install tensorflow matplotlib

# 导入必要的库
# tensorflow: 用于构建和训练神经网络
# tensorflow.keras: 提供高层API，包括层和模型
# matplotlib.pyplot: 用于绘制图表
# numpy: 用于数值计算
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载 MNIST 手写数字数据集
# MNIST数据集包含手写数字图片，用于训练和测试模型
# x_train: 训练图片数据，形状为 (60000, 28, 28)
# y_train: 训练标签，形状为 (60000,)
# x_test: 测试图片数据，形状为 (10000, 28, 28)
# y_test: 测试标签，形状为 (10000,)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 打印训练集和测试集的图片形状，用于确认数据加载正确
print("训练集图片形状：", x_train.shape)
print("测试集图片形状：", x_test.shape)

# 2. 数据预处理
# 原始图片是 28×28，需要变成 28×28×1，1 表示灰度图（单通道）
# reshape(-1, 28, 28, 1): -1表示自动计算样本数量，保持其他维度
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 像素值从 0~255 缩放到 0~1，以便神经网络更好地处理
# 除以255.0将像素值归一化到[0,1]区间
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. 构建 CNN 模型
# 使用Sequential模型，按顺序堆叠层
# Conv2D: 卷积层，提取图片特征
# 32个滤波器，每个3x3大小，激活函数为relu
# input_shape=(28, 28, 1): 输入图片形状
# MaxPooling2D: 最大池化层，减少参数数量和计算量，2x2池化
# 第二层Conv2D: 64个滤波器，3x3大小，relu激活
# 第二层MaxPooling2D: 2x2池化
# Flatten: 将多维输出展平为一维，用于全连接层
# Dense: 全连接层，64个神经元，relu激活
# Dense: 输出层，10个神经元（对应0-9数字），softmax激活用于多分类
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 查看模型结构，包括每层的输出形状和参数数量
model.summary()

# 4. 编译模型
# optimizer='adam': 使用Adam优化器，自动调整学习率
# loss='sparse_categorical_crossentropy': 损失函数，用于多分类问题，标签为整数
# metrics=['accuracy']: 评估指标为准确率
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. 训练模型
# 使用训练数据训练模型
# epochs=5: 训练5个周期
# batch_size=64: 每批处理64个样本
# validation_data=(x_test, y_test): 使用测试数据作为验证集
# 返回训练历史，包括损失和准确率
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# 6. 测试模型
# 在测试集上评估模型性能
# verbose=0: 不显示详细输出
# 返回测试损失和测试准确率
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("测试集损失：", test_loss)
print("测试集准确率：", test_acc)

# 7. 可视化准确率变化
# 创建新图表
plt.figure()
# 绘制训练准确率曲线
plt.plot(history.history['accuracy'], label='train accuracy')
# 绘制验证准确率曲线
plt.plot(history.history['val_accuracy'], label='test accuracy')
# 设置x轴标签为'Epoch'（训练周期）
plt.xlabel('Epoch')
# 设置y轴标签为'Accuracy'（准确率）
plt.ylabel('Accuracy')
# 设置图表标题
plt.title('CNN Accuracy')
# 显示图例
plt.legend()
# 显示图表
plt.show()

# 8. 可视化损失变化
# 创建新图表
plt.figure()
# 绘制训练损失曲线
plt.plot(history.history['loss'], label='train loss')
# 绘制验证损失曲线
plt.plot(history.history['val_loss'], label='test loss')
# 设置x轴标签
plt.xlabel('Epoch')
# 设置y轴标签
plt.ylabel('Loss')
# 设置图表标题
plt.title('CNN Loss')
# 显示图例
plt.legend()
# 显示图表
plt.show()

# 9. 随机展示 9 张预测结果
# 使用模型对前9张测试图片进行预测
# 返回预测概率数组，形状为(9, 10)
predictions = model.predict(x_test[:9])

# 创建8x8英寸的图表
plt.figure(figsize=(8, 8))
# 循环展示9张图片
for i in range(9):
    # 创建3x3子图中的第i+1个子图
    plt.subplot(3, 3, i + 1)
    # 显示第i张测试图片，reshape为28x28，灰度图
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    # 获取预测标签：概率最大的类别
    pred_label = np.argmax(predictions[i])
    # 获取真实标签
    true_label = y_test[i]
    # 设置子图标题：显示预测和真实标签
    plt.title(f"Pred: {pred_label}, True: {true_label}")
    # 关闭坐标轴
    plt.axis('off')

# 调整子图布局，避免重叠
plt.tight_layout()
# 显示图表
plt.show()
#