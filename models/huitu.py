import matplotlib.pyplot as plt
import numpy as np

# 定义生成训练曲线数据的函数
def generate_training_curve(epochs, max_accuracy, noise_level=0.02):
    accuracies = []
    for epoch in range(epochs):
        # 模拟训练曲线，逐渐增加准确率，并添加一些噪声
        accuracy = max_accuracy * (1 - np.exp(-epoch / (epochs / 10))) + np.random.normal(0, noise_level)
        accuracies.append(min(accuracy, max_accuracy))  # 确保准确率不会超过最大值
    return accuracies

# 设置参数
epochs = 50
best_accuracy = 65.88

# 生成三条训练曲线数据
curve1 = generate_training_curve(epochs, best_accuracy * 0.8)
curve2 = generate_training_curve(epochs, best_accuracy)
curve3 = generate_training_curve(epochs, best_accuracy * 0.9)

# 绘制训练曲线
plt.figure(figsize=(10, 6))
plt.plot(curve1, label='Training Curve 1')
plt.plot(curve2, label='Training Curve 2')
plt.plot(curve3, label='Training Curve 3')
plt.axhline(y=best_accuracy, color='r', linestyle='--', label='Best Accuracy (65.88%)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training Curves')
plt.legend()
plt.grid(True)
plt.show()