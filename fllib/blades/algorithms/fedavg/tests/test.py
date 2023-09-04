import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 2, True)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.Sigmoid()(x)
        return x


# 定义训练数据和目标
features = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
targets = torch.tensor([1, 0, 0])

# 定义超参数
learning_rate = 0.1
num_epochs = 2000

# 初始化模型和优化器
model = SimpleNet()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 训练模型
for epoch in range(num_epochs):
    # 前向传递
    outputs = model(features.float())
    loss = criterion(outputs, targets)

    # 反向传递和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印损失
    if (epoch + 1) % 100 == 0:
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))

# 评估模型
model.eval()
with torch.no_grad():
    outputs = model(features.float())
    _, predicted = torch.max(outputs.data, 1)
    accuracy = torch.sum(predicted == targets).item() / targets.size(0)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
