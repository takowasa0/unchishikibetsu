import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

# データセットのパス
base_path = os.path.dirname(os.path.abspath(__file__))  # このスクリプトの場所を基準にする
train_data_path = os.path.join(base_path, '../data/train')  # 訓練データ
val_data_path = os.path.join(base_path, '../data/val')      # 検証データ
model_save_path = os.path.join(base_path, '../models/poop_classifier.pth')  # 保存先

# データセットの準備
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_data_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = datasets.ImageFolder(val_data_path, transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# モデルの読み込みとカスタマイズ
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 出力クラス数を2（💩とその他）に設定

# 損失関数とオプティマイザ
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# トレーニングループ
num_epochs = 10
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# モデルの保存
torch.save(model, model_save_path)
print(f"モデルが保存されました: {model_save_path}")
