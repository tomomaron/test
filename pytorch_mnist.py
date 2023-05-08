import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

def setup_all_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

batch_size = 256
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.bn1 = nn.BatchNorm1d(hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.bn2 = nn.BatchNorm1d(hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        
    def forward(self, x):
        # # Batch Normalizationなし
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        
        # Batch Normalizationあり
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.fc3(x)
        return x
    
input_size = 28*28
hidden1_size = 100
hidden2_size = 50
output_size = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net(input_size, hidden1_size, hidden2_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train_model(model, train_loader, criterion, optimizer, device='cpu'):
    train_loss = 0.0
    num_train = 0
    
    model.train()
    
    for i, (images, labels) in enumerate(train_loader):
        num_train += len(labels)
        images, labels = images.view(-1, 28*28).to(device), labels.to(device)
        
        #勾配を初期化
        optimizer.zero_grad()
        
        #推論(順伝播)
        outputs = model(images)
        
        #損失の算出
        loss = criterion(outputs, labels)
        
        #誤差逆伝播
        loss.backward()
        
        #パラメータの更新
        optimizer.step()
        
        #lossを加算
        train_loss += loss.item()
        
    #lossの平均値を取る
    train_loss = train_loss / num_train
    
    return train_loss

def test_model(model, test_loader, criterion, device='cpu'):
    test_loss = 0.0
    num_test = 0
    
    model.eval()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            num_test += len(labels)
            images, labels = images.view(-1, 28*28).to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
        
        test_loss = test_loss / num_test
        
    return test_loss

def learning(model, train_loader, test_loader, criterion, optimizer, num_epochs, device='cpu'):
    train_loss_list = []
    test_loss_list = []
    
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device=device)
        test_loss = test_model(model, test_loader, criterion, device=device)
        
        print("epoch : {}, train_loss : {:.5f}, test_loss : {:.5f}" .format(epoch, train_loss, test_loss))
        
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    
    return train_loss_list, test_loss_list

setup_all_seed()
num_epochs = 10
train_loss_list, test_loss_list = learning(model, train_loader, test_loader, criterion, optimizer, num_epochs, device=device)

plt.plot(range(len(train_loss_list)), train_loss_list, color='b', label='train loss')
plt.plot(range(len(test_loss_list)), test_loss_list, color='r', label='test loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
#plt.grid()
plt.show()
