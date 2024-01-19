import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import random
import torchvision.datasets as datasets

from random import choice

device = "cpu"

class CustomLoss(nn.Module):
    
    def __init__(self, beta):
        super(CustomLoss, self).__init__()
        self.beta = beta
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(self, outputs, labels):
        ce_loss = self.cross_entropy(outputs, labels)
        
        # 计算损失函数值的80%和90%分位数（按损失函数值大小排序）
        sorted_ce_loss, _ = torch.sort(ce_loss, descending=True)
        idx_80th = int(0.2 * ce_loss.size(0))
        idx_90th = int(0.1 * ce_loss.size(0))
        loss_80th_percentile = sorted_ce_loss[idx_80th]
        loss_90th_percentile = sorted_ce_loss[idx_90th]
        
        # 处理损失函数值小于等于80%分位数的样本
        mask_80th = ce_loss <= loss_80th_percentile
        modified_loss = ce_loss * mask_80th.float()
        
        # 处理损失函数值在80%和90%分位数之间的样本
        mask_90th = (ce_loss > loss_80th_percentile) & (ce_loss <= loss_90th_percentile)
        modified_loss += (ce_loss * self.beta) * mask_90th.float()
        
        # 处理损失函数值大于90%分位数的样本
        mask_gt_90th = ce_loss > loss_90th_percentile
        max_loss = torch.maximum(torch.max(ce_loss), torch.tensor(1.0))
        modified_loss += (ce_loss * self.beta / max_loss) * mask_gt_90th.float()

        # 计算加权平均
        modified_loss = torch.mean(modified_loss)
        
        return modified_loss
    
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomCrop(32), 
    transforms.ToTensor()])

class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

batch_size = 32
num_classes = 2
learning_rate = 0.001
num_epochs = 300

noisy_ratio_list = [0.1, 0.2, 0.3, 0.4]
beta_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
for noisy_ratio in noisy_ratio_list:
    for beta in beta_list:
        dataset_train = datasets.ImageFolder('./chest_xray/train', transform)
        # 对应文件夹的label
        print(dataset_train.class_to_idx)
        dataset_test = datasets.ImageFolder('./chest_xray/val', transform)
        # 对应文件夹的label
        print(dataset_test.class_to_idx)
        # 导入数据
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

        model = ConvNet(num_classes).to(device)
        criterion = CustomLoss(beta)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        torch.manual_seed(3407)
        np.random.seed(10)
        random.seed(20)
        total_step = len(train_loader)
        loss_all = []
        acc_noisy_list = []
        acc_true_list = []
        for epoch in range(num_epochs):
            model.train()
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                labels_np = labels.clone()
                num_samples_to_change_train = int(noisy_ratio * len(labels_np))
                change_indices_train = np.random.choice(len(labels_np), num_samples_to_change_train, replace=False)
                labels_changed = labels_np.clone()

                for each in labels_np[change_indices_train]:
                    result_list = list(range(0, num_classes))
                    result_list.remove(each)
                    labels_changed[change_indices_train] = choice(result_list)
                
                outputs = model(images)
                loss = criterion(outputs, labels_changed)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                
            if (epoch + 1) % 2 == 0:
                model.eval()
                loss_all.append(loss.detach().cpu().numpy())
                with torch.no_grad():
                    correct_true = 0
                    correct_noisy = 0
                    total = 0
                    for testimages, testlabels in test_loader:
                        testimages = testimages.to(device)
                        testlabels = testlabels.to(device)
                        
                        outputs = model(testimages)
                        _, predicted = torch.max(outputs.data, 1)
                        total += testlabels.size(0)
                        correct_true += (predicted == testlabels).sum().item()

                        testlabels_np = testlabels.clone()
                        num_samples_to_change_testlabels = int(noisy_ratio * len(testlabels_np))
                        change_indices_testlabels = np.random.choice(len(testlabels_np), num_samples_to_change_testlabels, replace=False)
                        testlabels_changed = testlabels_np.clone()

                        for eachtest in testlabels_np[change_indices_testlabels]:
                            result_list = list(range(0, num_classes))
                            result_list.remove(eachtest)
                            testlabels_changed[change_indices_testlabels] = choice(result_list)

                        correct_noisy += (predicted == testlabels_changed).sum().item()
                    
                    accuracy_val_noisy = correct_noisy / total
                    accuracy_val_true = correct_true / total
                    
                    print(f"Validation accuracy on noisy after {epoch + 1} epochs: {accuracy_val_noisy}")
                    print(f"Validation accuracy on true after {epoch + 1} epochs: {accuracy_val_true}")

                    acc_noisy_list.append(accuracy_val_noisy)
                    acc_true_list.append(accuracy_val_true)
                model.train()

                
        pd.DataFrame(loss_all).to_csv("./results_chest_xray/loss_chest_xray_" + str(noisy_ratio) + "_" + str(beta) + ".csv")
        pd.DataFrame(acc_noisy_list).to_csv("./results_chest_xray/acc_noisy_chest_xray_" + str(noisy_ratio) + "_" + str(beta) + ".csv")
        pd.DataFrame(acc_true_list).to_csv("./results_chest_xray/acc_true_chest_xray_" + str(noisy_ratio) + "_" + str(beta) + ".csv")
