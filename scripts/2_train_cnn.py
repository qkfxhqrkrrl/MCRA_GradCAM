#%%

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random
import shutil

from torchvision import models
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
# %%

def train_test_split(exp_dir, data_par_dir, dist, test_size=0.3):
    print("\nSplitting data...")
    normal_data_paths = []
    accident_data_paths = []
    for acc_date in os.listdir(data_par_dir):
        if not os.path.isdir(os.path.join(data_par_dir, acc_date)): continue
        data_dir = os.path.join(data_par_dir, acc_date)
        file_names = os.listdir(data_dir)
        # print(file_names)
        lat, lon = file_names[0].rstrip(".parquet").split("_")[-2:]

        for h in range(4, 12):
            normal_date = pd.to_datetime(acc_date, format="%Y_%m_%d_%H") - pd.Timedelta(hours=h)
            normal_year, normal_month, normal_day, normal_hour = normal_date.strftime("%Y_%m_%d_%H").split("_")

            doExist = False
            for min in range(0, 60, 5):
                target_name = f"{normal_year}_{int(normal_month):02d}_{int(normal_day):02d}_{int(normal_hour)}_{min}_{lat}_{lon}.parquet"
                if target_name in file_names: 
                    normal_data_paths.append(os.path.join(data_dir, target_name))
                    doExist = True
                else: continue
            if doExist: break
        
        for min in range(0, 60, 5):
            acc_year, acc_month, acc_day, acc_hour = acc_date.split("_")
            target_name = f"{acc_year}_{int(acc_month):02d}_{int(acc_day):02d}_{acc_hour}_{min}_{lat}_{lon}.parquet"
            if target_name in file_names: accident_data_paths.append(os.path.join(data_dir, target_name))
            else: continue
        # break
    
    random.shuffle(normal_data_paths)
    random.shuffle(accident_data_paths)

    train_normal_paths, test_normal_paths = normal_data_paths[:-int(len(normal_data_paths) * test_size)], normal_data_paths[-int(len(normal_data_paths) * test_size):]
    train_accident_paths, test_accident_paths = accident_data_paths[:-int(len(accident_data_paths) * test_size)], accident_data_paths[-int(len(accident_data_paths) * test_size):]

    train_normal_dir = os.path.join(exp_dir, str(dist), "train", "normal")
    train_accident_dir = os.path.join(exp_dir, str(dist), "train", "accident")
    test_normal_dir = os.path.join(exp_dir, str(dist), "test", "normal")
    test_accident_dir = os.path.join(exp_dir, str(dist), "test", "accident")

    if not os.path.exists(train_normal_dir) or len(os.listdir(train_normal_dir)) == 0 :
        os.makedirs(train_normal_dir, exist_ok=True)
        for normal_path in tqdm(train_normal_paths):
            shutil.copy(normal_path, train_normal_dir)
            
    if not os.path.exists(train_accident_dir) or len(os.listdir(train_accident_dir)) == 0 :
        os.makedirs(train_accident_dir, exist_ok=True)
        for accident_path in tqdm(train_accident_paths):
            shutil.copy(accident_path, train_accident_dir)

    if not os.path.exists(test_normal_dir) or len(os.listdir(test_normal_dir)) == 0 :
        os.makedirs(test_normal_dir, exist_ok=True)
        for normal_path in tqdm(test_normal_paths):
            shutil.copy(normal_path, test_normal_dir)

    if not os.path.exists(test_accident_dir) or len(os.listdir(test_accident_dir)) == 0 :
        os.makedirs(test_accident_dir, exist_ok=True)
        for accident_path in tqdm(test_accident_paths):
            shutil.copy(accident_path, test_accident_dir)
    
    print("\nData split done...")

#%%

class ParquetDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, minmax = (0, 0)):
        self.data_dir = data_dir
        self.min, self.max = minmax
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self._load_data()
        print("Data loaded...")

    def _load_data(self):
        self.data = []
        self.label = []
        for case in ["normal", "accident"]:
            for f in os.listdir(os.path.join(self.data_dir, case)):
                if ".parquet" not in f: continue
                data = pd.read_parquet(os.path.join(self.data_dir, case, f))
                self.data.append(data.values)
                if case == "normal": self.label.append(0)
                else: self.label.append(1)
        self.data = np.expand_dims(np.array(self.data), axis=-1).repeat(3, axis=-1)
        if self.max != 0 :
            max = self.max
            min = self.min
        else:
            max = self.data.flatten().max()
            min = self.data.flatten().min()
        self.data = ((self.data - min) / (max - min) * 255)
        self.label = torch.tensor(self.label, dtype=torch.long)
        # self.data = ((self.data - min) / (max - min))
        return data

    def get_minmax(self):
        return self.min, self.max
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(data)
        return data, self.label[idx]
    
    
class CustomModel(nn.Module):
    def __init__(self, pre_trained, isVit = False, isTransfer=False) -> None:
        super().__init__()
        self.pretrained_model = pre_trained
        
        self.linear = nn.Sequential(
            nn.LayerNorm(1000),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(1000, 2),
        )
        
        self.linear.apply(self._init_weights)
        self.isTransfer = isTransfer
        self.isVit = isVit
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="leaky_relu",)
        
    def forward(self, X):
        if self.isTransfer:
            if self.isVit :
                with torch.no_grad():
                    out = self.pretrained_model(pixel_values=X)
                out = self.linear(out.logits)
            else :
                with torch.no_grad():
                    out = self.pretrained_model(X)
                out = self.linear(out)
        else :
            if self.isVit :
                out = self.pretrained_model(pixel_values=X)
                out = self.linear(out.logits)
            else :
                out = self.pretrained_model(X)
                out = self.linear(out)
            
        return out


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=3):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.crit = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        BCE_loss = self.crit(inputs, targets)
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return loss.mean()

                
def one_hot_encode(labels, num_classes, device):
    # One-hot encoding implementation
    onehot = torch.eye(num_classes).type(torch.FloatTensor).to(device)
    return onehot[labels]

def train(args):
    
    best_acc = float("-inf")
    best_f1 = float("-inf")
    optimizer = args.optimizer
    scheduler = args.scheduler
    train_loss = 0.0
    correct = 0
    total = 0
    
    # 정확도와 손실 값을 저장할 리스트 초기화
    train_acc_list = []
    test_acc_list = []
    loss_list = []
    loop = tqdm(range(args.num_epochs))
    for epoch in loop:
        
        for idx_epoch, (images, labels) in enumerate(args.train_loader):
            args.model.train()
            images = images.to(args.device, dtype=torch.float32)
            labels = labels.to(args.device)
            labels_onehot = one_hot_encode(labels, 2, args.device)
            optimizer.zero_grad()

            outputs = args.model(images).squeeze(-1)
            loss = args.criterion(outputs, labels_onehot)

            torch.nn.utils.clip_grad_norm_(args.model.parameters(), 20)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total * 100
        train_acc_list.append(train_acc)
        train_loss /= len(args.train_loader)
        loss_list.append(train_loss)

        args.model.eval()
        correct = 0
        total = 0

        preds = []
        labs = []
        with torch.no_grad():
            for images, labels in args.test_loader:
                images = images.to(args.device, dtype=torch.float32)
                labels = labels.to(args.device)

                outputs = args.model(images).squeeze(-1)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                preds.append(predicted.to("cpu").numpy())
                labs.append(labels.to("cpu").numpy())
                correct += (predicted == labels).sum().item()
        test_f1 = f1_score(np.concatenate(labs), np.concatenate(preds))

        test_acc = correct / total * 100
        test_acc_list.append(test_acc)

        if test_acc >= best_acc :
            best_acc = test_acc
        
        if test_f1 >= best_f1 :
            torch.save(args.model.state_dict(), os.path.join(args.weight_dir, f"{args.model_name}_{args.dist}.pth"))
            best_f1 = test_f1
            
        if (epoch+1) % 100 == 0:
            torch.save(args.model.state_dict(), os.path.join(args.weight_dir, f"{args.model_name}_{args.dist}_{epoch+1}.pth"))
        
        loop.set_description(f'Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%, Best_acc: {best_acc:.2f}, Best_f1: {best_f1:.2f}')
    
    return train_acc_list, test_acc_list, loss_list

def main():
    class TempArgs():
        pass

    args = TempArgs()
    args.random_seed = 123
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.data_dir = "./data/prq_data"
    args.num_epochs = 100
    args.batch_size = 4
    args.lr = 0.001
    args.gamma = 0.8 # Proportion to adjust learning rate
    args.weight_dir = "./weights"
    args.split_ratio = 0.3
    args.exp_dir = f"./data/exp_data/{args.random_seed}"
    os.makedirs(args.exp_dir, exist_ok=True)
    args.result_dir = "./results"
    
    model_dicts = {
        "resnext" : torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl'),
        "efficient" : models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1),
        "mobile" : models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
        "VGG": models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
        "resnet" : models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
    }

    for dist in [5, 10, 20, 30]:
        args.dist = dist
        exp_data_dir = os.path.join(args.data_dir, f"rtr_{args.dist*10}by{args.dist*10}_filtered")
        train_test_split(args.exp_dir, exp_data_dir, args.dist, test_size=args.split_ratio)
        train_dataset = ParquetDataset(data_dir=f"{args.exp_dir}/{str(args.dist)}/train")
        test_dataset = ParquetDataset(data_dir=f"{args.exp_dir}/{str(args.dist)}/test", minmax=train_dataset.get_minmax())
        args.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        args.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        args.criterion = FocalLoss()
        for name, model in model_dicts.items():
            print("Experiment : ", dist, name)
            args.model_name = name
            args.model = CustomModel(model, isVit=False, isTransfer=False).to(args.device)
            args.optimizer = torch.optim.Adam(args.model.parameters(), lr=args.lr)
            args.scheduler = torch.optim.lr_scheduler.StepLR(args.optimizer, step_size=20, gamma=args.gamma)
            train_acc_list, test_acc_list, loss_list = train(args)

            plt.plot(range(1, args.num_epochs+1), train_acc_list, label='Train Accuracy')
            plt.plot(range(1, args.num_epochs+1), test_acc_list, label='Test Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(os.path.join(args.result_dir, "training_results", f"{name}_{args.dist}_accuracy.jpg"))
            plt.title(f"{name}_accuracy")
            plt.close()
            # plt.show()
        
            plt.plot(range(1, args.num_epochs+1), loss_list) 
            plt.title(f"{name}_loss")
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(args.result_dir, "training_results", f"{name}_{args.dist}_loss.jpg"))
            plt.close()
        
            model = model.to("cpu")
            model = None

    

if __name__ == "__main__":
    main()

#%%