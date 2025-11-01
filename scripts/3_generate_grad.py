#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from tqdm import tqdm

from pytorch_grad_cam import GradCAM, ShapleyCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision
from torchvision import transforms
from torchvision import models
import json
from lime import lime_image

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

    def _load_data(self):
        self.data = []
        self.label = []
        self.paths = []
        for case in ["normal", "accident"]:
            for f in os.listdir(os.path.join(self.data_dir, case)):
                file_path = os.path.join(self.data_dir, case, f)
                if ".parquet" not in file_path: continue
                data = pd.read_parquet(file_path)
                self.data.append(data.values)
                if case == "normal": self.label.append(0)
                else: self.label.append(1)
                self.paths.append(file_path)
        self.data = np.expand_dims(np.array(self.data), axis=-1).repeat(3, axis=-1)
        if self.max != 0 :
            max = self.max
            min = self.min
        else:
            max = self.data.flatten().max()
            min = self.data.flatten().min()
        self.data = ((self.data - min) / (max - min) * 255).astype(np.float32)
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
        return data, self.label[idx], self.paths[idx]
    
#%%

def calc_accuracy(args):
    args.model.eval()
    correct = 0
    total = 0

    preds = []
    labs = []

    acc_as_acc = []
    norm_as_norm = []
    norm_as_acc = []
    acc_as_norm = []

    with torch.no_grad():
        for idx, (images, labels, paths) in enumerate(args.test_loader):

            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = args.model(images).squeeze(-1)
            _, predicted = torch.max(outputs.data, 1)
            # print(labels.shape, predicted.shape, len(test_paths_part))

            # predicted = torch.where(outputs > 0.45, 1, 0).type(torch.float)
            total += labels.size(0)
            preds.append(predicted.to("cpu").numpy())
            labs.append(labels.to("cpu").numpy())
            acc_as_acc.append((predicted == labels).cpu().numpy() & (labels.to("cpu").numpy() == 1))
            norm_as_norm.append((predicted == labels).cpu().numpy() & (labels.to("cpu").numpy() == 0))
            norm_as_acc.append((predicted != labels).cpu().numpy() & (labels.to("cpu").numpy() == 0))
            acc_as_norm.append((predicted != labels).cpu().numpy() & (labels.to("cpu").numpy() == 1))
            correct += (predicted == labels).sum().item()
    test_acc = correct / total * 100

    with torch.no_grad():
        for idx, (images, labels, paths) in enumerate(args.train_loader):

            images = images.to(args.device)
            labels = labels.to(args.device)

            outputs = args.model(images).squeeze(-1)
            _, predicted = torch.max(outputs.data, 1)
            # predicted = torch.where(outputs > 0.45, 1, 0).type(torch.float)
            total += labels.size(0)
            preds.append(predicted.to("cpu").numpy())
            labs.append(labels.to("cpu").numpy())
            # print(labels)
            acc_as_acc.append((predicted == labels).cpu().numpy() & (labels.to("cpu").numpy() == 1))
            norm_as_norm.append((predicted == labels).cpu().numpy() & (labels.to("cpu").numpy() == 0))
            norm_as_acc.append((predicted != labels).cpu().numpy() & (labels.to("cpu").numpy() == 0))
            acc_as_norm.append((predicted != labels).cpu().numpy() & (labels.to("cpu").numpy() == 1))
            correct += (predicted == labels).sum().item()
    train_acc = correct / total * 100
    print(f"Train accuracy : {train_acc} / Test accuracy : {test_acc}")
    num_TT, num_FF = np.array(acc_as_acc).flatten().sum(), np.array(norm_as_norm).flatten().sum(), 
    num_TF, num_FT = np.array(norm_as_acc).flatten().sum(), np.array(acc_as_norm).flatten().sum(), 
    print(f"TT : {num_TT} / FF : {num_FF} / TF : {num_TF} / FT : {num_FT}")


def Cell_values_from_GradCAM(args):
    
    # GPU 사용 가능 여부 확인
   
    if args.name == "VGG":
        target_layers = list(list(list(args.model.children())[-2].children())[-3].children())[-3] # VGG
    else:
        target_layers = list(list(list(args.model.children())[-2].children())[-3].children())[-1] # ResNet, EfficientNet, ResNext
    

    args.model.eval()
    if args.method == "Shap":
        cam = ShapleyCAM(model=args.model, target_layers=[target_layers])
    elif args.method == "GradCAM":
        cam = GradCAM(model=args.model, target_layers=[target_layers]) 
    
    metadata_whole = {}
    for idx, (images, labels, paths) in enumerate(tqdm(args.test_loader)):
        images = images.to(args.device)
        
        with torch.no_grad():
            output = args.model(images).squeeze(-1)
        _, preds = torch.max(output.data, 1)
        targets = [ClassifierOutputTarget(pred) for pred in preds.to("cpu").numpy()]
        grayscale_cams = cam(input_tensor=images, targets=targets)
        
        # return
        
        for grayscale_cam, pred, label, file_path in zip(grayscale_cams, preds, labels, paths):
            metadata = {}
            file_name = os.path.basename(file_path)
            # print(file_name)
            year, month, day, cur_hour, minute, lat, lon = file_name.rstrip('.parquet').split("_")
            metadata = {
                "pred" : pred.item(),
                "label" : label.item(),
                "timestamp" : pd.Timestamp(year=int(year), month=int(month), day=int(day), hour=int(cur_hour), minute=int(minute)).strftime("%Y-%m-%d %H:%M"),
                "lat": lat,
                "lon": lon.rstrip(".parquet")
            }
            metadata_whole[file_name] = metadata
            
            pd.DataFrame(grayscale_cam).to_parquet(os.path.join(args.save_dir, "SHAP_"+file_name), index=False, engine="pyarrow")

    torch.cuda.empty_cache()

    with open(f'./results/xai_results/{args.method}_metadata/{args.method}_metadata_{args.name}_{args.dist}.json', 'w') as outfile:
        json.dump(metadata_whole, outfile)
                
    return True
        
def Cell_values_from_LIME(args):

    def batch_predict(images):
        batch = torch.from_numpy(images).to(args.device).permute(0, 3, 1, 2)  
        with torch.no_grad():
            output = args.model(batch).squeeze(-1)
        probs = F.softmax(output, dim=1)
        return probs.detach().cpu().numpy()

    args.model.eval()
    explainer = lime_image.LimeImageExplainer()
    
    metadata_whole = {}
    for idx, (images, labels, paths) in enumerate(tqdm(args.test_loader)):
        images = images.to(args.device)
        inputs = images.cpu().detach().numpy().transpose(0, 2, 3, 1)

        with torch.no_grad():
            output = args.model(images).squeeze(-1)
        _, preds = torch.max(output.data, 1)

        
        for input, pred, label, file_path in zip(inputs, preds, labels, paths):
            metadata = {}
            file_name = os.path.basename(file_path)
            # print(file_name)
            year, month, day, cur_hour, minute, lat, lon = file_name.rstrip('.parquet').split("_")
            explanation = explainer.explain_instance(input,
                                                    batch_predict, # classification function
                                                    top_labels=1, 
                                                    hide_color=0, 
                                                    num_samples=1000)
            
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
            pd.DataFrame(mask).to_parquet(os.path.join(args.save_dir, "lime_"+file_name), index=False, engine="pyarrow")
            plt.imshow(mask)
            plt.close()
            metadata = {
                "pred" : pred.item(),
                "label" : label.item(),
                "timestamp" : pd.Timestamp(year=int(year), month=int(month), day=int(day), hour=int(cur_hour), minute=int(minute)).strftime("%Y-%m-%d %H:%M"),
                "lat": lat,
                "lon": lon.rstrip(".parquet")
            }
            metadata_whole[file_name] = metadata

    
    torch.cuda.empty_cache()

    with open(f'{args.meta_dir}/metadata_{args.name}_{args.dist}.json', 'w') as outfile:
        json.dump(metadata_whole, outfile)

if __name__ == "__main__":
    
    class TempArgs():
        pass

    args = TempArgs()
    args.batch_size = 16
    args.random_seed = 123
    args.method = "GradCAM"
    # args.learning_rate = 0.001
    args.learning_rate = "0_001"
    args.pretrained = 1
    args.exp_dir = f"./data/exp_data/{args.random_seed}"

    args.grad_dir = f"./results/xai_results/{args.method}/{args.random_seed}/{args.learning_rate}/{args.pretrained}"
    os.makedirs(args.grad_dir, exist_ok=True)

    args.meta_dir = f"./{args.method}_metadata/{args.random_seed}/{str(args.learning_rate).replace('.', '_')}/{str(args.pretrained)}"
    os.makedirs(args.meta_dir, exist_ok=True)

    args.weight_dir = f"./weights/{args.random_seed}/{args.learning_rate}/{args.pretrained}"
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dicts = {
        "VGG": models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
    }
    
    # isAccident = True
    par_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), * [os.pardir] * 2))
    data_dir = os.path.join(par_dir, "data")

    

    for dist in [10, 20, 30]:
        args.dist = dist

        train_dataset = ParquetDataset(data_dir=f"{args.exp_dir}/{str(args.dist)}/train")
        test_dataset = ParquetDataset(data_dir=f"{args.exp_dir}/{str(args.dist)}/test", minmax=train_dataset.get_minmax())
        args.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        args.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        for name, model in model_dicts.items():
            weight_path = os.path.join(args.weight_dir, f"{name}_{dist}.pth")
            args.name = name
            # print(list(list(model.children())[-3].children())[-1])
            args.save_dir = os.path.join(args.grad_dir, f"{name}_{dist}")
            os.makedirs(args.save_dir, exist_ok=True)
            args.model = CustomModel(model, isVit=False, isTransfer=False).to(args.device)
            args.model.load_state_dict(torch.load(weight_path))
            
            save_dir = os.path.join(args.grad_dir, f"{name}_{dist}")
            os.makedirs(save_dir, exist_ok=True)
            
            if args.method == "Shap" or args.method == "GradCAM":
                Cell_values_from_GradCAM(args)
            elif args.method == "Lime":
                Cell_values_from_LIME(args)
            else:
                print("Invalid method")
            print("Done getting grad cam values")
            break
        break

