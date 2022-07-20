from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import List, Dict
from functools import partial
from warnings import warn
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
import json


class CKA(object):
    '''support timm version model'''

    def __init__(self, model1: nn.Module,
                 model2: nn.Module,
                 model1_name: str = None,
                 model2_name: str = None,
                 model1_layers: List[str] = None,
                 model2_layers: List[str] = None,
                 device: str = 'cpu'):
        """
        :param model1: (nn.Module) timm version pytorch model
        :param model2: (nn.Module) timm version pytorch model
        :param model1_name: (str) timm version pytorch model name
        :param model2_name: (str) timm version pytorch model name
        :param model1_layers: (List) list of layer names to extract features from
        :param model2_layers: (List) list of layer names to extract features from
        :param device: device to run the model (cuda, cpu)
        """
        self.model1 = model1
        self.model2 = model2
        self.device = device

        # model_info: model configuration info
        self.model1_info = {}
        self.model2_info = {}

        if model1_name is None:
            self.model1_info['Name'] = model1.__repr__().split('(')[0]
        else:
            self.model1_info['Name'] = model1_name

        if model2_name is None:
            self.model2_info['Name'] = model2.__repr__().split('(')[0]
        else:
            self.model2_info['Name'] = model2_name

        if self.model1_info['Name'] == self.model2_info['Name']:
            warn(f"Both model have identical names - {self.model2_info['Name']}. "
                 "It may cause confusion when interpreting the results. "
                 "Consider giving unique names to the models :)")

        if model1_layers is None:
            self.model1_layers = self.extract_valid_layer(model1)
        else:
            self.model1_layers = model1_layers

        if len(list(model1.modules())) > 150 and model1_layers is None:
            warn("Model 1 seems to have a lot of layers. "
                 "Consider giving a list of layers whose features you are concerned with "
                 "through the 'model1_layers' parameter. Your CPU/GPU will thank you :)")

        if model2_layers is None:
            self.model2_layers = self.extract_valid_layer(model2)
        else:
            self.model2_layers = model2_layers

        if len(list(model2.modules())) > 150 and model2_layers is None:
            warn("Model 2 seems to have a lot of layers. "
                 "Consider giving a list of layers whose features you are concerned with "
                 "through the 'model2_layers' parameter. Your CPU/GPU will thank you :)")

        self.model1_info['Layers'] = []
        self.model2_info['Layers'] = []

        self._insert_hooks()
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)
        self.model1.eval()
        self.model2.eval()

        self.hookLayersActivationDict1 = {}
        self.hookLayersActivationDict2 = {}

    def extract_valid_layer(self, model: nn.Module) -> List[str]:
        '''extract layers with same batch dimension'''
        feature_dimension = {}
        input = torch.randn(3, 3, 224, 224).to(self.device)

        def hook_fn(name: str, module: nn.Module, input: torch.Tensor,
                    output: torch.Tensor):
            feature_dimension[name] = output.shape[0]

        for name, layer in model.named_modules():
            layer.register_forward_hook(partial(hook_fn, name))
        _ = model(input)
        valid_layer_names = [
            k for k, v in feature_dimension.items() if v == 3]
        return valid_layer_names

    def getActivation(self, model_name, layer_name):
        '''create hook to extract feature map'''
        def hook(model, input, output):
            if model_name == 'model1':
                self.hookLayersActivationDict1[layer_name] = output.detach()
            if model_name == 'model2':
                self.hookLayersActivationDict2[layer_name] = output.detach()
        return hook

    def _insert_hooks(self):
        for name, layer in self.model1.named_modules():
            if name in self.model1_layers:
                self.model1_info['Layers'] += [name]
                layer.register_forward_hook(
                    self.getActivation('model1', name))
        for name, layer in self.model2.named_modules():
            if name in self.model2_layers:
                self.model2_info['Layers'] += [name]
                layer.register_forward_hook(
                    self.getActivation('model2', name))

    def pairwise_distances(self, x):
        '''compute euclidean distance'''
        instances_norm = torch.sum(x**2, -1).reshape((-1, 1))
        return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()

    def GaussianKernelMatrix(self, x, sigma=0.9):
        '''compute Gram matrix using Gaussian kernel'''
        pairwise_distances_ = self.pairwise_distances(x)
        return torch.exp(-pairwise_distances_ / sigma)

    def HSIC(self, x, y, s_x=1, s_y=1):
        '''compute HSIC score'''
        m, _ = x.shape
        K = self.GaussianKernelMatrix(x, s_x)
        L = self.GaussianKernelMatrix(y, s_y)
        H = torch.eye(m) - 1.0/m * torch.ones((m, m))
        H = H.cuda()
        hsicScore = torch.trace(
            torch.mm(L, torch.mm(H, torch.mm(K, H))))/((m-1)**2)
        return hsicScore

    def compare(self, dataloader1: DataLoader, dataloader2: DataLoader = None):
        """
        Computes the feature similarity between the models on the given datasets.
        :param dataloader1: dataloader for model 1 (DataLoader)
        :param dataloader2: dataloader for model 2 (DataLoader) 
        If given, model 2 will run on this dataset. (default = None)
        """
        if dataloader2 is None:
            warn(
                "Dataloader for Model 2 is not given. Using the same dataloader for both models.")
            dataloader2 = dataloader1

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[
            0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[
            0]

        num_batches = min(len(dataloader1), len(dataloader2))

        for (x1, *_), (x2, *_) in tqdm(zip(dataloader1, dataloader2), total=num_batches):
            _ = self.model1(x1.to(self.device))
            _ = self.model2(x2.to(self.device))
            break

        self.col1, self.col2, self.hsicScoreList = [], [], []
        for layer1 in self.model1_layers:
            for layer2 in self.model2_layers:
                oaL1 = self.hookLayersActivationDict1[layer1].reshape(
                    self.hookLayersActivationDict1[layer1].size(0), -1)
                oaL2 = self.hookLayersActivationDict2[layer2].reshape(
                    self.hookLayersActivationDict2[layer2].size(0), -1)
                hsicCross = self.HSIC(oaL1, oaL2).detach().item()
                hsicL1 = self.HSIC(oaL1, oaL1).detach().item()
                hsicL2 = self.HSIC(oaL2, oaL2).detach().item()
                denom = np.sqrt(hsicL1 * hsicL2)
                if np.isnan(hsicCross) or np.isnan(hsicL1) or np.isnan(hsicL2) or denom == 0:
                    continue
                finalScore = hsicCross/denom
                self.col1.append(self.model1_layers.index(layer1))
                self.col2.append(self.model2_layers.index(layer2))
                self.hsicScoreList.append(finalScore)

    def save_cka_info(self, cka_info, file_path):
        with open(file_path, 'w') as fp:
            json.dump(cka_info, fp, indent=4)

    def export(self, file_path) -> Dict:
        '''Exports the CKA data along with the respective model layer names.'''
        cka_info = {
            "model1_name": self.model1_info['Name'],
            "model2_name": self.model2_info['Name'],
            "model1_layers": self.model1_info['Layers'],
            "model2_layers": self.model2_info['Layers'],
            "dataset1_name": self.model1_info['Dataset'],
            "dataset2_name": self.model2_info['Dataset'],
            "hsicScoreList": self.hsicScoreList,
            "col1": self.col1,
            "col2": self.col2
        }
        self.save_cka_info(cka_info, file_path)
        return cka_info


def plot_cka_graph(col1, col2, hsicScoreList, title, model1_name, model2_name, fig_save_path):
    '''plot cka diagram using pivot'''
    hsicData = {'L1': col1, 'L2': col2, 'hsic': hsicScoreList}
    hsicDataFrame = pd.DataFrame(data=hsicData)
    hsicDataFrame = hsicDataFrame.pivot("L1", "L2", "hsic")
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(hsicDataFrame.T, cmap='YlGnBu')
    plt.xlabel("Layers " + model2_name, fontsize='36')
    plt.ylabel("Layers " + model1_name, fontsize='36')
    if title is not None:
        ax.set_title(f"{title}", fontsize=36)
    else:
        ax.set_title(f"{model1_name} vs {model2_name}", fontsize=36)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks(ax.get_xticks()[::4])
    ax.set_yticks(ax.get_yticks()[::4])
    ax.invert_yaxis()
    plt.savefig(fig_save_path, fmt='png', bbox_inches='tight')
