from torch.utils.data import DataLoader
from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt
from typing import List, Dict
from functools import partial
from warnings import warn
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import json
import torch


class CKA:
    '''support timm version vit based model'''

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

        # store extracted feature map from hook
        self.model1_features = []
        self.model2_features = []

    def extract_valid_layer(self, model: nn.Module) -> List[str]:
        '''extract layers with same batch dimension'''
        feature_dimension = {}
        input = torch.randn(3, 3, 224, 224)

        def hook_fn(name, module: nn.Module, input: torch.Tensor,
                    output: torch.Tensor):
            feature_dimension[name] = output.shape[0]

        for name, layer in model.named_modules():
            layer.register_forward_hook(partial(hook_fn, name))
        _ = model(input)
        valid_layer_names = [k for k, v in feature_dimension.items() if v == 3]
        return valid_layer_names

    def _log_layer(self, model: str, name: str, layer: nn.Module,
                   inp: torch.Tensor, out: torch.Tensor):
        if model == "model1":
            self.model1_features[name] = out
        elif model == "model2":
            self.model2_features[name] = out
        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        for name, layer in self.model1.named_modules():
            if name in self.model1_layers:
                self.model1_info['Layers'] += [name]
                layer.register_forward_hook(
                    partial(self._log_layer, "model1", name))

        for name, layer in self.model2.named_modules():
            if name in self.model2_layers:
                self.model2_info['Layers'] += [name]
                layer.register_forward_hook(
                    partial(self._log_layer, "model2", name))

    def _HSIC(self, K: torch.Tensor, L: torch.Tensor) -> float:
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        if K.shape != L.shape:
            return 0.0
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) /
                   ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        result = (1 / (N * (N - 3)) * result).item()
        if np.isnan(result):
            return 0.0
        return result

    def compare(self, dataloader1: DataLoader,
                dataloader2: DataLoader = None):
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

        N = len(self.model1_layers) if self.model1_layers is not None else len(
            list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(
            list(self.model2.modules()))

        self.hsic_matrix = torch.zeros(N, M, 3)

        num_batches = min(len(dataloader1), len(dataloader2))

        for (x1, *_), (x2, *_) in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):

            self.model1_features = {}
            self.model2_features = {}
            _ = self.model1(x1.to(self.device))
            _ = self.model2(x2.to(self.device))

            for i, (name1, feat1) in enumerate(self.model1_features.items()):
                X = feat1.flatten(1)
                K = X @ X.t()
                K.fill_diagonal_(0.0)
                self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

                for j, (name2, feat2) in enumerate(self.model2_features.items()):
                    Y = feat2.flatten(1)
                    L = Y @ Y.t()
                    L.fill_diagonal_(0.0)
                    self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                    self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches

        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                        self.hsic_matrix[:, :, 2].sqrt())
        self.hsic_matrix = torch.nan_to_num(self.hsic_matrix)

    def save_cka_info(self, cka_info, file_path):
        with open(file_path, 'w') as fp:
            json.dump(cka_info, fp, indent=4)

    def export(self, file_path) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        """
        cka_info = {
            "model1_name": self.model1_info['Name'],
            "model2_name": self.model2_info['Name'],
            "model1_layers": self.model1_info['Layers'],
            "model2_layers": self.model2_info['Layers'],
            "dataset1_name": self.model1_info['Dataset'],
            "dataset2_name": self.model2_info['Dataset'],
            "CKA": self.hsic_matrix.numpy().tolist()}
        self.save_cka_info(cka_info, file_path)
        return cka_info


def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def save_results(save_path, title, model1_name, model2_name, hsic_matrix):
    fig, ax = plt.subplots()
    im = ax.imshow(hsic_matrix, origin='lower', cmap='magma')
    ax.set_ylabel(f"Layers From {model1_name}", fontsize=15)
    ax.set_xlabel(f"Layers From {model2_name}", fontsize=15)
    if title is not None:
        ax.set_title(f"{title}", fontsize=18)
    else:
        ax.set_title(f"{model1_name} vs {model2_name}", fontsize=18)
    add_colorbar(im)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()
