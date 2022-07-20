from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import json


class Trainer(object):

    def __init__(self, model, num_epochs, device, train_loader, validation_loader, test_loader, optimizer,
                 criterion, log_path, model_path, conf_path, metric_path, scheduler=None):
        '''
        :param log_path: (str) path to save learning curve data
        :param model_path: (str) path to save trained model
        :param conf_path: (str) path to save confusion matrix
        :param metric_path: (str) path to save classification report
        '''
        self.model = model
        self.num_epochs = num_epochs
        self.device = device
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.log_path = log_path
        self.model_path = model_path
        self.conf_path = conf_path
        self.metric_path = metric_path

    def train_one_epoch(self, model, device, train_loader, optimizer, criterion,
                        epoch_number, num_epochs, scheduler=None):
        running_train_loss = 0.0
        running_train_correct_predictions = 0
        num_items = 0
        model.train()
        with tqdm(train_loader, total=len(train_loader)) as loop:
            for data in loop:
                optimizer.zero_grad()
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                num_items += inputs.size(0)
                running_train_loss += loss.item()
                avg_loss = running_train_loss / num_items
                pred = outputs.argmax(dim=1, keepdim=True)
                running_train_correct_predictions += pred.eq(
                    labels.view_as(pred)).sum().item()
                avg_accuracy = running_train_correct_predictions * 100 / num_items
            loop.set_description(f"Epoch [{epoch_number+1}/{num_epochs}]")
            loop.set_postfix(loss=avg_loss, acc=avg_accuracy,
                             lr=optimizer.param_groups[0]['lr'])
            if scheduler:
                scheduler.step()
        return running_train_loss, running_train_correct_predictions

    def validate_one_epoch(self, model, device, validation_loader, criterion, epoch_number, num_epochs):
        running_validation_loss = 0.0
        running_validation_correct_predictions = 0
        num_items = 0
        model.eval()
        with tqdm(validation_loader, total=len(validation_loader)) as loop:
            with torch.no_grad():
                for data in loop:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    num_items += inputs.size(0)
                    running_validation_loss += loss.item()
                    avg_loss = running_validation_loss / num_items
                    pred = outputs.argmax(dim=1, keepdim=True)
                    running_validation_correct_predictions += pred.eq(
                        labels.view_as(pred)).sum().item()
                    avg_accuracy = running_validation_correct_predictions * 100 / num_items
                loop.set_description(
                    f"Epoch [{epoch_number+1}/{num_epochs}]")
                loop.set_postfix(val_loss=avg_loss, val_acc=avg_accuracy)
        return running_validation_loss, running_validation_correct_predictions

    def finetune_model(self):
        best_validation_loss = float('inf')
        train_acc = []
        train_loss = []
        val_acc = []
        val_loss = []
        state_dict = None

        for epoch in range(self.num_epochs):
            running_train_loss, running_train_accuracy = self.train_one_epoch(
                self.model, self.device, self.train_loader, self.optimizer,
                self.criterion, epoch, self.num_epochs, self.scheduler)
            running_validation_loss, running_validation_accuracy = self.validate_one_epoch(
                self.model, self.device, self.validation_loader,
                self.criterion, epoch, self.num_epochs)

            avg_train_loss = running_train_loss / \
                len(self.train_loader.dataset)
            avg_validation_loss = running_validation_loss / \
                len(self.validation_loader.dataset)
            avg_train_accuracy = running_train_accuracy / \
                len(self.train_loader.dataset)
            avg_validation_accuracy = running_validation_accuracy / \
                len(self.validation_loader.dataset)

            train_acc.append(avg_train_accuracy)
            train_loss.append(avg_train_loss)
            val_acc.append(avg_validation_accuracy)
            val_loss.append(avg_validation_loss)

            if avg_validation_loss < best_validation_loss:
                best_validation_loss = avg_validation_loss
                state_dict = self.model.state_dict()

        torch.save(state_dict, self.model_path)
        self.save_training_data(train_acc, train_loss, val_acc,
                                val_loss, self.log_path)

    def save_training_data(self, train_acc, train_loss, val_acc, val_loss, file_path):
        data = {'train_acc': train_acc,
                'train_loss': train_loss,
                'val_acc': val_acc,
                'val_loss': val_loss}
        with open(file_path, 'w') as fp:
            json.dump(data, fp, indent=4)

    def evaluate_model(self):
        y_pred = []
        y_true = []
        for data in self.test_loader:
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            with torch.no_grad():
                output = self.model(inputs)
            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)
            labels = labels.data.cpu().numpy()
            y_true.extend(labels)
        classes = self.test_loader.dataset.classes
        cf_matrix = confusion_matrix(y_true, y_pred, normalize="true")
        df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                             columns=[i for i in classes])
        df_cm.to_csv(self.conf_path)
        info = f"Test Accuracy: {np.sum(np.array(y_pred) == np.array(y_true))/len(y_pred)*100} %"
        print(info)
        self.save_metrics(y_true, y_pred)

    def save_metrics(self, y_true, y_pred):
        labels = ['Bacterial Blight (CBB)',
                  'Brown Streak Disease (CBSD)',
                  'Green Mottle (CGM)',
                  'Mosaic Disease (CMD)',
                  'Healthy']
        report = pd.DataFrame.from_dict(
            classification_report(y_true, y_pred,
                                  target_names=labels,
                                  output_dict=True))
        report.to_csv(self.metric_path, index=False)
