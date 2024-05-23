import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
import torchvision.utils as vutils
import torch.utils.data
from FLAlgorithms.trainmodel.models import Net_private
import torch.nn.functional as F

class User:

    def __init__(self, id, train_data, test_data, model, batch_size=0, learning_rate=0, beta=0, lamda=0,
                 local_epochs=0):
        self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.lamda = lamda
        self.local_epochs = local_epochs
        self.trainloader = DataLoader(train_data, self.batch_size, shuffle=True)
        self.testloader = DataLoader(test_data, self.batch_size, shuffle=True)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        self.gpu_id = 0
        # those parameters are for persionalized federated learing.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        # self.local_model = copy.deepcopy(self.model)
        self.persionalized_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        self.len_dataloader = len(self.trainloader)
        self.Net_private = Net_private().cuda(self.gpu_id)
    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
        # for key in model.state_dict().keys():
        #     if 'source_encoder_fc' not in key:
        #         self.model.state_dict()[key].data.copy_(model.state_dict()[key])
        #         self.local_model.state_dict()[key].data.copy_(model.state_dict()[key])


    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def get_updated_parameters(self):
        return self.local_weight_updated

    def update_parameters(self, new_params):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads




    def test(self):
        self.model.eval()
        test_acc = 0
        y_pred = 0
        len_test = len(self.testloader)
        i = 0
        # for x, y in self.testloaderfull:
        for x, y in self.testloader:
            x = x.cuda(self.gpu_id)
            y = y.cuda(self.gpu_id)
            y = y.reshape(-1)
            result1 = self.Net_private(x)
            result = self.model(x, result1[0])
            class_label = F.log_softmax(result[2] + result1[1])
            test_acc = test_acc + (torch.sum(torch.argmax(class_label, dim=1) == y)).item()
            i += 1
        print(self.id, test_acc)
        return test_acc, self.test_samples

    def train_error_and_loss(self):
        self.model.eval()
        train_acc = 0
        loss = 0


        for x, y in self.trainloader:
            x = x.cuda(self.gpu_id)
            y = y.cuda(self.gpu_id)
            result1 = self.Net_private(x)
            result = self.model(x, result1[0])
            class_label = F.log_softmax(result[2] + result1[1])

            train_acc += (torch.sum(torch.argmax(class_label, dim=1) == y)).item()

            loss = self.loss(class_label, y)
            loss += (torch.sum(loss)).item()


        return train_acc, loss, self.train_samples

    def test_persionalized_model(self):
        self.model.eval()
        test_acc = 0
        self.update_parameters(self.persionalized_model_bar)


        for x, y in self.testloader:
            x = x.cuda(self.gpu_id)
            y = y.cuda(self.gpu_id)
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

        self.update_parameters(self.local_model)
        return test_acc, y.shape[0]

    def train_error_and_loss_persionalized_model(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        self.update_parameters(self.persionalized_model_bar)
        # for x, y in self.trainloaderfull:
        self.trainloader = self.trainloader.cuda(self.gpu_id)
        for x, y in self.trainloader:
            x = x.cuda(self.gpu_id)
            y = y.cuda(self.gpu_id)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # print(self.id + ", Train Accuracy:", train_acc)
            # print(self.id + ", Train Loss:", loss)
        self.update_parameters(self.local_model)
        return train_acc, loss, self.train_samples

    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)  # 指针指向下一条记录
            X = X.cuda(self.gpu_id)
            y = y.cuda(self.gpu_id)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
            X = X.cuda(self.gpu_id)
            y = y.cuda(self.gpu_id)
        return (X, y)

    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
            # print(X.shape)
            # print(y)
            X = X.cuda(self.gpu_id)
            y = y.cuda(self.gpu_id)

        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
            X = X.cuda(self.gpu_id)
            y = y.cuda(self.gpu_id)
        return (X, y)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))

    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))  # 存在返回true
