import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
from FLAlgorithms.trainmodel.functions import SIMSE, DiffLoss, MSE
from torch.autograd import Variable
from FLAlgorithms.trainmodel.marginloss import ArcMarginProduct
from FLAlgorithms.trainmodel.models import Net_private
import torch.nn.functional as F

import numpy as np
class UserFDARN(User):
    def __init__(self, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                 local_epochs, optimizer):
        super().__init__(numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs)


        self.loss = nn.CrossEntropyLoss()
        self.gpu_id = 0
        self.Net_private = Net_private().cuda(self.gpu_id)
        self.arcface_loss = ArcMarginProduct(128, 4).cuda(self.gpu_id)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-6)
        self.optimizer2 = torch.optim.SGD(self.Net_private.parameters(), lr=self.learning_rate)
        self.step_decay_weight = 0.95
        self.lr_decay_step = 2000
        self.active_domain_loss_step = 200
        self.beta_weight = 0.6
        self.gamma_weight = 0.4
        self.arcface_weight = 0.4
        self.spread = 0.4
        self.loss_classification = torch.nn.CrossEntropyLoss().cuda(self.gpu_id)
        self.loss_recon1 = MSE().cuda(self.gpu_id)
        self.loss_recon2 = SIMSE().cuda(self.gpu_id)
        self.loss_diff = DiffLoss().cuda(self.gpu_id)
        self.loss_similarity = torch.nn.CrossEntropyLoss().cuda(self.gpu_id)
        self.loss_private = torch.nn.CrossEntropyLoss().cuda(self.gpu_id)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data

        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]



    def exp_lr_scheduler(self, optimizer, step):

        # Decay learning rate by a factor of step_decay_weight every lr_decay_step
        current_lr = self.learning_rate * (self.step_decay_weight ** (step / self.lr_decay_step))

        # if step % self.lr_decay_step == 0:
            # print('learning rate is set to %f' % current_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        return optimizer


    def spread_out_loss(self, embeddings):

        # embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_class = embeddings.shape[0]
        gram_matrix = torch.mul(embeddings, embeddings)
        gram_matrix = torch.diag(gram_matrix)
        a_diag = torch.diag_embed(gram_matrix)
        gram_matrix = gram_matrix - a_diag
        loss = torch.sum(gram_matrix.pow(2)) / (embeddings_class * (embeddings_class - 1))
        return loss


    def train(self, epochs):
        LOSS = 0
        self.model.train()
        len_dataloader = self.len_dataloader
        dann_epoch = np.floor(self.active_domain_loss_step / len_dataloader * 1.0)

        current_step = 0
        for epoch in range(1, self.local_epochs + 1):
            i = 0
            while i < self.len_dataloader:

                X, y = self.get_next_train_batch()
                self.model.zero_grad()
                self.Net_private.zero_grad()
                self.optimizer.zero_grad()
                self.optimizer2.zero_grad()
                loss = 0
                n = int(self.id) - 1
                # if n < 5:
                #     domain_label = torch.ones(self.batch_size) * 0
                # elif 4 < n < 10:
                #     domain_label = torch.ones(self.batch_size) * 1
                # elif 7 < n < 12:
                #     domain_label = torch.ones(self.batch_size) * 2
                # else:
                #     domain_label = torch.ones(self.batch_size) * 3
                domain_label = torch.ones(self.batch_size)*n
                domain_label = domain_label.long().cuda(self.gpu_id)
                domain_label = Variable(domain_label)

                result1 = self.Net_private(X)
                private_code, class_label_private = result1

                if current_step > self.active_domain_loss_step:
                    p = float(i + (epoch - dann_epoch) * len_dataloader / (self.local_epochs - dann_epoch) / len_dataloader)
                    p = 2. / (1. + np.exp(-10 * p)) - 1
                    result = self.model(X, private_code, p=p)
                    share_code, domainv_label, class_label_shared, private_domain= result
                    if domainv_label.shape[0] == self.batch_size:

                        source_arcface = self.arcface_loss(domainv_label, domain_label)
                        source_dann = self.gamma_weight * self.loss_similarity(source_arcface, domain_label)
                        # source_dann = self.gamma_weight * self.loss_similarity(domainv_label, domain_label)
                        loss += source_dann

                else:
                    result = self.model(X, private_code)
                    share_code, _, class_label_shared, private_domain = result

                class_label =  F.log_softmax(class_label_shared + class_label_private)
                # class_label = F.log_softmax(class_label_private)
                label_classification = self.loss_classification(class_label, y)
                loss += label_classification

                diff = self.beta_weight * self.loss_diff(private_code, share_code)
                loss += diff

                if  private_domain.shape[0] == self.batch_size:

                    private_arcface = self.arcface_loss(private_domain, domain_label)
                    private_domain = self.arcface_weight * self.loss_private(private_arcface, domain_label)
                    # private_domain = self.arcface_weight * self.loss_private(private_domain, domain_label)
                    loss += private_domain

                embeddings = self.arcface_loss.state_dict()['weight']
                spread = self.spread * self.spread_out_loss(embeddings)
                loss += spread


                loss.backward()
                optimizer = self.exp_lr_scheduler(optimizer=self.optimizer, step=current_step)
                self.optimizer2.step()
                optimizer.step()

                i += 1
                current_step += 1
                self.clone_model_paramenter(self.model.parameters(), self.local_model)


        return LOSS
