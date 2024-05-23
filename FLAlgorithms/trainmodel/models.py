import torch
import torch.nn as nn
import torch.nn.functional as F
from FLAlgorithms.trainmodel.functions import ReverseLayerF



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()



        #################
        # shared encoder
        #################

        self.shared_encoder_fc = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(1024))
        #################
        # shared classify
        #################
        
        self.shared_encoder_pred_class = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 97),)

            # torch.nn.BatchNorm1d(97))


        ########################
        # shared domain classify
        ########################

        self.shared_encoder_pred_domain = torch.nn.Sequential(
            torch.nn.Linear(1024, 128),
            nn.ReLU(),
            # torch.nn.BatchNorm1d(1024),
            # torch.nn.Linear(1024, 128),
            # nn.ReLU(),
            # torch.nn.BatchNorm1d(4)
        )



    def forward(self, x, private_code, p=0.0):
        result = []


        
        shared_code = self.shared_encoder_fc(x)

        result.append(shared_code)


        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)
        # domain_label = F.log_softmax(domain_label)

        result.append(domain_label)


        class_label_shared = self.shared_encoder_pred_class(shared_code)


        result.append(class_label_shared)


        domain_private_label = self.shared_encoder_pred_domain(private_code)


        result.append(domain_private_label)



        return result


class Net_private(nn.Module):
    def __init__(self):
        super(Net_private, self).__init__()



        self.source_encoder_fc = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(1024))

        #################
        # private classify
        #################

        self.private_encoder_pred_class = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 97),)

            # torch.nn.BatchNorm1d(97))




    def forward(self, x):
        result2 = []

        private_code = self.source_encoder_fc(x)

        result2.append(private_code)

        class_label_private = self.private_encoder_pred_class(private_code)


        result2.append(class_label_private)


        return result2


class Net_digit(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #########################
        # private source encoder
        #########################


        self.source_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=3, padding=1),
            torch.nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=3, padding=1),

            torch.nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(128))
        self.source_encoder_fc = torch.nn.Sequential(
            torch.nn.Linear(8192, 3072),
            nn.ReLU(),
            torch.nn.BatchNorm1d(3072),
            torch.nn.Linear(3072, 768),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(768))

        #################
        # shared encoder
        #################

        self.shared_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=3, padding=1),
            torch.nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=3, padding=1),

            torch.nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(128))
        self.shared_encoder_fc = torch.nn.Sequential(
            torch.nn.Linear(8192, 3072),
            nn.ReLU(),
            torch.nn.BatchNorm1d(3072),
            torch.nn.Linear(3072, 768),
            nn.ReLU(),
            torch.nn.BatchNorm1d(768))

        #################
        # shared classify
        #################

        self.shared_encoder_pred_class = torch.nn.Sequential(
            torch.nn.Linear(768, 100),
            nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Linear(100, 10),
            nn.ReLU(),
            torch.nn.BatchNorm1d(10))

        #################
        # private classify
        #################

        self.private_encoder_pred_class = torch.nn.Sequential(
            torch.nn.Linear(768, 100),
            nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            torch.nn.Linear(100, 10),
            nn.ReLU(),
            torch.nn.BatchNorm1d(10))



        ########################
        # shared domain classify
        ########################

        self.shared_encoder_pred_domain = torch.nn.Sequential(
            torch.nn.Linear(768, 128),
            nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 5),
            nn.ReLU(),
            torch.nn.BatchNorm1d(5))

        ########################
        # private domain classify
        ########################

        self.private_encoder_pred_domain = torch.nn.Sequential(
            torch.nn.Linear(768, 128),
            nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 5),
            nn.ReLU(),
            torch.nn.BatchNorm1d(5)
            )


        #################
        # shared decoder
        #################

        self.shared_decoder_fc = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            nn.ReLU(),
            torch.nn.BatchNorm1d(768))


        self.shared_decoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(16),
            nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(16),
            nn.ReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            nn.ReLU(),
            torch.nn.Conv2d(16, 3, kernel_size=3, padding=1)
        )

    def forward(self, x, p=0.0):
        result = []
        private_featrure = self.source_encoder(x)
        private_featrure = private_featrure.view(-1, 128*8*8)
        private_code = self.source_encoder_fc(private_featrure)

        result.append(private_code)


        shared_featrue = self.shared_encoder(x)
        shared_featrue = shared_featrue.view(-1, 128*8*8)
        shared_code = self.shared_encoder_fc(shared_featrue)

        result.append(shared_code)

        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)
        domain_label = F.log_softmax(domain_label)

        result.append(domain_label)

        class_label_shared = self.shared_encoder_pred_class(shared_code)
        class_label_private = self.private_encoder_pred_class(shared_code)
        class_label = F.log_softmax(class_label_shared + class_label_private)
        # class_label = F.log_softmax(class_label_private)  # use for ablation study

        result.append(class_label)


        union_code = private_code + shared_code
        rec_vec = self.shared_decoder_fc(union_code)
        rec_vec = rec_vec.view(-1, 3, 16, 16)
        rec_code = self.shared_decoder(rec_vec)


        result.append(rec_code)



        # reversed_private_code = ReverseLayerF.apply(private_code, p)
        # domain_private_label = self.private_encoder_pred_domain(reversed_private_code)
        # domain_private_label = self.private_encoder_pred_domain(private_code)  # change to one domain net
        domain_private_label = self.shared_encoder_pred_domain(private_code)
        domain_private = F.log_softmax(domain_private_label)

        # result.append(domain_private_label)
        result.append(domain_private)


        return result
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, (12, 2))
#         self.conv2 = nn.Conv2d(64, 128, (12, 2))
#
#
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#
#         self.fc1 = nn.Linear(356352, 128)
#         self.fc2 = nn.Linear(128, 15)
#
#     def forward(self, x):
#
#         x = self.conv1(x)
#         x = nn.ReLU()(x)
#         x = nn.MaxPool2d(2, 1)(x)
#         x = self.dropout1(x)
#         x = self.conv2(x)
#         x = nn.ReLU()(x)
#         x = nn.MaxPool2d(2, 1)(x)
#         x = self.dropout2(x)
#
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = nn.ReLU()(x)
#         x = self.fc2(x)
#
#         output = F.log_softmax(x, dim=1)
#         return output
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 128, (12, 2)),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d((2, 1)),
#             torch.nn.Conv2d(128, 256, (12, 1)),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d((2, 1)),
#             torch.nn.Conv2d(256, 512, (12, 1)),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d((2, 1)),
#         )
#         self.dense = torch.nn.Sequential(
#             torch.nn.Linear(9600, 512),
#             nn.ReLU(inplace=True),
#             torch.nn.Linear(512, 15)
#         )
#     def forward(self, x):
#         x = self.conv1(x)
#         x = x.view(x.shape[0], -1)
#         x = self.dense(x)
#         output = F.log_softmax(x, dim=1)
#
#         return output

class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim = 784, output_dim = 10):
        super(Mclr_Logistic, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

class Mclr_CrossEntropy(nn.Module):
    def __init__(self, input_dim = 784, output_dim = 10):
        super(Mclr_CrossEntropy, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = self.linear(x)
        return outputs

class DNN(nn.Module):
    def __init__(self, input_dim = 784, mid_dim = 100, output_dim = 10):
        super(DNN, self).__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)
        
    def forward(self, x):
        # define forward pass
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x