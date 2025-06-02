'''
model for csf-mil
'''

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# net for single scale
class Attention_fea(nn.Module):
    def __init__(self):
        super(Attention_fea, self).__init__()
        self.L = 512 # resnet:1024; ctranspath:768; retccl:2048; conch:512
        self.D = 512
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x):
        H = x.squeeze(0)

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N,

        M = torch.mm(A, H)  # KxL

        return M

# siamese net
class Siamese_Net(nn.Module):
    def __init__(self):
        super(Siamese_Net, self).__init__()
        self.L = 512 # resnet:1024; ctranspath:768; retccl:2048; conch:512
        self.D = 64
        self.K = 1

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        feature_small = self.feature_extractor_part2(x)
        return feature_small

# net for cross scale
class Attention_fea_2(nn.Module):
    def __init__(self):
        super(Attention_fea_2, self).__init__()
        self.L = 64
        self.D = 64
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x):
        H = x.squeeze(0)

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        return M

# classifier
class Classifier_head(nn.Module):
    def __init__(self):
        super(Classifier_head, self).__init__()
        self.L = 64*4
        self.D = 64
        self.K = 1

        self.classifier = nn.Sequential(
            nn.Linear(self.L , self.D),
            nn.Sigmoid(),
            nn.Linear(self.D, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        Y_prob = self.classifier(x)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat
