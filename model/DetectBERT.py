import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class DetectBERT(nn.Module):
    def __init__(self, cfg, n_classes, input_size=128, hidden_size=128):
        super(DetectBERT, self).__init__()
        self.cfg = cfg
        self._fc1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=hidden_size)
        self.layer2 = TransLayer(dim=hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self._fc2 = nn.Linear(hidden_size, self.n_classes)


    def forward(self, **kwargs):

        h = kwargs['data'].float() #[B, n, 1024]
        
        h = self._fc1(h) #[B, n, 512]
        
        if self.cfg.Model.aggregation == "DetectBERT":
            #---->cls_token
            B = h.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
            h = torch.cat((cls_tokens, h), dim=1)

            #---->Translayer x1
            h = self.layer1(h) #[B, N, 512]

            #---->Translayer x2
            h = self.layer2(h) #[B, N, 512]

            #---->cls_token
            h = self.norm(h)[:,0]
        elif self.cfg.Model.aggregation == "addition":
            h = h.sum(dim=1)
        elif self.cfg.Model.aggregation == "average":
            h = h.mean(dim=1)
        elif self.cfg.Model.aggregation == "random":
            random_index = torch.randint(0, h.size(1), (1,))
            h = h[:, random_index.item(), :]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

if __name__ == "__main__":
    from utils import read_yaml
    cfg = read_yaml('./config.yaml')
    data = torch.randn((1, 1000, 128)).cuda()
    model = DetectBERT(cfg, n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)