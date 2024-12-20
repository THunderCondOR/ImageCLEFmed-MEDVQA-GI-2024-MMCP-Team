import torch.nn as nn
import torch 
from monai.networks.layers.utils import get_act_layer

class LabelEmbedder(nn.Module):
    def __init__(self, text_encoder, emb_dim=32, num_classes=2, input_dim=77, hidden_dim=128, act_name=("SWISH", {})):
        super().__init__()
        self.text_encoder = text_encoder
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        # self.embedding = nn.Embedding(num_classes, hidden_dim)

        # self.embedding = nn.Embedding(num_classes, emb_dim//4)
        #self.linear1 = nn.Linear(hidden_dim, emb_dim)
        # self.emb_layer_norm = nn.LayerNorm(emb_dim)
        #self.linear2 = nn.Linear(emb_dim, emb_dim)
        #self.act = get_act_layer(act_name)

    def forward(self, condition):
        self.text_encoder.requires_grad_(False)
        with torch.no_grad():
            c = self.text_encoder(condition, return_dict=False)[0]
        #c = self.linear1(c)
        #c = self.emb_layer_norm(c)
        #c = self.act(c)
        #c = self.linear2(c)
        #c = self.embedding(condition) #[B,] -> [B, C]
        #c = self.emb_net(condition.to(torch.float32))
        # c = self.emb_net(condition[:,None].float())
        # c = (2*condition-1)[:, None].expand(-1, self.emb_dim).type(torch.float32)
        return c
    
class TextEmbedder(nn.Module):
    def __init__(self, text_encoder, emb_dim=32):
        super().__init__()
        self.text_encoder = text_encoder
        self.emb_dim = emb_dim

    def forward(self, condition):
        c = self.text_encoder(condition, return_dict=False)[0]
        return c