import torch.nn as nn
from transformer import Transformer

class LanguageModel(nn.Module):
    """ one head of self-attention """

    def __init__(self, vocab_size, n_embd, block_size,n_layer, n_head, device):
        super().__init__()
        self.transformer = Transformer(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size, n_layer=n_layer, n_head=n_head, device=device, isDecoder = True)
        
    def forward(self, x, target = None):
        out, map = self.transformer(x)
        loss = 0
        if(target is not None):
            B, T, C = out.shape
            out = out.view(B*T, C)
            target = target.view(B*T)
            loss = nn.functional.cross_entropy(out, target)
        return loss, map
