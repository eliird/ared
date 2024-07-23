import torch
from emotionnet import EmotionNet



class MMER(torch.nn.Module):
    def __init__(self, hyper_params) -> None:
        super(MMER, self).__init__()
        self.base = EmotionNet(hyper_params)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, y, z):
        out = self.base(x, y, z)
        out = self.softmax(out)
        # out = self.softmax(out.reshape(x.size(0)*6, 5))
        return out