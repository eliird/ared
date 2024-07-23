import torch
from facenet_pytorch import InceptionResnetV1

class VisionFeatureExtractor(torch.nn.Module):
    def __init__(self, num_imgs=50) -> None:
        super(VisionFeatureExtractor, self).__init__()
        self.base = InceptionResnetV1(pretrained='vggface2')
        self.lstm = torch.nn.LSTM(512, 256, batch_first=True)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256, 7),
            )

        self.num_imgs = num_imgs

    def forward(self, imgs):
        imgs = imgs.view(imgs.size(0) * self.num_imgs, 3, 224, 224)
        imgs = self.base(imgs)
        imgs = imgs.view(int(imgs.size(0)/self.num_imgs), self.num_imgs, 512)
        imgs, _ = self.lstm(imgs)
        imgs = self.classifier(imgs)
        return torch.softmax(imgs[:, -1, :], -1)
    