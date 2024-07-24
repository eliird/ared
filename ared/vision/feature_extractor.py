import torch
from facenet_pytorch import InceptionResnetV1

class VisionFeatureExtractor(torch.nn.Module):
    """
    Defines a PyTorch module for extracting visual features from a sequence of images using an InceptionResnetV1 model and an LSTM.
    
    The `VisionFeatureExtractor` class takes a sequence of images as input, passes them through the InceptionResnetV1 model to extract visual features, and then processes the sequence of features using an LSTM. The final output is a classification of the sequence of images.
    
    Args:
        num_imgs (int): The number of images in the input sequence.
    
    Returns:
        torch.Tensor: A tensor of shape (batch_size, 7) containing the classification probabilities for each of the 7 classes.
    """
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
    