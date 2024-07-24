import torch
from torchvision.transforms import transforms
from .feature_extractor import VisionFeatureExtractor
from PIL import Image
import cv2
import random

class VidePreprocessor:
    '''
    When passing the weights to extract features make sure to choose the appropriate scene or face model
    '''
    # image transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocessing = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    
    def __init__(self,   vision_extractor_weights: str='./weights/MELDSceneNet_best.pth', device: str='cuda', max_num_imgs=50,):
        self.transform = self.preprocessing
        self.max_num_imgs = max_num_imgs
        self.device = device
        self.vision_feature_extractor = VisionFeatureExtractor()
        self.vision_feature_extractor.to(device)
        self.vision_feature_extractor.load_state_dict(torch.load(vision_extractor_weights))
        self.vision_features = {}

        def get_activation(name):
            def hook(model, input, output):
                self.vision_features[name] = output[0].detach()
            return hook

        self.vision_feature_extractor.lstm.register_forward_hook(get_activation('lstm'))
            
    
    def process_video(self, path):
        cap = cv2.VideoCapture(path)
        images = []
        while(True):
            ret, frame = cap.read()
            if not ret:
                if len(images) >= self.max_num_imgs:
                    break
                # replicate the last image
                images.append(images[-1].copy())
            # TODO check for the RGB orientation
            images.append(Image.fromarray(frame))
        images = random.sample(images, self.max_num_imgs)
        return self.process_images(images)
            
    def process_images(self, images: list):
        '''
        images is a list containing PIL image of the cropped face
        '''
        assert len(images) <= self.max_num_imgs
        
        num_images = len(images)
        vid_tensor = torch.zeros((1, num_images, 3, 224, 224))
        
        for i in range(num_images):
            vid_tensor[0][i] = self.transform(images[i])
        
        with torch.no_grad():
            self.vision_feature_extractor(vid_tensor.to(self.device))
        
        return self.vision_features['lstm'].squeeze()
         