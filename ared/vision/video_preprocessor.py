import torch
from torchvision.transforms import transforms
from .feature_extractor import VisionFeatureExtractor
from PIL import Image
import cv2
import random
from typing import Union, List
from ared.utils import load_first_50_images, load_random_50_images

class VidePreprocessor:
    """
    Preprocesses a video by extracting visual features using a pre-trained vision feature extractor.
    
    The `VidePreprocessor` class provides methods to process a video or a list of images and extract visual features using a pre-trained vision 
    feature extractor. The class supports configuring the device (CPU or GPU) and the maximum number of images to process from a video.
    
    The `process_video` method reads frames from a video file, preprocesses the frames using the `transform` method, 
    and extracts visual features using the `vision_feature_extractor`. The `process_images` method takes a list of PIL images, preprocesses them, 
    and extracts visual features.
    
    The extracted visual features are stored in the `vision_features` dictionary, with the key 'lstm' containing the final LSTM layer output.
    """
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
        """
        This is the constructor for the `VidePreprocessor` class, which is responsible for initializing the video preprocessing pipeline. 
        It takes the following parameters:
            
            - `vision_extractor_weights`: The path to the pre-trained weights for the vision feature extractor model.
            - `device`: The device to use for the vision feature extractor, either 'cuda' for GPU or 'cpu' for CPU.
            - `max_num_imgs`: The maximum number of images to process from a video.
            
            The constructor sets up the image transformation pipeline, loads the pre-trained vision feature extractor model, 
            and registers a hook to capture the final LSTM layer output, which is stored in the `vision_features` dictionary.
        """
        self.transform = self.preprocessing
        self.max_num_imgs = max_num_imgs
        self.device = device
        self.vision_feature_extractor = VisionFeatureExtractor()
        self.vision_feature_extractor.to(device)
        self.vision_feature_extractor.load_state_dict(torch.load(vision_extractor_weights, map_location=self.device))
        self.vision_features = {}

        def get_activation(name):
            def hook(model, input, output):
                self.vision_features[name] = output[0].detach()
            return hook

        self.vision_feature_extractor.lstm.register_forward_hook(get_activation('lstm'))
    
    def process_video(self, video: Union[str, list[Image.Image]]) -> torch.Tensor:
        """
        Processes a video or a list of images and extracts visual features using a pre-trained vision feature extractor.
        
        Args:
            video (Union[str, list[Image.Image]]): Either a path to a video file or a list of PIL Image objects.
        
        Returns:
            torch.Tensor: The final LSTM layer output of the vision feature extractor, representing the extracted visual features.
        """   
        if isinstance(video, str):
            images = load_random_50_images(video)
        else:
            images = video    
        return self._process_images(images)
            
    def _process_images(self, images: list):
        """
        Processes a list of images and extracts visual features using a pre-trained vision feature extractor.
        
        Args:
            images (list): A list of PIL Image objects.
        
        Returns:
            torch.Tensor: The final LSTM layer output of the vision feature extractor, representing the extracted visual features.
        """

        assert len(images) <= self.max_num_imgs
        
        num_images = len(images)
        vid_tensor = torch.zeros((1, num_images, 3, 224, 224))
        
        for i in range(num_images):
            vid_tensor[0][i] = self.transform(images[i])
        
        with torch.no_grad():
            self.vision_feature_extractor(vid_tensor.to(self.device))
        
        return self.vision_features['lstm'].squeeze()
         