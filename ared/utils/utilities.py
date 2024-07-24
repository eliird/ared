from PIL import Image
import cv2
import librosa as lb
import random


emotion2id = {
        'neutral': 0, 
        'surprise': 1, 
        'fear': 2, 
        'sadness': 3, 
        'joy': 4, 
        'disgust': 5,
        'anger': 6
    }

id2emotion = {item:key for key, item in emotion2id.items()}


class NameSpace(dict):
    """Converts Dictionary into namespace so the elements can be accessed using . operator
    Example:
        m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(NameSpace, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(NameSpace, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(NameSpace, self).__delitem__(key)
        del self.__dict__[key]
        
        
def load_first_50_images(path: str) -> list:
    """

    Parameters
    ----------
    path : path of the video

    Returns
    list[PIL.Image] 

    
    """
    images = []
    count = 0
    cap = cv2.VideoCapture(path)
    while count < 50:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(frame))
        
        count += 1
    return images

def load_random_50_images(path: str) -> list:
    """
    Load 50 random images from a video file.
    
    Parameters
    ----------
    path : str
        The path to the video file.
    
    Returns
    -------
    list[PIL.Image]
        A list of 50 random images from the video file.
    """
    images = []
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(frame))
    if len(images) < 50:
        for _ in range(50 - len(images)):
            images.append(images[-1])
    else:
        images = random.sample(images, 50)
    return images

def load_audio_from_file(path: str, sr: int=44100):
    """
    Load audio from a file.
    
    Parameters
    ----------
    path : str
        The path to the audio file.
    sr : int, optional
        The sample rate of the audio file, defaults to 44100.
    
    Returns
    -------
    numpy.ndarray
        The audio data as a 1D or 2D numpy array, depending on whether the audio is mono or stereo.
    """
    y, _ = lb.load(path, sr=sr, mono=False)
    return y