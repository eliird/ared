import os
import time
import cv2
from datetime import datetime


def get_video_duration(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist")
    
    data = cv2.VideoCapture(path)
    # count the number of frames 
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT) 
    fps = data.get(cv2.CAP_PROP_FPS) 
    
    # calculate duration of the video 
    seconds = round(frames / fps) 
    # video_time = datetime.timedelta(seconds=seconds) 
    # print(f"duration in seconds: {seconds}") 
    # print(f"video time: {video_time}") 
    return seconds


def timeit(func):
    """

    Parameters
    ----------
    func :
        A function that needs to be timed

    Returns
        A wrapped function that princts the time it took for the execution of that function
    """
    def wrapper(*args, **kwargs):
        """

        Parameters
        ----------
        *args : parameters to the function that needs to be wrapped
            
        **kwargs : keyword arguments of the function that needs to be wrapped
            

        Returns
         the time decorator for the original function
        
        """
        start = time.time()
        result = func(*args, **kwargs)
        print(f'time taken by {func.__name__} is {time.time() - start}')
        return result
    return wrapper