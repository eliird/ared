import random
import time
import warnings
from matplotlib import pyplot as plt
import torch

from ared import EmotionDetector
from ared.utils import (
    load_first_50_images, load_audio_from_file, get_video_duration
)

warnings.filterwarnings('ignore')
random.seed(20)


def main():
    # paths containing the weights of the model
    vis_weights = './weights/vision/MELDSceneNet_best.pt'
    audio_weights = './weights/audio/model_best_sentiment.pth'
    text_wreights = './weights/text/'
    device = 'cuda:0'

    # load the emotion detection model
    detector = EmotionDetector(vis_model_weights=vis_weights,
                               text_model_weights=text_wreights,
                               audio_model_weights=audio_weights,
                               device=device)

    video_path = './dia0_utt0.mp4'
    images = load_first_50_images(video_path)
    audio_data = load_audio_from_file(video_path)
    sampling_rate = 44100

    utterance = "also i was the point person on my company's transition from the kl five to gr six systems."

    prediction_time = []
    audio_duration = []
    
    with torch.no_grad():
        # detect the emotion of the video
        for i in range(1, 15):
            audio = audio_data[:, :int(i*16000)]
            print(audio.shape)
            st = time.time()
            emotion, probab = detector.detect_emotion(video=images, audio=audio, text=utterance)
            et = time.time()
            
            print("Prediction Time: ", et - st)
            print("Audio Duration : ", audio.shape[1]/sampling_rate)
            print(1, emotion)
            print('*' * 20)
            prediction_time.append(et - st)
            audio_duration.append(audio.shape[1]/sampling_rate)
    plt.plot(audio_duration, prediction_time, 'o')
    plt.title('Audio Duration vs Prediction Time')
    plt.xlabel('Audio Duration')
    plt.ylabel('Prediction Time')
    plt.savefig('AudioDuration_vs_PredictionTime.png')
    
    
if __name__ == '__main__':
    main()