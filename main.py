import os
import numpy as np
import librosa
from pytube import YouTube
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import subprocess
import glob

class GenreClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.encoder = LabelEncoder()

    def extract_features(self, file_path):
        y, sr = librosa.load(file_path, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)

    def load_dataset(self, dataset_path):
        genres = os.listdir(dataset_path)
        X, y = [], []
        for genre in genres:
            files = glob.glob(os.path.join(dataset_path, genre, '*.wav'))
            for file in files:
                features = self.extract_features(file)
                X.append(features)
                y.append(genre)
        X = np.array(X)
        y = self.encoder.fit_transform(y)
        return X, y

    def train(self, dataset_path):
        X, y = self.load_dataset(dataset_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        joblib.dump(self.model, 'genre_model.pkl')
        joblib.dump(self.encoder, 'genre_encoder.pkl')

    def predict(self, file_path):
        features = self.extract_features(file_path)
        model = joblib.load('genre_model.pkl')
        encoder = joblib.load('genre_encoder.pkl')
        prediction = model.predict([features])
        return encoder.inverse_transform(prediction)[0]

class YouTubeAudioDownloader:
    def download_audio(self, url):
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).first()
        stream.download(filename='downloaded_song.mp4')
        subprocess.call(['ffmpeg', '-y', '-i', 'downloaded_song.mp4', '-ar', '22050', '-ac', '1', 'downloaded_song.wav'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return 'downloaded_song.wav'

def main():
    url = input("Enter YouTube URL: ")
    downloader = YouTubeAudioDownloader()
    audio_path = downloader.download_audio(url)
    classifier = GenreClassifier()
    genre = classifier.predict(audio_path)
    print("Predicted Genre:", genre)

if __name__ == '__main__':
    main()
