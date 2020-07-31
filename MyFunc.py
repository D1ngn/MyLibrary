import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 画像処理用
import cv2
from PIL import Image

# 音声処理用
import librosa
import soundfile as sf


"""
画像処理
"""




"""
音声処理
"""
#　音声データをロードし、指定された秒数とサンプリングレートでリサンプル
def load_audio_file(file_path, length, sampling_rate=16000):
    data, sr = librosa.load(file_path, sr=sampling_rate)
    # データが設定値よりも大きい場合は大きさを超えた分をカットする
    # データが設定値よりも小さい場合はデータの後ろを0でパディングする
    if len(data) > sampling_rate*length:
        data = data[:sampling_rate*length]
    else:
        data = np.pad(data, (0, max(0, sampling_rate*length - len(data))), "constant")
    return data

# 音声データを指定したサンプリングレートで保存
def save_audio_file(file_path, data, sampling_rate=16000):
    # librosa.output.write_wav(file_path, data, sampling_rate) # 正常に動作しないので変更
    sf.write(file_path, data, sampling_rate)

# 2つのオーディオデータを足し合わせる
def audio_mixer(data1, data2):
    assert len(data1) == len(data2)
    mixed_audio = data1 + data2
    return mixed_audio

# 音声データをスペクトログラムに変換する
def wave_to_spec(data, n_fft, hop_length, win_length):
    # 短時間フーリエ変換(STFT)を行い、スペクトログラムを取得
    spec = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(spec) # 振幅スペクトログラムを取得
    phase = np.exp(1.j * np.angle(spec)) # 位相スペクトログラムを取得(フェーザ表示)
    # mel_spec = librosa.feature.melspectrogram(data, sr=sr, n_mels=128) # メルスペクトログラムを用いる場合はこっちを使う
    return mag, phase

"""
ファイル管理用
"""




"""
図に各関数のグラフをプロット
"""
# 正弦波(sin関数)を描画
def sin(x):
    """
    plot sin wave
    """
    sin_x = np.sin(x)
    plt.plot(x, sin_x)
    plt.xlabel("x")
    plt.ylabel("sin_x")
    plt.show()

# 正規分布(ガウス分布を描画)
def gaussian(x, mean, std):
    """
    plot gaussian distibution
    """
    px = (1/np.sqrt(2*np.pi*std**2))*np.exp(-(x-mean)**2/(2*std**2))

    sns.set()
    plt.plot(x, px)
    plt.xlabel("x")
    plt.ylabel("px")
    plt.show()

if __name__ == "__main__":
    # 図のスタイルを変更
    sns.set()

    x = np.arange(-5, 5, 0.1)
    mean = 0
    std = 1

    gaussian(x, mean, std)

    # sin(x)
