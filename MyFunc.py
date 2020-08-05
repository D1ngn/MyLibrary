import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 画像処理用
import cv2
from PIL import Image

# 音声処理用
import librosa
import soundfile as sf

# 機械学習用
# モデル
from sklearn.linear_model import LinearRegression
# 学習用
from sklearn.model_selection import train_test_split
#評価用
from sklearn.metrics import accuracy_score




"""
画像処理
"""




"""
音声処理
"""
#　音声データをロードし、指定された秒数とサンプリングレートでリサンプル
def load_audio_file(file_path, length, num_channels, sampling_rate=16000):
    data, sr = sf.read(file_path)
    # データが設定値よりも大きい場合は大きさを超えた分をカットする
    # データが設定値よりも小さい場合はデータの後ろを0でパディングする
    # 1ch(モノラル)の場合
    if num_channels == 1:
        if len(data) > sampling_rate*length:
            data = data[:sampling_rate*length]
        else:
            data = np.pad(data, (0, max(0, sampling_rate*length - len(data))), "constant")
    # マルチチャンネルの場合
    elif num_channels > 1:
        if data.shape[0] > sampling_rate*length:
            data = data[:sampling_rate*length, :]
        else:
            data = np.pad(data, [(0, max(0, sampling_rate*length-data.shape[0])), (0, 0)], "constant")
    else:
        print("please designate correct num_channels")
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
機械学習手法
"""
# 線形回帰
def linear_regression(X_train, X_test, y_train, y_test):
    """
    args:
        X_train, X_test: Features matrix (shape: [n_samples, n_features])
        y_train, y_test: Target array (shape: [n_samples,])

    return:
        y_model: predicted result of model
    """
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train) # モデルの学習
    y_model = model.predict(X_test) #　モデルの推論
    plt.scatter(X_train, y_train) # 学習データをプロット
    plt.plot(X_test, y_model) # テスト結果を直線でつないでプロット
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    return y_model


# 分類



# 次元削減




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

    # ガウス分布用
    # x = np.arange(-5, 5, 0.1)
    # mean = 0
    # std = 1
    #
    # gaussian(x, mean, std)

    #　三角関数用
    # x = np.arange(-5, 5, 0.1)
    # sin(x)

    # 線形回帰用
    x = 10 * np.random.rand(50)
    X = x[:, np.newaxis] # shape: [n_samples,] → [n_samples, n_features]の形式へ
    y = 3 * x + np.random.randn(50)
    X1, X2, y1, y2 = train_test_split(X, y, random_state=0, test_size=0.33) # データをtrain用とtest用に2:1で分ける
    output = linear_regression(X1, X2, y1, y2)
