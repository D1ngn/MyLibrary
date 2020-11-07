import os
import pyaudio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import seaborn as sns
import wave
from scipy.stats import kurtosis


# def wave_plot(input_path, output_path, fig_title=None):
#     # open wave file
#     wf = wave.open(input_path,'r')
#
#     # load wave data
#     length = 3 # 読み出すオーディオの長さ[s]
#     rate = wf.getframerate()  # サンプリングレート[1/s]
#     chunk_size = rate * length
#     amp  = (2**8) ** wf.getsampwidth() / 2
#     data = wf.readframes(chunk_size)   # バイナリ読み込み
#     data = np.frombuffer(data,'int16') # intに変換
#     data = data / amp                  # 振幅正規化
#
#     # make time axis
#     size = float(chunk_size)  # 波形サイズ
#     x = np.arange(0, size/rate, 1.0/rate)
#
#     # 図に描画
#     # sns.set() # スタイルをきれいにする
#     fig = plt.figure(facecolor='w', linewidth=5, edgecolor='black')
#     # ax = fig.add_subplot(1, 1, 1, ylim=(-0.5, 0.5)) # 図を1行目1列の1番目に表示(図を1つしか表示しない場合)
#     ax = fig.add_subplot(1, 1, 1, title=fig_title) # 図を1行目1列の1番目に表示(図を1つしか表示しない場合)
#     ax.set_xlabel('time[s]') # x軸名を設定
#     ax.set_ylabel('magnitude') # y軸名を設定
#     ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0)) # x軸の主目盛を1.0ごとに表示
#     ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.10)) # y軸の主目盛を0.10ごとに表示
#     # p_file = pathlib.Path(output_path)
#     # file_name = p_file.name.split('.')[0] # データの名前を設定
#     file_name = os.path.basename(output_path).split('.')[0] # データの名前を設定
#     ax.plot(x, data, label='{}'.format(file_name)) # データをプロット
#     ax.legend(edgecolor="black") # 凡例を追加
#     fig.savefig(output_path) # グラフを保存

def wave_plot(data, output_path, length, rate, fig_title=None):
    # open wave file
    # wf = wave.open(input_path,'r')

    # load wave data
    # length = 3 # 読み出すオーディオの長さ[s]
    # rate = wf.getframerate()  # サンプリングレート[1/s]
    chunk_size = rate * length
    # amp  = (2**8) ** wf.getsampwidth() / 2
    amp = data.max()
    # data = wf.readframes(chunk_size)   # バイナリ読み込み
    # data = np.frombuffer(data,'int16') # intに変換
    data = data / amp                  # 振幅正規化

    # make time axis
    size = float(chunk_size)  # 波形サイズ
    x = np.arange(0, size/rate, 1.0/rate)

    # 足りない長さはpadding
    data = np.pad(data, [0,x.shape[0]-data.shape[0]], 'constant')

    # 図に描画
    # sns.set() # スタイルをきれいにする
    fig = plt.figure(facecolor='w', linewidth=5, edgecolor='black')
    # ax = fig.add_subplot(1, 1, 1, ylim=(-0.5, 0.5)) # 図を1行目1列の1番目に表示(図を1つしか表示しない場合)
    ax = fig.add_subplot(1, 1, 1, title=fig_title) # 図を1行目1列の1番目に表示(図を1つしか表示しない場合)
    ax.set_xlabel('time[s]') # x軸名を設定
    ax.set_ylabel('magnitude') # y軸名を設定
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0)) # x軸の主目盛を1.0ごとに表示
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.10)) # y軸の主目盛を0.10ごとに表示
    # p_file = pathlib.Path(output_path)
    # file_name = p_file.name.split('.')[0] # データの名前を設定
    file_name = os.path.basename(output_path).split('.')[0] # データの名前を設定
    ax.plot(x, data, label='{}'.format(file_name)) # データをプロット
    ax.legend(edgecolor="black") # 凡例を追加
    fig.savefig(output_path) # グラフを保存




# 録音用の関数を定義
def recording():
    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    return frames


def main():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    # RATE = 44100
    RATE = 24000
    CHUNK = 1024
    RECORD_SECONDS = 5
    AUDIO_LENGTH = 5

    sns.set() # スタイルをきれいにする

    # STEP1
    # ２つの独立音源 s1 , s2 を用意する。
    # 録音するか、または録音済みの音声ファイルを読み込んで、２つの独立音源 s1 , s2 の標本配列を得る。
    first_sample_path = "../data/BASIC5000_0025.wav"
    save_path = "output/first_sample.png"
    rate, s1 = wav.read(first_sample_path)
    wave_plot(s1, save_path, AUDIO_LENGTH, rate)

    second_sample_path = "../data/1-137-A-32.wav"
    save_path = "output/second_sample.png"
    rate, s2 = wav.read(second_sample_path)
    wave_plot(s2, save_path, AUDIO_LENGTH, rate)

    # 2つの音声の長さが異なる場合小さい方に合わせる
    if len(s1) < len(s2):
        s2 = s2[ : len(s1)]
    elif len(s1) > len(s2):
        s1 = s1[ : len(s2)]

    # 次に、 s1 , s2 の独立音源同士の独立性を確認するため、２つの独立音源の標本同士の散布図を描いてみる。
    # 独立した信号同士、かつそれぞれの信号は 0 を中心とした分布 (音声なので)　なので、
    # 散布図は概ね左右上下対称となることが想定される。
    plt.figure()
    plt.scatter(s1, s2, s=1, marker='x', alpha=0.2)
    plt.xlabel('first sample')
    plt.ylabel('second sample')
    plt.title(None)
    plt.savefig("output/scatter_plot.png")

    # STEP2
    # s1 , s2 の音量をランダムな強さ R によって加法合成した２つの合成音源 x1 , x2 を得る。
    # Randomly mix
    # R = np.random.rand(4).reshape(2, 2)
    #
    # np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    # print('合成する各音源の音量倍率 R:\n{}'.format(R))
    #
    # x1, x2 = np.dot(R, (s1, s2))
    #
    # plt.figure()
    # plt.xlim(len(x1))
    # plt.title('time series of x1, randomly mixed (s1: {0:.2f}, s2: {1:.2f})'.format(R[0,0], R[0,1]))
    # plt.plot(x1)
    #
    #
    # plt.figure()
    # plt.title('time series of x2. randomly mixed (s1: {0:.2f}, s2: {1:.2f})'.format(R[1,0], R[1,1]))
    # plt.xlim(len(x2))
    # plt.plot(x2)









if __name__ == "__main__":
    main()
