import os

# 画像処理用
import cv2
from PIL import Image

# 音声処理用
import librosa
import soundfile as sf
import wave
from scipy import signal

# 機械学習用
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d # 3次元描画用


"""
画像処理
"""




"""
音声処理
"""
#　音声データをロードし、指定された秒数とサンプリングレートでリサンプル
def load_audio_file(file_path, length, sampling_rate=16000):
    data, sr = sf.read(file_path)
    # データが設定値よりも大きい場合は大きさを超えた分をカットする
    # データが設定値よりも小さい場合はデータの後ろを0でパディングする
    # シングルチャンネル(モノラル)の場合 (data.shape: [num_samples,])
    if data.ndim == 1:
        if len(data) > sampling_rate*length:
            data = data[:sampling_rate*length]
        else:
            data = np.pad(data, (0, max(0, sampling_rate*length - len(data))), "constant")
    # マルチチャンネルの場合 (data.shape: [num_samples, num_channels])
    elif data.ndim == 2:
        if data.shape[0] > sampling_rate*length:
            data = data[:sampling_rate*length, :]
        else:
            data = np.pad(data, [(0, max(0, sampling_rate*length-data.shape[0])), (0, 0)], "constant")
    else:
        print("number of audio channels are incorrect")
    return data

# 音声データを指定したサンプリングレートで保存
def save_audio_file(file_path, data, sampling_rate=16000):
    # librosa.output.write_wav(file_path, data, sampling_rate) # 正常に動作しないので変更
    sf.write(file_path, data, sampling_rate)

# 音声ファイルを再生
def play_audio(data, sampling_rate):
    # dataを再生する
    sd.play(data, sampling_rate)
    print("start")
    # 再生が終わるまで待つ
    status = sd.wait()
    print("finish")

# 音声を録音
def rec_audio(audio_length, sampling_rate, channels, save_path):
    print("start recording...")
    data = sd.rec(int(audio_length*sampling_rate), sampling_rate, channels=channels)
    # 録音が終わるまで待つ
    sd.wait()
    print("finish recording!")
    save_audio_file(save_path, data, sampling_rate)

# 2つの音声データを足し合わせる
def audio_mixer(data1, data2):
    assert len(data1) == len(data2)
    mixed_audio = data1 + data2
    return mixed_audio

# 片方の音声からもう一方の音声を引く
def audio_subtracter(data1, data2):
    assert len(data1) == len(data2)
    subtracted_audio = data1 - data2
    return subtracted_audio

# 音声データを指定したサンプリング周波数でリサンプルする
def audio_resampler(input_data, input_sr, output_sr):
    output_len = int(len(input_data) * (output_sr / input_sr))
    resampled_data = signal.resample(input_data, output_len)
    return resampled_data


# 音声データをスペクトログラムに変換する
def wave_to_spec(data, n_fft, hop_length, win_length=None):
    # 短時間フーリエ変換(STFT)を行い、スペクトログラムを取得
    spec = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(spec) # 振幅スペクトログラムを取得
    phase = np.exp(1j * np.angle(spec)) # 位相スペクトログラムを取得(フェーザ表示)
    # mel_spec = librosa.feature.melspectrogram(data, sr=sr, n_mels=128) # メルスペクトログラムを用いる場合はこっちを使う
    return mag, phase

# スペクトログラムを音声データに変換する
def spec_to_wav(spec, hop_length):
    # 逆短時間フーリエ変換(iSTFT)を行い、スペクトログラムから音声データを取得
    wav_data = librosa.istft(spec, hop_length=hop_length)
    return wav_data

# スペクトログラムを図にプロットする関数
def spec_plot(base_dir, wav_path, save_path, audio_length):
    # soxコマンドによりwavファイルからスペクトログラムの画像を生成
    cmd1 = "sox {} -n trim 0 {} rate 16.0k spectrogram".format(wav_path, audio_length)
    subprocess.call(cmd1, shell=True)
    # 生成されたスペクトログラム画像を移動
    #(コマンドを実行したディレクトリにスペクトログラムが生成されてしまうため移動)
    spec_path = os.path.join(base_dir, "spectrogram.png")
    cmd2 = "mv {} {}".format(spec_path, save_path)
    subprocess.call(cmd2, shell=True)

# waveファイルを読み込み波形のグラフを保存する
def wave_plot(input_path, output_path, audio_length, fig_title=None, x_scale=1.0, y_scale=0.10, ylim_min=-0.5, ylim_max=0.5):
    # open wave file
    wf = wave.open(input_path,'r')

    # load wave data
    rate = wf.getframerate()  # サンプリングレート[1/s]
    chunk_size = rate * audio_length
    amp  = (2**8) ** wf.getsampwidth() / 2
    data = wf.readframes(chunk_size)   # バイナリ読み込み
    data = np.frombuffer(data,'int16') # intに変換
    data = data / amp                  # 振幅正規化

    # make time axis
    size = float(chunk_size)  # 波形サイズ
    x = np.arange(0, size/rate, 1.0/rate)

    # 図に描画
    sns.set() # スタイルをきれいにする
    fig = plt.figure(facecolor='w', linewidth=5, edgecolor='black')
    # ax = fig.add_subplot(1, 1, 1, title=fig_title) # 図を1行目1列の1番目に表示(図を1つしか表示しない場合)
    ax = fig.add_subplot(1, 1, 1, title=fig_title, ylim=(ylim_min, ylim_max)) # 図を1行目1列の1番目に表示(図を1つしか表示しない場合)
    ax.set_xlabel('time[s]') # x軸名を設定
    ax.set_ylabel('magnitude') # y軸名を設定
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(x_scale)) # x軸の主目盛を1.0ごとに表示
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(y_scale)) # y軸の主目盛を0.10ごとに表示
    file_name = os.path.basename(output_path).split('.')[0] # データの名前を設定
    ax.plot(x, data, label='{}'.format(file_name)) # データをプロット
    ax.legend(edgecolor="black") # 凡例を追加
    fig.savefig(output_path) # グラフを保存

# 混合音声とモデルが推定した音声の質を評価(SDR, SIR, SARを算出)
def audio_eval(audio_length, target_audio_path, interference_audio_path, mixed_audio_path, estimated_audio_path):
    """
    bss_eval_sourcesとbss_eval_imagesに関しては
    「http://bass-db.gforge.inria.fr/bss_eval/」
    を参照
    参考にしたソースコードは「https://github.com/craffel/mir_eval」
    """
    from .audio_evaluation.separation import bss_eval_sources, bss_eval_images

    target = load_audio_file(target_audio_path, audio_length)[np.newaxis, :]
    interference = load_audio_file(interference_audio_path, audio_length)[np.newaxis, :]
    mixed = load_audio_file(mixed_audio_path, audio_length)[np.newaxis, :]
    estimated = load_audio_file(estimated_audio_path, audio_length)[np.newaxis, :]

    reference = np.concatenate([target, interference], 0) # 目的音と外的雑音を結合する
    mixed = np.concatenate([mixed, mixed], 0) # referenceと同じ形になるように結合
    estimated = np.concatenate([estimated, estimated], 0) # referenceと同じ形になるように結合

    # シングルチャンネル用 (シングルチャンネルの場合音声はshape:[1, num_samples]の形式)
    if target.ndim == 2:
        mixed_result = bss_eval_sources(reference, mixed) # 混合音声のSDR, SIR, SARを算出
        reference_result = bss_eval_sources(reference, estimated) # モデルが推定した音声のSDR, SIR, SARを算出
        # 混合音声の評価結果
        sdr_mix = mixed_result[0][0]
        sir_mix = mixed_result[1][0]
        sar_mix = mixed_result[2][0]
        # 推定音声の評価結果
        sdr_est = reference_result[0][0]
        sir_est = reference_result[1][0]
        sar_est = reference_result[2][0]

    # マルチチャンネル用 (マルチチャンネルの場合音声はshape:[1, num_samples, num_channels]の形式)
    elif target.ndim == 3:
        mixed_result = bss_eval_images(reference, mixed) # 混合音声のSDR, SIR, SARを算出
        reference_result = bss_eval_images(reference, estimated) # モデルが推定した音声のSDR, SIR, SARを算出
        # 混合音声の評価結果
        sdr_mix = mixed_result[0][0]
        sir_mix = mixed_result[2][0]
        sar_mix = mixed_result[3][0]
        # 推定音声の評価結果
        sdr_est = reference_result[0][0]
        sir_est = reference_result[2][0]
        sar_est = reference_result[3][0]
    else:
        print("number of audio channels are incorrect")

    return sdr_mix, sir_mix, sar_mix, sdr_est, sir_est, sar_est

# 振幅スペクトログラム、位相差スペクトログラム、ログメルスペクトログラムを算出
class SpectrogramFeatures():
    # def __init__(self, args=None, wav=None, center=None, config=None):
    def __init__(self, wav=None, center=False, sampling_rate=16000, fft_size=512, hop_length=160):
        # self.args = args
        self.wav = wav
        self.center = center
        self.sampling_rate = sampling_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.wav_base_channel = 0 # 位相差スペクトログラムを算出する際の基準となる音声チャンネル（0チャンネル目を基準）
        self.num_mels = 64

        self.sin_window = np.zeros(self.fft_size)
        # 160であっているのか不明
        self.window_length4critical_sampling = 160 * int(np.floor(self.fft_size / 160))
        for i in range(self.window_length4critical_sampling):
            self.sin_window[i] = np.sin(np.pi * i / (self.window_length4critical_sampling - 1))

        self.wav_ch = self.wav.shape[1]
        # 1ch目の音声をテンプレートとして取り出し、短時間フーリエ変換
        wav_c_contiguous_template = np.require(self.wav[:, 0], dtype=np.float32, requirements=['C'])
        spec_template = librosa.core.stft(wav_c_contiguous_template, n_fft=self.fft_size, hop_length=self.hop_length, \
        center=self.center, window=self.sin_window) # used for num_frame
        self.num_bin = int(fft_size / 2) + 1 # 周波数ビンの数
        self.num_frame = spec_template.shape[1] # フレームの数
        self.complex_spec = np.ones((self.wav_ch, self.num_bin, self.num_frame), dtype='complex64')

        self.complex_spec[0] = spec_template
        # 複数チャンネルの音声をスペクトログラム（振幅＋位相）に変換
        for i in range(1, self.wav_ch):
            wav_c_contiguous = np.require(self.wav[:, i], dtype=np.float32, requirements=['C'])
            self.complex_spec[i] = librosa.core.stft(wav_c_contiguous, n_fft=self.fft_size, hop_length=self.hop_length, \
            center=self.center, window=self.sin_window)

    # 振幅スペクトログラムを算出
    def amplitude(self):
        self.amp = np.zeros((self.wav_ch, self.num_bin, self.num_frame), dtype='float32')

        for i in range(self.wav_ch):
            self.amp[i] = np.abs(self.complex_spec[i])
        """self.amp: (channels, freq_bin, time_steps)"""
        return self.amp

    # 位相差スペクトログラムを算出（マルチチャンネル音声間の位相差）
    def phasediff(self):
        self.phasediff = np.zeros((self.wav_ch - 1, self.num_bin, self.num_frame), dtype='float32')

        spec_base_angle = np.angle(self.complex_spec[self.wav_base_channel]) # 基準チャンネルの偏角
        channel_num_list = np.delete(np.arange(self.wav_ch), self.wav_base_channel) # 基準チャンネル以外のチャンネル番号リスト
        # 各チャンネルの音声の基準チャンネルのからの位相差（偏角）を算出
        for idx, channnel_num in enumerate(channel_num_list):
            spec_angle = np.angle(self.complex_spec[channnel_num]) - spec_base_angle
            spec_angle[spec_angle < 0] += 2 * np.pi
            self.phasediff[idx] = spec_angle

        return self.phasediff

    # ログメルスペクトログラムを算出
    def log_mel_spec(self):
        mel_fb = librosa.filters.mel(self.sampling_rate, self.fft_size, n_mels=self.num_mels)
        self.log_mel_spec = np.ones((self.wav_ch, self._num_mels, self.frame_num), dtype='float32')

        for i in range(self.wav_ch):
            power_spec = np.abs(self.complex_spec[i]) ** 2 # パワースペクトログラムを算出
            mel_power_spec = np.dot(mel_fb, power_spec)
            self.log_mel_spec[i] = 10.0 * np.log10(np.maximum(1e-10, mel_power_spec)) # logがマイナス無限にならないように対数変換

        return self.log_mel_spec

    # GCC-PHAT
    def nCr(self, n, r):
        import math
        return math.factorial(n) // math.factorial(r) // math.factorial(n-r)

    def gcc_phat(self):
        gcc_channels = self.nCr(self.wav_ch, 2)
        self.gcc_feat = np.zeros((gcc_channels, n_mels, self.num_frame))

        cnt = 0
        for m in range(self.wav_ch):
            for n in range(m + 1, self.wav_ch):
                R = np.conj(self.complex_spec[m, :, :]) * self.complex_spec[n, :, :]
                cc = np.fft.irfft(np.exp(1.j * np.angle(R)), axis=0)
                cc = np.concatenate((cc[-n_mels // 2:, :], cc[:n_mels // 2, :]), axis=0)
                self.gcc_feat[cnt, :, :] = cc
                cnt += 1

        return self.gcc_phat

"""
自然言語処理用
"""
# Mecabを用いた形態素解析(Morphological Analysis)
def mecab_wakati(text):
    import MeCab
    import re
    # MeCabで分かち書き　-dに辞書を指定
    tagger = MeCab.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    text = tagger.parse(text)
    # 半角全角英数字除去
    text = re.sub(r'[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+', " ", text)
    # 記号もろもろ除去
    text = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", text)
    # スペースで区切って形態素の配列へ
    wakati = text.split(" ")
    # 空の要素は削除
    wakati = list(filter(("").__ne__, wakati))
    return wakati

# Janomeを用いた形態素解析(Morphological Analysis)
def Janome_wakati(text):
    from janome.tokenizer import Tokenizer
    tagger = Tokenizer()
    wakati = [tok for tok in tagger.tokenize(text, wakati=True)]
    return wakati



"""
ファイル管理用
"""




"""
図に各関数のグラフをプロット
"""
# 任意のグラフを表示する
def fig_plot(x, y, x_scale=1.0, y_scale=0.10, ylim_min=-0.5, ylim_max=0.5, fig_title=None, x_label_name='x', y_label_name='y'):
    # 図に描画
    sns.set() # スタイルをきれいにする
    fig = plt.figure(facecolor='w', linewidth=1, edgecolor='black')
    # ax = fig.add_subplot(1, 1, 1, title=fig_title) # 図を1行目1列の1番目に表示(図を1つしか表示しない場合)
    ax = fig.add_subplot(1, 1, 1, title=fig_title, ylim=(ylim_min, ylim_max)) # 図を1行目1列の1番目に表示(図を1つしか表示しない場合)
    ax.set_xlabel(x_label_name) # x軸名を設定
    ax.set_ylabel(y_label_name) # y軸名を設定
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(x_scale)) # x軸の主目盛を1.0ごとに表示
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(y_scale)) # y軸の主目盛を0.10ごとに表示
    ax.scatter(x, y) # データをプロット
    plt.show()
    # ax.legend(edgecolor="black") # 凡例を追加
    # file_name = os.path.basename(output_path).split('.')[0] # データの名前を設定
    # fig.savefig(output_path) # グラフを保存

# 混同行列の作成
def mk_confusion_matrix(label, predicted):
    """
    label: true label of data
    predicted: predicted result of data
    """
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(label, predicted)

    sns.heatmap(matrix, square=True, annot=True, cbar=False, cmap='RdPu')
    plt.xlabel('predicted value')
    plt.ylabel('true value')
    plt.show()


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

# 3次元点群を描画
def plot_3D(X, y, elev=30, azim=30):
    r = np.exp(-(X**2).sum(1)) # 中央の点群を中心とする放射基底関数を使用
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')
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

    # 3次元点群描画用
    # X, y = make_circles(100, factor=.1, noise=.1)
    # plot_3D(X, y)

    # # 混同行列描画用
    # label = [1, 2, 3, 4, 5, 6, 7, 8]
    # predicted = [1, 3, 3, 5, 5, 8, 7, 6]
    # mk_confusion_matrix(label, predicted)

    # Janomeを用いた形態素解析用
    # text = "機械学習が好きです。"
    # Janomewakati(text)

    # Mecabを用いた形態素解析用
    # text = "【人工知能】は「人間」の仕事を奪った"
    # wakati = mecab_wakati(text)
    # print(wakati)

    # # Spectrogram特徴量の抽出用
    # wav_path = "../speech_denoising_MCDUnet/data/shokudo_noise/shokudo_rec1_split_3_sec/shokudo_rec1_split_0.wav"
    # wav = load_audio_file(wav_path, length=3)
    # spec = SpectrogramFeatures(wav)
    # amp_spec = spec.amplitude()
    # phasediff_spec = spec.phasediff()
    # print(amp_spec.shape)
    # print(phasediff_spec.shape)

    # # 音声ファイル再生用
    # import sounddevice as sd
    # file_path = "../AudioDatasets/NoisySpeechDetabase/clean_trainset_28spk_wav_16kHz/p226_001.wav"
    # audio_length = 3
    # sampling_rate = 16000
    # data = load_audio_file(file_path, audio_length, sampling_rate)
    # play_audio(data, sampling_rate)

    # # 音声録音用
    # import sounddevice as sd
    # audio_length = 30
    # sampling_rate = 16000
    # channels = 1
    # save_path = "./robot_self_noise.wav"
    # rec_audio(audio_length, sampling_rate, channels, save_path)

    # file_path = "./test/target_voice.wav"
    # save_path = "./test/istft.wav"
    # audio_length = 3
    # sampling_rate = 16000
    # n_fft=1024
    # hop_length=768
    # audio_data = load_audio_file(file_path, audio_length, sampling_rate)
    # mag, phase = wave_to_spec(audio_data, n_fft, hop_length)
    # target_spec = mag * phase
    # istft_data = spec_to_wav(target_spec, hop_length)
    # save_audio_file(save_path, istft_data, sampling_rate)

    # fft_size = 1024
    # hop_length = 768
    # file_path = "../AudioDatasets/NoisySpeechDetabase/clean_trainset_28spk_wav_16kHz/p226_001.wav"
    # audio_length = 3
    # sampling_rate = 16000
    # audio_data = load_audio_file(file_path, audio_length, sampling_rate)
    # amp, phase = wave_to_spec(audio_data, fft_size, hop_length, win_length=None)
    # print(amp.shape)

    # input_path = "../AudioDatasets/DEMAND/Multichannel_noise_at_LIVING/ch01.wav"
    # output_path = "./test.png"
    # audio_length = 30
    # wave_plot(input_path, output_path, audio_length, fig_title=None, ylim_min=-0.01, ylim_max=0.01)
