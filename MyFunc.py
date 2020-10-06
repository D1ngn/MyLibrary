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
# データセット
from sklearn.datasets import make_blobs # クラスタリング用の等方性ガウス点群を生成
from sklearn.datasets import make_circles # 大きい円と小さい円を描くように２次元の点群を生成
# 特徴量エンジニアリング
from sklearn.feature_extraction.text import TfidfVectorizer
# データ作成用
from sklearn.model_selection import train_test_split, GridSearchCV # データ分割用, グリッドサーチ用
# モデル
from sklearn.pipeline import make_pipeline # 複数モデルを連続で実行するためのパイプライン
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC # Support Vector Classifier
from sklearn.decomposition import PCA # Principal Component Analysis
#評価用
from sklearn.metrics import accuracy_score




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
機械学習手法
"""
# 回帰(教師あり学習)
# 線形回帰()
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


# 分類(教師あり学習)
# ナイーブベイズ分類(各ラベルからのデータが単純なガウス分布や多項分布に基づいていると仮定)
# 多項分布に基づくテキスト分類
class NaiveBayes():
    def __init__(self):
        self.model = make_pipeline(TfidfVectorizer(), MultinomialNB()) # TF-IDFによる単語の重み付け(ベクトル化)+多項分布ナイーズベイズ

    def fit(self, data, target):
        self.model.fit(data, target) # モデル学習

    def text_category_classification(self, text, labels):
        pred = self.model.predict([text]) # 予測したラベルのインデックスを取得
        category = labels[pred[0]]
        return category


# サポートベクターマシンによる点群データの分類
class SupportVectorClassify():
    def __init__(self):
        # モデルのインスタンスを生成　
        # C：ソフトマージン用のパラメータ(マージンの中にポイントがどれだけ入れるかを指定する 大きいほど許容数が少なく、小さいほど許容数が多い)
        self.model = SVC(kernel='linear', C=1E10) # linear:線形分類
        # self.model = SVC(kernel='rbf', C=1E10) # rbf(radial basis function)：放射基底関数(カーネルトリックを用いたカーネル変換)

    def fit(self, X, y):
        self.model.fit(X, y) # モデル学習

    # サポートベクターマシンの決定直線を描画
    def plot_svc_decision_function(self, X, y, ax=None, plot_support=True):
        plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
        if ax is None:
            ax = plt.gca() # get current axis

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x = np.linspace(xlim[0], xlim[1], 30)
        y = np.linspace(ylim[0], ylim[1], 30)
        Y, X = np.meshgrid(y, x) # 格子点の座標データを生成

        xy = np.vstack([X.ravel(), Y.ravel()]).T

        P = self.model.decision_function(xy).reshape(X.shape)

        ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--']) # 決定境界を描画

        if plot_support:
            ax.scatter(self.model.support_vectors_[:, 0],self.model.support_vectors_[:, 1],
             s=300, linewidth=1, facecolors='none', edgecolor='black') # サポートベクターを黒丸で囲む
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        plt.show()


# ランダムフォレストによる分類




# クラスタリング(教師なし学習)



# 次元削減(教師なし学習)
class DimReductionPCA():
    def __init__(self, n_components):
        """
        n_components: Number of components
        """
        self.model = PCA(n_components)

    def fit(self, X):
        self.model.fit(X)

    def cal_cumulative_contribuntion_rate(self, X):
        """
        データを記述するのに必要は成分数を推定するために
        成分数に対する累積寄与率を表示
        """
        pca = PCA().fit(X)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()


# グリッドサーチによるハイパーパラメータの探索(主成分分析とサポートベクターマシンを用いた顔認識)
class GridSearch_PCA_SVC():
    def __init__(self, n_components, param_grid):
        pca = PCA(n_components=n_components, whiten=True, random_state=42, svd_solver='randomized')
        svc = SVC(kernel='rbf', class_weight='balanced')
        self.model = make_pipeline(pca, svc)
        self.grid = GridSearchCV(self.model, param_grid)

    def fit(self, X, y):
        self.grid.fit(X, y)
        print("best_params:", self.grid.best_params_)

    def predict(self, X):
        result = self.grid.best_estimator_.predict(X)
        return result

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


    # 線形回帰用
    # x = 10 * np.random.rand(50)
    # X = x[:, np.newaxis] # shape: [n_samples,] → [n_samples, n_features]の形式へ
    # y = 3 * x + np.random.randn(50)
    # X1, X2, y1, y2 = train_test_split(X, y, random_state=0, test_size=0.33) # データをtrain用とtest用に2:1で分ける
    # output = linear_regression(X1, X2, y1, y2)

    # 多項分布ナイーブベイズに基づくテキストカテゴリ分類用
    # from sklearn.datasets import fetch_20newsgroups
    # categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space', 'comp.graphics']
    # train = fetch_20newsgroups(subset='train', categories=categories)　# データ取得
    # test = fetch_20newsgroups(subset='test', categories=categories)　# データ取得
    # naive_bayes = NaiveBayes() # モデル初期化
    # naive_bayes.fit(train.data, train.target) # モデル学習
    # predicted_result = naive_bayes.text_category_classification('sending a payload to the ISS', train.target_names) #　テキスト分類
    # print(predicted_result)

    # サポートベクターマシンによる点群分類用
    # X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
    # points_svc = SupportVectorClassify() # モデル初期化
    # points_svc.fit(X, y)
    # points_svc.plot_svc_decision_function(X, y)

    # # グリッドサーチによるハイパーパラメータの探索(主成分分析とサポートベクターマシンを用いた顔認識)用
    # from sklearn.datasets import fetch_lfw_people
    # faces = fetch_lfw_people(min_faces_per_person=60)
    # Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)
    # n_components = 150 # 主成分分析の次元数
    # param_grid = {'svc__C': [1, 5, 10, 50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]} # グリッドサーチで最適値を探索するパラメータと範囲を指定
    # gps = GridSearch_PCA_SVC(n_components, param_grid)
    # gps.fit(Xtrain, ytrain)
    # yfit = gps.predict(Xtest)
    # # print(yfit)
    # fig, ax = plt.subplots(4, 6)
    # for i, axi in enumerate(ax.flat):
    #     axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    #     axi.set(xticks=[], yticks=[])
    #     axi.set_ylabel(faces.target_names[yfit[i]].split()[-1], color='black' if yfit[i] == ytest[i] else 'red')
    # fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)
    # plt.show()

    # # 主成分分析による次元削減
    # # ランダムな点群
    # n_components = 2
    # rng = np.random.RandomState(1)
    # X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
    # pca = DimReductionPCA(n_components=n_components)
    # pca.fit(X) # 学習（次元削減）
    # print(pca.model.components_) # 各成分のベクトル
    # print(pca.model.explained_variance_) # 各成分の寄与率
    # fig_plot(X[:, 0], X[:, 1])
    # # 手書き数字画像
    # from sklearn.datasets import load_digits
    # digits = load_digits()
    # """
    # digits: (n_samples=1797, n_features=64)
    # """
    # pca = DimReductionPCA(n_components=2)
    # projected = pca.model.fit_transform(digits.data) # 次元削減
    # """
    # projected: (n_samples=1797, n_features=2)
    # """
    # pca.cal_cumulative_contribuntion_rate(digits.data) # 累積寄与率の表示（n_componentsの最適数を定めるため）

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
