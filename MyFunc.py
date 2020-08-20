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
def wave_to_spec(data, n_fft, hop_length, win_length):
    # 短時間フーリエ変換(STFT)を行い、スペクトログラムを取得
    spec = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(spec) # 振幅スペクトログラムを取得
    phase = np.exp(1.j * np.angle(spec)) # 位相スペクトログラムを取得(フェーザ表示)
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
def wave_plot(input_path, output_path, audio_length, fig_title=None):
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
    ax = fig.add_subplot(1, 1, 1, title=fig_title, ylim=(-0.5, 0.5)) # 図を1行目1列の1番目に表示(図を1つしか表示しない場合)
    ax.set_xlabel('time[s]') # x軸名を設定
    ax.set_xlabel('time[s]') # x軸名を設定
    ax.set_ylabel('magnitude') # y軸名を設定
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0)) # x軸の主目盛を1.0ごとに表示
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.10)) # y軸の主目盛を0.10ごとに表示
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
        print("SDR_mix: {:.3f}, SIR_mix: {:.3f}, SAR_mix: {:.3f}".format(mixed_result[0][0], mixed_result[1][0], mixed_result[2][0]))
        print("SDR_est: {:.3f}, SIR_est: {:.3f}, SAR_est: {:.3f}".format(reference_result[0][0], reference_result[1][0], reference_result[2][0]))

    # マルチチャンネル用 (マルチチャンネルの場合音声はshape:[1, num_samples, num_channels]の形式)
    elif target.ndim == 3:
        mixed_result = bss_eval_images(reference, mixed) # 混合音声のSDR, SIR, SARを算出
        reference_result = bss_eval_images(reference, estimated) # モデルが推定した音声のSDR, SIR, SARを算出
        print("SDR_mix: {:.3f}, SIR_mix: {:.3f}, SAR_mix: {:.3f}".format(mixed_result[0][0], mixed_result[2][0], mixed_result[3][0]))
        print("SDR_est: {:.3f}, SIR_est: {:.3f}, SAR_est: {:.3f}".format(reference_result[0][0], reference_result[2][0], reference_result[3][0]))

    else:
        print("number of audio channels are incorrect")


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
PyTorch用
"""
# モデルのパラメータ数をカウント
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



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
