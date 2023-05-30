import os
import subprocess

# 画像処理用
# import cv2
# from PIL import Image

# 音声処理用
import librosa
import soundfile as sf
import wave
import sounddevice as sd
from scipy import signal

# 機械学習用
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d # 3次元描画用


"""
データ処理
"""
# 正規化処理
# データを標準化（平均0、分散1に正規化（Z-score Normalization））
def standardize(data):
    data_mean = data.mean(keepdims=True)
    data_std = data.std(keepdims=True, ddof=0) # 母集団の標準偏差（標本標準偏差を使用する場合はddof=1）
    standardized_data = (data - data_mean) / data_std
    return standardized_data


# 最小値0、最大値1に正規化(Min-Max Normalization)
def min_max_normalize(data):
    data_min = data.min(keepdims=True)
    data_max = data.max(keepdims=True)
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data

"""
音声処理
"""
# 音声データをロードし、指定された秒数とサンプリングレートでリサンプル
def load_audio_file(file_path, length, sample_rate):
    data, sr = sf.read(file_path)
    # データが設定値よりも大きい場合は大きさを超えた分をカットする
    # データが設定値よりも小さい場合はデータの後ろを0でパディングする
    # シングルチャンネル(モノラル)の場合 (data.shape: [num_samples,])
    if data.ndim == 1:
        if len(data) > sample_rate*length:
            data = data[:sample_rate*length]
        else:
            data = np.pad(data, (0, max(0, sample_rate*length - len(data))), "constant")
        """data: (num_samples, )"""
    # マルチチャンネルの場合 (data.shape: [num_samples, num_channels])
    elif data.ndim == 2:
        if data.shape[0] > sample_rate*length:
            data = data[:sample_rate*length, :]
        else:
            data = np.pad(data, [(0, max(0, sample_rate*length-data.shape[0])), (0, 0)], "constant")
        """data: (num_samples, num_channels)"""
    else:
        print("number of audio channels are incorrect")
    return data

# 音声データを指定したサンプリングレートで保存
def save_audio_file(file_path, data, sample_rate=16000):
    """"data: (num_samples, num_channels)"""
    sf.write(file_path, data, sample_rate)

# 音声ファイルを再生
def play_audio(data, sample_rate):
    # dataを再生する
    sd.play(data, sample_rate)
    print("start")
    # 再生が終わるまで待つ
    status = sd.wait()
    print("finish")

# 音声を録音
def rec_audio(audio_length, sample_rate, channels, save_path):
    print("start recording...")
    data = sd.rec(int(audio_length*sample_rate), sample_rate, channels=channels)
    # 録音が終わるまで待つ
    sd.wait()
    print("finish recording!")
    save_audio_file(save_path, data, sample_rate)

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


# 音声データを振幅スペクトログラムと位相スペクトログラムに変換する
def wave_to_spec(data, fft_size, hop_length, win_length=None):
    # 短時間フーリエ変換(STFT)を行い、スペクトログラムを取得
    complex_spec = librosa.stft(data, n_fft=fft_size, hop_length=hop_length, win_length=win_length, window='hann')
    amp_spec = np.abs(complex_spec) # 振幅スペクトログラムを取得
    phase_spec = np.exp(1j * np.angle(complex_spec)) # 位相スペクトログラムを取得(フェーザ表示)
    return amp_spec, phase_spec

# マルチチャンネルの音声データをスペクトログラムに変換する
def wave_to_spec_multi(data, sample_rate, fft_size, hop_length):
    """
    data: (num_channels, num_samples)
    sample_rate: sampling rate (int)
    fft_size: length of each segment (int)
    hop_length: shift size of each segment (int)
    """
    f, t, complex_spec = signal.stft(data, fs=sample_rate, window='hann', nperseg=fft_size, noverlap=fft_size-hop_length)
    """f: (freq_bins,), t: (time_frames,), spectrogram: (num_microphones, freq_bins, time_frames)"""
    amp_spec = np.abs(complex_spec) # 振幅スペクトログラムを取得
    phase_spec = np.exp(1j * np.angle(complex_spec)) # 位相スペクトログラムを取得(フェーザ表示)
    return amp_spec, phase_spec

# スペクトログラムを音声データに変換する
def spec_to_wave(spectrogram, hop_length):
    # 逆短時間フーリエ変換(iSTFT)を行い、スペクトログラムから音声データを取得
    wave_data = librosa.istft(spectrogram, hop_length=hop_length)
    return wave_data

# スペクトログラムを音声データに変換する（librosaが使えない場合）
def spec_to_wave_without_librosa(spectrogram, sample_rate, fft_size, hop_length):
    """
    spectrogram: (freq_bins, time_frames)
    sample_rate: sampling rate (int)
    fft_size: length of each segment (int)
    hop_length: shift size of each segment (int)
    """
    # 逆短時間フーリエ変換(iSTFT)を行い、スペクトログラムから音声データを取得
    wave_data = signal.istft(spectrogram, fs=sample_rate, window='hann', nperseg=fft_size, noverlap=fft_size-hop_length)
    return wave_data

# スペクトログラムを図にプロットする関数
def spec_plot(base_dir, wav_path, save_path):
    # soxコマンドによりwavファイルからスペクトログラムの画像を生成
    cmd1 = "sox {} -n rate 16.0k spectrogram".format(wav_path)
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
    # ax.legend(edgecolor="black") # 凡例を追加
    fig.savefig(output_path) # グラフを保存

# SNRを測る
def calculate_snr(target, out):
    """
    target: 目的音 (num_samples, )
    out: 雑音除去後の信号 (num_samples, )
    """
    wave_length = np.minimum(np.shape(target)[0], np.shape(out)[0])
    # 消し残った雑音
    target = target[:wave_length]
    out = out[:wave_length]
    noise = target - out
    snr = 10. * np.log10(np.sum(np.square(target)) / np.sum(np.square(noise)))
    return snr

# 混合音声とモデルが推定した音声の質を評価(SDR, SIR, SARを算出)
def audio_eval(sample_rate, target_audio_path, interference_audio_path, mixed_audio_path, estimated_audio_path):
    """
    bss_eval_sourcesとbss_eval_imagesに関しては
    「http://bass-db.gforge.inria.fr/bss_eval/」
    を参照
    参考にしたソースコードは「https://github.com/craffel/mir_eval」
    """
    from .audio_evaluation.separation import bss_eval_sources, bss_eval_images

    # target = load_audio_file(target_audio_path, audio_length, sample_rate)[np.newaxis, :]
    # interference = load_audio_file(interference_audio_path, audio_length, sample_rate)[np.newaxis, :]
    # mixed = load_audio_file(mixed_audio_path, audio_length, sample_rate)[np.newaxis, :]
    # estimated = load_audio_file(estimated_audio_path, audio_length, sample_rate)[np.newaxis, :]

    # 音声データの読み込み
    target = sf.read(target_audio_path)[0][np.newaxis, :]
    interference = sf.read(interference_audio_path)[0][np.newaxis, :]
    mixed = sf.read(mixed_audio_path)[0][np.newaxis, :]
    estimated = sf.read(estimated_audio_path)[0][np.newaxis, :]
    
    # 各音声の長さの最大値を取得（評価時に音声の長さを揃える必要があるため）
    max_audio_length = np.amax(np.array([target.shape[1], interference.shape[1], mixed.shape[1], estimated.shape[1]]))
  
    # データが三次元の時、(手前, 奥), (上,下), (左, 右)の順番でパディングを実行
    target = np.pad(target, [(0, 0), (0, max_audio_length - target.shape[1]), (0, 0)], 'constant')
    interference = np.pad(interference, [(0, 0), (0, max_audio_length - interference.shape[1]), (0, 0)], 'constant') 
    mixed = np.pad(mixed, [(0, 0), (0, max_audio_length - mixed.shape[1]), (0, 0)], 'constant') 
    estimated = np.pad(estimated, [(0, 0), (0, max_audio_length - estimated.shape[1]), (0, 0)], 'constant') 

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

# 混合音声とモデルが推定した音声の質を評価(SDR, SIR, SARを算出)
def audio_eval_from_data(target_audio_data, interference_audio_data, mixed_audio_data, estimated_audio_data):
    """
    bss_eval_sourcesとbss_eval_imagesに関しては
    「http://bass-db.gforge.inria.fr/bss_eval/」
    を参照
    参考にしたソースコードは「https://github.com/craffel/mir_eval」
    """
    from .audio_evaluation.separation import bss_eval_sources, bss_eval_images

    target = target_audio_data[np.newaxis, :]
    interference = interference_audio_data[np.newaxis, :]
    mixed = mixed_audio_data[np.newaxis, :]
    estimated = estimated_audio_data[np.newaxis, :]

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

# ESPNetを用いた音声認識
class ASR():
    def __init__(self, lang='eng'):
        # 必要モジュールをインポート（あらかじめ「pip3 install espnet_model_zoo」を実行）
        from espnet_model_zoo.downloader import ModelDownloader
        from espnet2.bin.asr_inference import Speech2Text
        # E2E-ASRモデルのインスタンスを作成
        d = ModelDownloader()
        # 英語版
        if lang == 'eng':
            self.speech2text = Speech2Text(
                # タスク（音声認識）と使用するコーパスを指定し、学習済みモデルをダウンロード
                **d.download_and_unpack(task="asr", corpus="librispeech")
            )
        # 日本語版
        elif lang == 'jp':
            self.speech2text = Speech2Text(
                # タスク（音声認識）と使用するコーパスを指定し、学習済みモデルをダウンロード
                **d.download_and_unpack(task="asr", corpus="jsut")
            )
    # 音声認識を実行
    def speech_recognition(self, audio_path):
        audio_data, _ = sf.read(audio_path)
        text, token, *_ = self.speech2text(audio_data)[0]
        return text

# Juliusを用いた音声認識
def asr_julius(input_file_path):
    temp_file = "julius_asr_recog_result.txt"
    # juliusによる音声認識を実行し、結果をファイルに出力
    # # 混合ガウスモデル（GMM）ベースの音響モデルを用いる場合→今は「前に進め」、「後ろに退がれ」など（オリジナルの単語辞書に登録されたもの）を認識
    # asr_cmd = "echo {} | julius -C ~/julius/dictation-kit-4.5/main.jconf -C ~/julius/dictation-kit-4.5/am-gmm.jconf -nostrip -input rawfile -quiet > {}".format(input_file_path, temp_file)
    # DNNベースの音響モデルを用いる場合→今はさまざまな日本語を認識（英語は不可）
    asr_cmd = "echo {} | julius -C ~/julius/dictation-kit-4.5/main.jconf -C ~/julius/dictation-kit-4.5/am-dnn.jconf -dnnconf ~/julius/dictation-kit-4.5/julius.dnnconf -nostrip -input rawfile -quiet > {}".format(input_file_path, save_path)
    subprocess.call(asr_cmd, shell=True)
    # 出力ファイルから認識結果の部分のみを抽出
    with open(temp_file) as f:
        lines = f.readlines()
    recog_text_line = [line.strip() for line in lines if line.startswith('sentence1')] # "sentence1"から始まる行をサーチ
    recog_result = recog_text_line[0][12:-2] # "sentence1: "から"。"の間の文章を抽出
    # 余分なファイルが残らないように削除
    os.remove(temp_file)
    return recog_result

# 正解ラベルのテキストと音声認識結果のテキストの距離を算出
def editDistance(r, h):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.

    Main algorithm used is dynamic programming.

    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d

def getStepList(r, h, d):
    '''
    This function is to get the list of steps in the process of dynamic programming.

    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and r.
    '''
    x = len(r)
    y = len(h)
    list = []
    while True:
        if x == 0 and y == 0: 
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] and r[x-1] == h[y-1]: 
            list.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y-1]+1:
            list.append("i")
            x = x
            y = y - 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1]+1:
            list.append("s")
            x = x - 1
            y = y - 1
        else:
            list.append("d")
            x = x - 1
            y = y
    return list[::-1]

# 音声認識評価結果をファイルに書き込む
def alignedPrint(list, r, h, result, result_path):
    '''
    This funcition is to print the result of comparing reference and hypothesis sentences in an aligned way.
    
    Attributes:
        list   -> the list of steps.
        r      -> the list of words produced by splitting reference sentence.
        h      -> the list of words produced by splitting hypothesis sentence.
        result -> the rate calculated based on edit distance.
    '''
    # 結果を保存するファイルの中身を初期化
    if os.path.exists(result_path):
        os.remove(result_path)
    
#     print("REF:", end=" ")
    with open(result_path, mode='a') as f:
        print("REF:", end=" ", file=f)
    for i in range(len(list)):
        if list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
#             print(" "*(len(h[index])), end=" ")
            with open(result_path, mode='a') as f:
                print(" "*(len(h[index])), end=" ", file=f)
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) < len(h[index2]):
#                 print(r[index1] + " " * (len(h[index2])-len(r[index1])), end=" ")
                with open(result_path, mode='a') as f:
                    print(r[index1] + " " * (len(h[index2])-len(r[index1])), end=" ", file=f)
            else:
#                 print(r[index1], end=" "),
                with open(result_path, mode='a') as f:
                    print(r[index1], end=" ", file=f)
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
#             print(r[index], end=" "),
            with open(result_path, mode='a') as f:
                print(r[index], end=" ", file=f)
#     print("\nHYP:", end=" ")
    with open(result_path, mode='a') as f:
        print("\nHYP:", end=" ", file=f)
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
#             print(" " * (len(r[index])), end=" ")
            with open(result_path, mode='a') as f:
                print(" " * (len(r[index])), end=" ", file=f)
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
#                 print(h[index2] + " " * (len(r[index1])-len(h[index2])), end=" ")
                with open(result_path, mode='a') as f:
                    print(h[index2] + " " * (len(r[index1])-len(h[index2])), end=" ", file=f)
            else:
#                 print(h[index2], end=" ")
                with open(result_path, mode='a') as f:
                    print(h[index2], end=" ", file=f)
        else:
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
#             print(h[index], end=" ")
            with open(result_path, mode='a') as f:
                print(h[index], end=" ", file=f)
#     print("\nEVA:", end=" ")
    with open(result_path, mode='a') as f:
        print("\nEVA:", end=" ", file=f)
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
#             print("D" + " " * (len(r[index])-1), end=" ")
            with open(result_path, mode='a') as f:
                print("D" + " " * (len(r[index])-1), end=" ", file=f)
        elif list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
#             print("I" + " " * (len(h[index])-1), end=" ")
            with open(result_path, mode='a') as f:
                print("I" + " " * (len(h[index])-1), end=" ", file=f)
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
#                 print("S" + " " * (len(r[index1])-1), end=" ")
                with open(result_path, mode='a') as f:
                    print("S" + " " * (len(r[index1])-1), end=" ", file=f)
            else:
#                 print("S" + " " * (len(h[index2])-1), end=" ")
                with open(result_path, mode='a') as f:
                    print("S" + " " * (len(h[index2])-1), end=" ", file=f)
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
#             print(" " * (len(r[index])), end=" ")
            with open(result_path, mode='a') as f:
                print(" " * (len(r[index])), end=" ", file=f)
#     print("\nWER: " + result)
    with open(result_path, mode='a') as f:
        print("\nWER: " + result, file=f)
    
# 音声認識性能（Word Error Rate; WER）の評価
def asr_eval(ref_text, hyp_text, result_path):
    """
    ref_text: 正解ラベルのテキスト （例） ['IT', 'IS', 'MARVELLOUS']
    hyp_text: 音声認識結果のテキスト （例） ['IT', 'WAS', 'MADNESS']
    result_path: 音声認識性能の評価結果を保存するファイルのパス
    """
    # build the matrix
    d = editDistance(ref_text, hyp_text)

    # find out the manipulation steps
    list = getStepList(ref_text, hyp_text, d)

    # print the result in aligned way
    result = float(d[len(ref_text)][len(hyp_text)]) / len(ref_text) * 100
    result_str = str("%.2f" % result) + "%"
    alignedPrint(list, ref_text, hyp_text, result_str, result_path)
    return result
    

# 振幅スペクトログラム、位相差スペクトログラム、ログメルスペクトログラムを算出
class SpectrogramFeatures():
    # def __init__(self, args=None, wav=None, center=None, config=None):
    def __init__(self, wav=None, center=False, sample_rate=16000, fft_size=512, hop_length=160):
        # self.args = args
        self.wav = wav
        self.center = center
        self.sample_rate = sample_rate
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
        mel_fb = librosa.filters.mel(self.sample_rate, self.fft_size, n_mels=self.num_mels)
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

# ビームフォーミング関連
# ステアリングベクトルを算出
def calculate_steering_vector(mic_alignments, source_locations, freqs, sound_speed=340, is_use_far=False):
    """
    mic_alignments: (3D-coordinate(x,y,z)=3, num_microphones(M))
    source_locations: (3D-coordinate(x,y,z)=3, num_sources(Ns))
    freqs: (freq_bins(Nk), )
    sound_speed: constant number
    is_use_far: Far -> True, Near -> False
    return: steering vector (Nk, Ns, M)
    """
    # マイクロホン数を取得
    n_channels = np.shape(mic_alignments)[1]
    # 音源数を取得
    n_sources = np.shape(source_locations)[1]
    # Far-field仮定（無限遠に音源が存在すると仮定）の場合
    if is_use_far == True:
        # 音源位置を正規化
        norm_source_locations = source_locations / np.linalg.norm(source_locations, 2, axis=0, keepdims=True)
        """norm_source_locations: (3D-coordinate(x,y,z)=3, num_sources)"""
        # 位相を求める
        steering_phase = np.einsum('k,ism,ism->ksm', 2.j*np.pi/sound_speed*freqs, norm_source_locations[...,None], mic_alignments[:, None, :])
        """steering_phase: (freq_bins, num_sources, num_microphones)"""
        # ステアリングベクトルを算出
        steering_vector = 1./np.sqrt(n_channels)*np.exp(steering_phase)
        """steering_vector: (freq_bins, num_sources, num_microphones)"""
        return steering_vector
    # Near-field仮定（音源がマイクロホン近くに存在すると仮定）の場合
    else:
        # 音源とマイクロホンの距離を求める
        distance = np.sqrt(np.sum(np.square(source_locations[..., None]-mic_alignments[:, None, :]), axis=0))
        """distance: (num_sources, num_microphones)"""
        # 遅延時間 [sec]
        delay = distance / sound_speed
        """delay: (num_sources, num_microphones)"""
        # ステアリングベクトルの位相を求める
        steering_phase = np.einsum('k,sm->ksm', -2.j*np.pi*freqs, delay)
        """steering_phase: (freq_bins, num_sources, num_microphones)"""
        # 音量の減衰
        steering_decay_ratio = 1./distance
        # ステアリングベクトルを求める
        steering_vector = steering_decay_ratio[None, ...]*np.exp(steering_phase)
        # 大きさ1で正規化する
        steering_vector = steering_vector / np.linalg.norm(steering_vector, 2, axis=2, keepdims=True)
        """steering_vector: (freq_bins, num_sources, num_microphones)"""
        return steering_vector
        
# 時間周波数マスクを推定する
def estimate_mask(stft_data, steering_vectors, omega):
    """
    stft_data: マイクロホン入力信号 (num_microhones, freq_bins, time_frames)
    steering_vectors: ステアリングベクトル (freq_bins, target_signal_range, num_microhones)
    omega: 目的音の範囲 (num_sources, target_signal_range=72)
    """
    inner_product = np.einsum("kim,mkt->kit", np.conjugate(steering_vectors), stft_data)
    """inner_product: (freq_bins, target_signal_range, time_frames)"""
    n_omega = np.shape(omega)[1]
    estimate_doas = np.argmax(np.abs(inner_product), axis=1)
    """estimate_doas: (freq_bins, time_frames)"""
    estimate_doas_mask = np.identity(n_omega)[estimate_doas]
    """estimate_doas_mask: (freq_bins, time_frames, target_signal_range)"""
    mask = np.einsum("kti,si->skt", estimate_doas_mask, omega)
    """mask: (num_sources, freq_bins, time_frames)"""
    return mask

# マスクと入力信号から共分散行列を推定
def estimate_covariance_matrix(stft_data, mask):
    """
    stft_data: 入力信号 (num_microphones, freq_bins, time_frames)
    mask: 音源ごとの時間周波数マスク (num_sources, freq_bins, time_frames)
    """
    # 目的音の共分散行列を推定する
    Rs = np.einsum("skt,mkt,nkt->skmn", mask, stft_data, np.conjugate(stft_data))
    """Rs: (num_sources, freq_bins, num_microphones, num_microphones)"""
    sum_target_mask = np.sum(mask, axis=2)
    """sum_target_mask: (num_sources, freq_bins)"""
    Rs = Rs / np.maximum(sum_target_mask, 1.e-18)[..., None, None]
    """Rs: (num_sources, freq_bins, num_microphones, num_microphones)"""
    # 雑音の共分散行列を推定する
    Rn = np.einsum("skt,mkt,nkt->skmn", 1-mask, stft_data, np.conjugate(stft_data))
    """Rn: (num_sources, freq_bins, num_microphones, num_microphones)"""
    sum_noise_mask = np.sum(1-mask, axis=2)
    """sum_noise_mask: (num_sources, freq_bins)"""
    Rn = Rn / np.maximum(sum_noise_mask, 1.e-18)[..., None, None]
    # 固有値分解をして半正定値行列に変換
    eigenvalues_Rs, eigenvectors_Rs = np.linalg.eigh(Rs)
    """eigenvalues_Rs: (num_sources, freq_bins, num_microphones), eigenvectors_Rs: (num_sources, freq_bins, num_microphones, num_microphones)"""
    Rs_org = Rs.copy()
    eigenvalues_Rs[np.real(eigenvalues_Rs) < 1.e-18] = 1.e-18 # 固有値が0より小さい場合は0に置き換える
    Rs = np.einsum("skmi,ski,skni->skmn", eigenvectors_Rs, eigenvalues_Rs, np.conjugate(eigenvectors_Rs))
    """Rn: (num_sources, freq_bins, num_microphones, num_microphones)"""
    eigenvalues_Rn, eigenvectors_Rn = np.linalg.eigh(Rn)
    """eigenvalues_Rn: (num_sources, freq_bins, ), eigenvectors_Rn: (num_sources, freq_bins, num_microphones, )"""
    Rn_org = Rn.copy()
    eigenvalues_Rn[np.real(eigenvalues_Rn) < 1.e-18] = 1.e-18 # 固有値が0より小さい場合は0に置き換える
    Rn = np.einsum("skmi,ski,skni->skmn", eigenvectors_Rn, eigenvalues_Rn, np.conjugate(eigenvectors_Rn))
    """Rn: (num_sources, freq_bins, num_microphones, num_microphones)"""
    return Rs, Rn

# 音源のスパース性を仮定し、共分散行列からステアリングベクトルを推定する
def estimate_steering_vector(Rs):
    """
    Rs: 共分散行列 (num_sources, freq_bins, num_microphones, num_microphones)
    """
    # 固有値分解を実施して最大固有値を与える固有ベクトルを取得
    w, v = np.linalg.eigh(Rs)
    """w: (num_sources, freq_bins, num_microphones), v: (num_sources, freq_bins, num_microphones, num_microphones)"""
    steering_vector = v[..., -1]
    """steering_vector: (num_sources, freq_bins, num_microphones)"""
    return steering_vector

# 遅延和ビームフォーマ
def ds_beamformer(stft_data, steering_vectors):
    """
    stft_data: (num_microphones, freq_bins, time_frames)
    steering_vectors: (freq_bins, num_sources, num_microphones)
    """
    # 遅延和アレイを実行する
    s_hat = np.einsum("ksm,mkt->skt", np.conjugate(steering_vectors), stft_data)
    """s_hat: (num_sources, freq_bins, time_frames)"""
    # ステアリングベクトルをかける（マイクロホン入力信号中の目的音成分を推定）
    c_hat = np.einsum("skt,ksm->mskt", s_hat, steering_vectors)
    """c_hat: (num_microphones, num_sources, freq_bins, time_frames)"""
    return c_hat

# 最小分散無歪応答ビームフォーマ
def mvdr_beamformer(stft_data, steering_vectors):
    """
    stft_data: (num_microphones, freq_bins, time_frames)
    steering_vectors: (freq_bins, num_sources, num_microphones)
    """
    # 共分散行列を計算する
    Rcov =  np.einsum("mkt,nkt->kmn", stft_data, np.conjugate(stft_data))
    """Rcov: (freq_bins, num_microphones, num_microphones)"""
    # 共分散行列の逆行列を計算する
    Rcov_inverse = np.linalg.pinv(Rcov)
    """Rcov_inverse: (freq_bins, num_microphones, num_microphones)"""
    # 分離フィルタを計算する
    Rcov_inverse_a = np.einsum("kmn,kn->km", Rcov_inverse, steering_vectors[:, 0, :]) # 分子
    """Rcov_inverse_a: (freq_bins, num_microphones)"""
    a_H_Rcov_inverse_a = np.einsum("kn,kn->k", np.conjugate(steering_vectors[:, 0, :]), Rcov_inverse_a) # 分母
    """a_H_Rcov_inverse_a: (freq_bins,)"""
    w_mvdr = Rcov_inverse_a / np.maximum(a_H_Rcov_inverse_a, 1.e-18)[:, None]
    """w_mvdr: (freq_bins, num_microphones)"""
    # 分離フィルタを掛ける
    s_hat = np.einsum("km,mkt->kt", np.conjugate(w_mvdr), stft_data)
    """s_hat: (freq_bins, time_frames)"""
    # ステアリングベクトルを掛ける（マイクロホン入力信号中の目的音成分を推定）
    c_hat = np.einsum("kt,km->mkt", s_hat, steering_vectors[:, 0, :])
    """c_hat: (num_microphones, freq_bins, time_frames)"""
    return c_hat

# 最小分散無歪応答ビームフォーマ（共分散行列のみから計算）
def mvdr_beamformer_new(stft_data, Rs, Rn):
    """
    stft_data: マイクロホン入力信号 (num_microphones, freq_bins, time_frames)
    Rs: 目的音の共分散行列 (num_sources, freq_bins, num_microphones, num_microphones)
    Rn: 雑音の共分散行列 (num_sources, freq_bins, num_microphones, num_microphones)
    """
    # 共分散行列の逆行列を計算する
    Rn_inverse = np.linalg.pinv(Rn)
    """Rn_inverse: (num_sources, freq_bins, num_microphones, num_microphones)"""
    # フィルタを計算する
    Rn_inverse_Rs = np.einsum("skmi,skin->skmn", Rn_inverse, Rs)
    """Rn_inverse_Rs: (num_sources, freq_bins, num_microphones, num_microphones)"""
    w_mvdr = Rn_inverse_Rs / np.maximum(np.trace(Rn_inverse_Rs, axis1=-2, axis2=-1), 1.e-18)[..., None, None]
    """w_mvdr: (num_sources, freq_bins, num_microphones, num_microphones)"""
    # フィルタを掛ける
    c_hat = np.einsum("skmn,mkt->nskt", np.conjugate(w_mvdr), stft_data)
    """c_hat: (num_microphones, num_sources, freq_bins, time_frames)"""
    return c_hat

# Max-SNR（GEV）ビームフォーマ
def max_snr_beamformer(stft_data, Rs, Rn):
    """
    stft_data: マイクロホン入力信号 (num_microphones, freq_bins, time_frames)
    Rs: 目的音の共分散行列 (num_sources, freq_bins, num_microphones, num_microphones)
    Rn: 雑音の共分散行列 (num_sources, freq_bins, num_microphones, num_microphones)
    """
    # 音源数を取得
    Ns = np.shape(Rs)[0]
    # 周波数の数を取得
    Nk = np.shape(Rs)[1]
    # 一般化固有値分解
    max_snr_filter = None
    max_snr_filter_all = None
    for s in range(int(Ns)):
        for k in range(int(Nk)):
            w, v = scipy.linalg.eigh(Rs[s, k, ...], Rn[s, k, ...])
            """w: (num_microphones, ), v: (num_microphones, num_microphones)"""
            # 最大固有値に対応する固有ベクトルを取得
            if k == 0:
                max_snr_filter = v[None, :, -1]
            else:
                max_snr_filter = np.concatenate((max_snr_filter, v[None, :, -1]), axis=0)
        if s == 0:
            max_snr_filter_all = max_snr_filter[None, ...]
        else:
            max_snr_filter_all = np.concatenate((max_snr_filter_all, max_snr_filter[None, ...]), axis=0)
    """max_snr_filter_all: (num_sources, freq_bins, num_microphones)"""
    Rs_w = np.einsum("skmn,skn->skm", Rs, max_snr_filter_all)
    """Rs_w: (num_sources, freq_bins, num_microphones)"""
    beta = Rs_w / np.einsum("skm,skm->sk", np.conjugate(max_snr_filter_all), Rs_w)[:, :, None]
    """beta: (num_sources, freq_bins, num_microphones)"""
    w_max_snr = beta[:, :, None, :] * max_snr_filter_all[:, :, :, None]
    """w_max_snr: (num_sources, freq_bins, num_microphones, num_microphones)"""
    # フィルタを掛ける
    c_hat = np.einsum("skim,ikt->mskt", np.conjugate(w_max_snr), stft_data)
    """c_hat: (num_microphones, num_sources, freq_bins, time_frames)"""
    return c_hat

# MUSIC法を用いた音源定位
def localize_music(spec, mic_alignments, sample_rate, fft_size, freq_range=[200, 3000]):
    """
    spec: (num_channels, freq_bins, time_frames)
    mic_alignments: (3D coordinates [m], num_microphones)
    """
    # MUSIC法を用いて音源定位（cは音速）
    doa = pa.doa.algorithms['MUSIC'](mic_alignments, sample_rate, fft_size, c=343., num_src=1) # Construct the new DOA object
    doa.locate_sources(spec, freq_range=freq_range)
    speaker_azimuth = doa.azimuth_recon / np.pi * 180.0 # rad→deg
    # 0°〜360°表記を-180°〜180°表記に変更
    if speaker_azimuth[0] > 180:
        speaker_azimuth = int(speaker_azimuth[0] - 360)
    else:
        speaker_azimuth = int(speaker_azimuth[0])
    return speaker_azimuth

# マスク推定
# Ideal Ratio Maskを算出
def calc_ideal_ratio_mask(self, target_spec, noise_spec):
    """
    target_spec: (freq_bins=257, time_steps=513)
    noise_spec: (freq_bins=257, time_steps=513)
    """
    # 参考：「https://gist.github.com/jonashaag/677e1ddab99f3daba367de9ec022e942#file-cirm-py-L39」
    # 0除算を避ける
    target_IRM = np.sqrt(target_spec ** 2 / np.maximum((target_spec ** 2 + noise_spec ** 2), 1e-7))
    noise_IRM = np.sqrt(noise_spec ** 2 / np.maximum((target_spec ** 2 + noise_spec ** 2), 1e-7))
    return target_IRM, noise_IRM


"""
自然言語処理用
"""
# Mecabを用いた形態素解析(Morphological Analysis)
def mecab_wakati(text):
    import MeCab
    import re
    import mojimoji
    import emoji # ver=1.7.0でないとエラーが出る可能性あり
    # MeCabで分かち書き -dに辞書を指定
    tagger = MeCab.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    text = tagger.parse(text)
    # 半角全角数字の除去
    text = re.sub(r'[0-9０-９]+', " ", text)
    # 記号の除去
    text = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", text)
    # 絵文字の削除
    text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)
    # URLの削除
    text = re.sub('https?://[\da-zA-Z!\?/\+\-_~=;\.,\*&@#\$%\(\)\'\[\]]+', '', text)
    # 全角から半角に変換（カナは除く）
    result = mojimoji.zen_to_han(text, kana=False)
    # 半角カナから全角カナに変換
    result = mojimoji.han_to_zen(result, ascii=False)
    # 全ての文字を小文字に変換
    result = result.lower()
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
    # sample_rate = 16000
    # data = load_audio_file(file_path, audio_length, sample_rate)
    # play_audio(data, sample_rate)

    # # 音声録音用
    # import sounddevice as sd
    # audio_length = 5
    # sample_rate = 16000
    # channels = 8
    # save_path = "./test1103.wav"
    # rec_audio(audio_length, sample_rate, channels, save_path)

    # file_path = "./test/target_voice.wav"
    # save_path = "./test/istft.wav"
    # audio_length = 3
    # sample_rate = 16000
    # n_fft=1024
    # hop_length=768
    # audio_data = load_audio_file(file_path, audio_length, sample_rate)
    # mag, phase = wave_to_spec(audio_data, n_fft, hop_length)
    # target_spec = mag * phase
    # istft_data = spec_to_wave(target_spec, hop_length)
    # save_audio_file(save_path, istft_data, sample_rate)

    # fft_size = 1024
    # hop_length = 768
    # file_path = "../AudioDatasets/NoisySpeechDetabase/clean_trainset_28spk_wav_16kHz/p226_001.wav"
    # audio_length = 3
    # sample_rate = 16000
    # audio_data = load_audio_file(file_path, audio_length, sample_rate)
    # amp, phase = wave_to_spec(audio_data, fft_size, hop_length, win_length=None)
    # print(amp.shape)

    input_path = "../AudioDatasets/NoisySpeechDatabase/clean_testset_wav_16kHz_original_length/p232_001.wav"
    output_path = "./test.png"
    audio_length = 30
    wave_plot(input_path, output_path, audio_length, fig_title=None, ylim_min=-0.01, ylim_max=0.01)

    # # ビームフォーミング用
    # # 周波数の数
    # fft_size = 512
    # hop_length = 160
    # Nk = fft_size / 2 + 1
    # # サンプリングレート [Hz]
    # sample_rate = 16000
    # # 各ビンの周波数
    # freqs = np.arange(0, Nk, 1) * sample_rate / fft_size
    # # 音源とマイクロホンの距離 [m]
    # distance_mic_to_source=2. 
    # # 音源方向（音源が複数ある場合はリストに追加）
    # azimuth = [0] # 方位角
    # elevation = [np.pi/6] # 仰角
    # # 部屋（シミュレーション環境）の設定
    # room_width = 5.0
    # room_length = 5.0
    # room_height = 5.0
    # # マイクロホンアレイの中心位置
    # nakbot_height = 0.57 # Nakbotの全長
    # mic_array_height = nakbot_height - 0.04 # 0.04はTAMAGO-03マイクロホンアレイの頂上部からマイクロホンアレイ中心までの距離
    # mic_array_loc = np.r_[room_width/2, room_length/2, 0] + [0, 0, mic_array_height] # 部屋の中央に配置されたNakbot上のマイクロホンアレイ
    # print("マイクロホンアレイ中心座標：", mic_array_loc)
    # # TAMAGO-03のマイクロホンアレイのマイクロホン配置（単位はm）
    # mic_alignments = np.array(
    # [
    #     [0.035, 0.0, 0.0],
    #     [0.035/np.sqrt(2), 0.035/np.sqrt(2), 0.0],
    #     [0.0, 0.035, 0.0],
    #     [-0.035/np.sqrt(2), 0.035/np.sqrt(2), 0.0],
    #     [-0.035, 0.0, 0.0],
    #     [-0.035/np.sqrt(2), -0.035/np.sqrt(2), 0.0],
    #     [0.0, -0.035, 0.0],
    #     [0.035/np.sqrt(2), -0.035/np.sqrt(2), 0.0]
    # ])
    # n_channels = np.shape(mic_alignments)[0]
    # print("マイクロホン数：", n_channels)
    # # get the microphone array （各マイクロホンの空間的な座標）
    # R = mic_alignments.T + mic_array_loc[:, None]
    # """R: (3D coordinates [m], num_microphones)"""
    # # 音源の位置（HARK座標系に対応） [仰角θ, 方位角φ]
    # doas = np.array(
    # [[elevation[0], azimuth[0]], # １個目の音源 
    # # [elevation[1], azimuth[1]] # ２個目の音源
    # ])
    # source_locations = np.zeros((3, np.shape(doas)[0]), dtype=doas.dtype)
    # """source_locations: (xyz, num_sources)"""
    # source_locations[0,  :] = np.cos(doas[:, 1]) * np.cos(doas[:, 0]) # x = rcosφcosθ
    # source_locations[1,  :] = np.sin(doas[:, 1]) * np.cos(doas[:, 0]) # y = rsinφcosθ
    # source_locations[2,  :] = np.sin(doas[:, 0]) # z = rsinθ
    # source_locations *= distance_mic_to_source
    # source_locations += mic_array_loc[:, None] # マイクロホンアレイからの相対位置→絶対位置
    # audio_path = "./audio_separation/data/multichannel_audio_for_test/p232_007_target.wav"
    # multi_audio_data = load_audio_file(audio_path, length=3, sample_rate=sample_rate)
    # multi_amp_spec, _ = wave_to_spec_multi(multi_audio_data.T, sample_rate=sample_rate, fft_size=fft_size, hop_length=hop_length)
    # steering_vectors = calculate_steering_vector(R, source_locations, freqs, sound_speed=340, is_use_far=False)
    # mvdr_output = mvdr_beamformer(multi_amp_spec, steering_vectors)
    # print(mvdr_output.shape)

#     # 音声認識動作テスト
#     # 英語版
#     lang = 'eng'
#     audio_path = "../sample_audio/eng/estimated_voice.wav"
#     # 日本語版
# #     lang = 'jp'
# #     audio_path = "../sample_audio/jp/BASIC5000_0001.wav"
#     asr_ins = ASR(lang)
#     result = asr_ins.speech_recognition(audio_path)
#     print("recognition_result:", result)

