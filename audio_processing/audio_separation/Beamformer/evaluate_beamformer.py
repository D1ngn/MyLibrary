import os
import sys
import glob
import time
import numpy as np
import scipy
import scipy.signal as signal
import librosa

from tqdm import tqdm
from natsort import natsorted

sys.path.append('../../..')
from MyLibrary.MyFunc import load_audio_file, wave_to_spec_multi, calculate_steering_vector, wave_plot, audio_eval, load_audio_file, save_audio_file, wave_to_spec, spec_to_wave, estimate_mask, estimate_covariance_matrix, estimate_steering_vector


def modify_angle_diff(diff):
    diff = np.where(diff < -np.pi, diff + np.pi * 2, diff)
    diff = np.where(diff > np.pi, diff - np.pi * 2, diff)
    return diff

# 遅延和ビームフォーマ
def ds_beamformer(stft_data, steering_vector):
    """
    stft_data: (num_microphones, freq_bins, time_frames)
    steering_vector: (num_sources, freq_bins, num_microphones)
    """
    s_hat = np.einsum("skm,mkt->skt", np.conjugate(steering_vector), stft_data)
    """s_hat: (num_sources, freq_bins, time_frames)"""
    # ステアリングベクトルを掛ける
    c_hat = np.einsum("skt,skm->mskt", s_hat, steering_vector)
    """c_hat: (num_microphones, num_sources, freq_bins, time_frames)"""
    return c_hat

# 最小分散無歪応答ビームフォーマ（最尤ビームフォーマ）
def mvdr_beamformer(stft_data, Rn, steering_vector):
    """
    stft_data: (num_microphones, freq_bins, time_frames)
    Rn: (num_sources, freq_bins, num_microphones, num_microphones)
    steering_vector: (num_sources, freq_bins, num_microphones)
    """
    # 共分散行列の逆行列を計算する
    Rn_inverse = np.linalg.pinv(Rn)
    """Rn_inverse: (num_sources, freq_bins, num_microphones, num_microphones)"""
    # 分離フィルタを計算する
    Rn_inverse_a = np.einsum("skmn,skn->skm", Rn_inverse, steering_vector) # 分子
    """Rn_inverse_a: (num_sources, freq_bins, num_microphones)"""
    a_H_Rn_inverse_a = np.einsum("skn,skn->sk", np.conjugate(steering_vector), Rn_inverse_a) # 分母
    """a_H_Rn_inverse_a: (num_sources, freq_bins)"""
    w_mvdr = Rn_inverse_a / np.maximum(a_H_Rn_inverse_a, 1.e-18)[:, :, None]
    """w_mvdr: (num_sources, freq_bins, num_microphones)"""
    # 分離フィルタを掛ける
    s_hat = np.einsum("skm,mkt->skt", np.conjugate(w_mvdr), stft_data)
    """s_hat: (num_sources, freq_bins, time_frames)"""
    # ステアリングベクトルを掛ける（マイクロホン入力信号中の目的音成分を推定）
    c_hat = np.einsum("skt,skm->mskt", s_hat, steering_vector)
    """c_hat: (num_microphones, num_sources, freq_bins, time_frames)"""
    return c_hat

# 最小分散無歪応答ビームフォーマ（共分散行列のみから計算）
def mvdr_beamformer_new(stft_data, Rs, Rn):
    """
    stft_data: (num_microphones, freq_bins, time_frames)
    Rs: (num_sources, freq_bins, num_microphones, num_microphones)
    Rn: (num_sources, freq_bins, num_microphones, num_microphones)
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

# MaxSNR
def max_snr_beamformer(stft_data, Rs, Rn):
    """
    stft_data: (num_microphones, freq_bins, time_frames)
    Rs: (num_sources, freq_bins, num_microphones, num_microphones)
    Rn: (num_sources, freq_bins, num_microphones, num_microphones)
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

# MWFを実行
def mwf(stft_data, Rs, Rn):
    """
    stft_data: (num_microphones, freq_bins, time_frames)
    Rs: (num_sources, freq_bins, num_microphones, num_microphones)
    Rn: (num_sources, freq_bins, num_microphones, num_microphones)
    """
    # 入力信号に対する共分散行列の逆行列を計算
    Rx_inverse = np.linalg.pinv(Rs + Rn)
    """Rx_inverse: (num_sources, freq_bins, num_microphones, num_microphones)"""
    # フィルタ生成
    W_mwf = np.einsum("skmi,skin->skmn", Rx_inverse, Rs)
    """W_mwf: (num_sources, freq_bins, num_microphones, num_microphones)"""
    # フィルタを掛ける
    c_hat = np.einsum("skim,ikt->mskt", np.conjugate(W_mwf), stft_data)
    """c_hat: (num_microphones, num_sources, freq_bins, time_frames)"""
    return c_hat

if __name__ == "__main__":
    # ビームフォーミング用
    # 音声の長さ[sec]
    audio_length = 3
    # 周波数の数
    fft_size = 512
    Nk = fft_size / 2 + 1
    # 短時間フーリエ変換におけるフレームにずらし幅
    hop_length = 160
    # サンプリングレート [Hz]
    sample_rate = 16000
    # 各ビンの周波数
    freqs = np.arange(0, Nk, 1) * sample_rate / fft_size
    # 方位角の閾値
    azimuth_thresh = 30
    # 音声ファイルのディレクトリを指定
    test_data_dir = "../data/multichannel_audio_for_test/"
    # test_data_dir = "../data/NoisySpeechDataset_for_unet_fft_512_8ch_interference80percent_1113/test"
    mixed_audio_path_list = natsorted(glob.glob(os.path.join(test_data_dir, "*_mixed.wav")))
    ######################シミュレーション環境の設定#########################
    # 音源とマイクロホンの距離 [m]
    distance_mic_to_source = 2. 
    # 音源方向（音源が複数ある場合はリストに追加）
    azimuth = [0] # 方位角
    elevation = [np.pi/6] # 仰角
    # 部屋（シミュレーション環境）の設定
    room_width = 5.0
    room_length = 5.0
    room_height = 5.0
    # マイクロホンアレイの中心位置
    nakbot_height = 0.57 # Nakbotの全長
    mic_array_height = nakbot_height - 0.04 # 0.04はTAMAGO-03マイクロホンアレイの頂上部からマイクロホンアレイ中心までの距離
    mic_array_loc = np.r_[room_width/2, room_length/2, 0] + [0, 0, mic_array_height] # 部屋の中央に配置されたNakbot上のマイクロホンアレイ
    print("マイクロホンアレイ中心座標：", mic_array_loc)
    # TAMAGO-03のマイクロホンアレイのマイクロホン配置（単位はm）
    mic_alignments = np.array(
    [
        [0.035, 0.0, 0.0],
        [0.035/np.sqrt(2), 0.035/np.sqrt(2), 0.0],
        [0.0, 0.035, 0.0],
        [-0.035/np.sqrt(2), 0.035/np.sqrt(2), 0.0],
        [-0.035, 0.0, 0.0],
        [-0.035/np.sqrt(2), -0.035/np.sqrt(2), 0.0],
        [0.0, -0.035, 0.0],
        [0.035/np.sqrt(2), -0.035/np.sqrt(2), 0.0]
    ])
    n_channels = np.shape(mic_alignments)[0]
    print("マイクロホン数：", n_channels)
    # get the microphone array （各マイクロホンの空間的な座標）
    R = mic_alignments.T + mic_array_loc[:, None]
    """R: (3D coordinates [m], num_microphones)"""
    # 音源の位置（HARK座標系に対応） [仰角θ, 方位角φ]
    doas = np.array(
    [[elevation[0], azimuth[0]], # １個目の音源 
    # [elevation[1], azimuth[1]] # ２個目の音源
    ])
    # 音源の数
    n_sim_sources = np.shape(doas)[0]
    source_locations = np.zeros((3, np.shape(doas)[0]), dtype=doas.dtype)
    """source_locations: (xyz, num_sources)"""
    source_locations[0,  :] = np.cos(doas[:, 1]) * np.cos(doas[:, 0]) # x = rcosφcosθ
    source_locations[1,  :] = np.sin(doas[:, 1]) * np.cos(doas[:, 0]) # y = rsinφcosθ
    source_locations[2,  :] = np.sin(doas[:, 0]) # z = rsinθ
    source_locations *= distance_mic_to_source
    source_locations += mic_array_loc[:, None] # マイクロホンアレイからの相対位置→絶対位置

    # ステアリングベクトルを算出するための仮想的な音源方向
    virtual_doas = np.array(
        [[np.pi/2, theta/180 * np.pi] for theta in np.arange(0, 360, 5)]
    )
    virtual_source_locations = np.zeros((3, np.shape(virtual_doas)[0]), dtype=virtual_doas.dtype)
    """virtual_source_locations: (xyz, num_sources)"""
    virtual_source_locations[0,  :] = np.cos(virtual_doas[:, 1]) * np.sin(virtual_doas[:, 0]) 
    virtual_source_locations[1,  :] = np.sin(virtual_doas[:, 1]) * np.sin(virtual_doas[:, 0])
    virtual_source_locations[2,  :] = np.cos(virtual_doas[:, 0])
    virtual_source_locations *= 100
    virtual_source_locations += mic_array_loc[:, None] # マイクロホンアレイからの相対位置→絶対位置
    # 仮想的な音源方向（0°, 5°,・・・, 355°）のステアリングベクトル作成
    virtual_steering_vectors = calculate_steering_vector(R, virtual_source_locations, freqs, is_use_far=True)
    """virtual_steering_vectors: (freq_bins, num_virtual_sources=72, num_microphones)"""

    # 所望音の方向から±thresh度以内
    omega = np.array([np.abs(modify_angle_diff(virtual_doas[:, 1] - doas[s, 1])) < azimuth_thresh / 180 * np.pi for s in range(n_sim_sources)]).astype(np.float)
    """omega: (n_sources, num_virtual_sources=72)"""

    ######################雑音除去＋音声評価#########################
    # 音声評価結果の合計値を格納するリストを用意
    sdr_mix_list = []
    sir_mix_list = []
    sar_mix_list = []
    sdr_est_list = []
    sir_est_list = []
    sar_est_list = []
    # 合計処理時間を測るための変数を用意
    processing_duration_sum = 0
    
    for mixed_audio_path in tqdm(mixed_audio_path_list):
        # 音声データをロード
        mixed_audio_data = load_audio_file(mixed_audio_path, audio_length, sample_rate)
        mixed_audio_data = np.require(mixed_audio_data, dtype=np.float32, requirements=['F']) # Fortran-contiguousに変換（これがないとエラーが出る）
        """mixed_audio_data: (num_samples, num_channels=8)"""
        # 処理の開始時間
        iter_start_time = time.perf_counter()
        # ビームフォーマによる雑音除去処理
        # 短時間フーリエ変換
        # f, t, stft_data = signal.stft(mixed_audio_data.T, fs=sample_rate, window="hann", nperseg=fft_size)
        # """f: (freq_bins,), t: (1,), stft_data:(num_microphones, freq_bins, time_frames)"""
        multi_complex_spec = [] # それぞれのチャンネルの複素スペクトログラムを格納するリスト
        for i in range(mixed_audio_data.shape[1]):
            # オーディオデータをスペクトログラムに変換
            complex_spec = librosa.stft(mixed_audio_data[:, i], n_fft=fft_size, hop_length=hop_length, win_length=None, window='hann')
            multi_complex_spec.append(complex_spec)
        multi_complex_spec = np.array(multi_complex_spec)
        """multi_complex_spec: (num_microphones, freq_bins, time_steps)"""
        # 時間周波数マスクを推定→ニューラルネットワークに置き換えるのもあり
        tf_mask = estimate_mask(multi_complex_spec, virtual_steering_vectors, omega)
        """tf_mask: (num_sources, freq_bins, time_frames)"""
        # 共分散行列とステアリングベクトルを推定
        Rs, Rn = estimate_covariance_matrix(multi_complex_spec, tf_mask)
        """Rs or Rn: (num_sources, freq_bins, num_microphones, num_microphones)"""
        desired_steering_vectors = estimate_steering_vector(Rs)
        """desired_steering_vectors: (num_sources, freq_bins, num_microphones)"""
        # ビームフォーマを実行
        # ds_out = ds_beamformer(multi_complex_spec, desired_steering_vectors)
        mvdr_out = mvdr_beamformer(multi_complex_spec, Rn, desired_steering_vectors)
        # mvdr_new_out = mvdr_beamformer_new(multi_complex_spec, Rs, Rn)
        # max_snr_out = max_snr_beamformer(multi_complex_spec, Rs, Rn)
        # mwf_out = mwf(multi_complex_spec, Rs, Rn)
        """mvdr_out: (num_microphones, num_sources, freq_bins, time_frames)"""
        # 時間領域の波形に変換
        # t, ds_out = signal.istft(ds_out, fs=sample_rate, window="hann", nperseg=fft_size)
        # """dsbf_out: (num_microphones, num_sources, num_samples)"""
        # マルチチャンネルスペクトログラムを音声波形に変換
        multichannel_estimated_voice_data= np.zeros(mixed_audio_data.shape, dtype='float64') # マルチチャンネル音声波形を格納する配列
        # 1chごとスペクトログラムを音声波形に変換
        for i in range(mvdr_out.shape[0]):
            estimated_voice_data = librosa.istft(mvdr_out[i, 0, :, :], hop_length=hop_length) # 1つ目の音源（目的音）を選択
            multichannel_estimated_voice_data[:, i] = estimated_voice_data 
        """multichannel_estimated_voice_data: (num_samples, num_channels)"""
        # 処理の終了時間
        iter_finish_time = time.perf_counter()
        # 1ループ当たりの処理時間（音声波形→STFT→雑音除去→iSTFT→音声波形）
        iter_processing_duration = iter_finish_time - iter_start_time
        processing_duration_sum += iter_processing_duration
        # オーディオデータを保存
        # ds_out = ds_out[:, 0, :].T # 1つ目の音源（目的音）を選択し、保存できる形に転置
        # """ds_out: (num_samples, num_microphones)"""
        estimated_voice_path = "./estimated_voice.wav"
        save_audio_file(estimated_voice_path, multichannel_estimated_voice_data, sample_rate)
        file_num = os.path.basename(mixed_audio_path).split('.')[0].rsplit('_', maxsplit=1)[0] # p257_013
        target_voice_path = os.path.join(test_data_dir, file_num + "_target.wav")
        interference_audio_path = os.path.join(test_data_dir, file_num + "_interference.wav")
        # 音声評価
        sdr_mix, sir_mix, sar_mix, sdr_est, sir_est, sar_est = audio_eval(audio_length, sample_rate, \
        target_voice_path, interference_audio_path, mixed_audio_path, estimated_voice_path)
        # 音声評価結果を記録
        sdr_mix_list.append(sdr_mix)
        sir_mix_list.append(sir_mix)
        sar_mix_list.append(sar_mix)
        sdr_est_list.append(sdr_est)
        sir_est_list.append(sir_est)
        sar_est_list.append(sar_est)
        # 推定音声が蓄積されないように削除
        os.remove(estimated_voice_path)

    # データの数を取得
    num_file = len(mixed_audio_path_list)
    print("合計処理時間：", str(processing_duration_sum) + 'sec')
    print("平均処理時間：", str(processing_duration_sum/num_file) + 'sec')
    print("平均 | SDR_mix: {:.3f}, SIR_mix: {:.3f}, SAR_mix: {:.3f}".format(np.mean(sdr_mix_list), np.mean(sir_mix_list), np.mean(sar_mix_list)))
    print("平均 | SDR_est: {:.3f}, SIR_est: {:.3f}, SAR_est: {:.3f}".format(np.mean(sdr_est_list), np.mean(sir_est_list), np.mean(sar_est_list)))
    print("標準偏差 | SDR_mix: {:.3f}, SIR_mix: {:.3f}, SAR_mix: {:.3f}".format(np.std(sdr_mix_list), np.std(sir_mix_list), np.std(sar_mix_list)))
    print("標準偏差 | SDR_est: {:.3f}, SIR_est: {:.3f}, SAR_est: {:.3f}".format(np.std(sdr_est_list), np.std(sir_est_list), np.std(sar_est_list)))