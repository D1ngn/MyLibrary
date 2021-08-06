# 必要ライブラリのインポート
import os
import numpy as np
import soundfile as sf
import pyroomacoustics as pa
import glob
import random

from tqdm import tqdm

# 乱数を初期化
random.seed(0)

# 音声に室内インパルス応答（Room Impulse Response）を畳み込んで空間化
def rir_convolve(wave_files, sample_rate, audio_length, doas, distance_mic_to_source, \
                 mic_array_loc, R, room_dim, max_order, absorption, SNR):
    """
    wave_files: シングルチャンネルの音声のパスを格納したリスト
    sample_rate: サンプリング周波数 [Hz]
    audio_length: 音声の長さ [sec]
    doas: 音源の到来方向
    distance_mic_to_source: 音源とマイクロホンの距離 [m]
    mic_array_loc: マイクロホンアレイの位置座標
    R: 各マイクロホンの空間的な座標
    room_dim: 部屋の３次元形状を表す（単位はm）
    max_order: 部屋の壁で何回音が反射するか（反射しない場合0）
    absorption: 部屋の壁でどの程度音が吸収されるか （吸収されない場合None）
    SNR: 音声と雑音の比率 [dB]
    """
    n_sources = len(wave_files)
#     print("音源数:", n_sources)
    source_locations = np.zeros((3, doas.shape[0]), dtype=doas.dtype)
    """source_locations: (xyz, num_sources)"""
    
    source_locations[0,  :] = np.cos(doas[:, 1]) * np.cos(doas[:, 0]) # x = rcosφcosθ
    source_locations[1,  :] = np.sin(doas[:, 1]) * np.cos(doas[:, 0]) # y = rsinφcosθ
    source_locations[2,  :] = np.sin(doas[:, 0]) # z = rsinθ
    source_locations *= distance_mic_to_source
    source_locations += mic_array_loc[:, None] # マイクロホンアレイからの相対位置→絶対位置
    for i in range(n_sources):
        x = source_locations[0, i]
        y = source_locations[1, i]
        z = source_locations[2, i]
#         print("{}個目の音源の位置： (x, y, z) = ({}, {}, {})".format(i+1, x, y, z))

    # 音源数分の音声ファイルを読み込む（音声の長さ自由版）
    for s, wave_file in enumerate(wave_files):
        audio_data, _ = sf.read(wave_file)
        if s == 0:
            clean_data = audio_data[np.newaxis, :]
        else:
            clean_data = np.append(clean_data, audio_data[np.newaxis, :], axis=0)
    """clean_data: (num_sources, num_samples)"""
        
    # 部屋を生成する
    room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=max_order, absorption=absorption)
    # 用いるマイクロホンアレイの情報を設定する
    room.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))
    # 各音源をシミュレーションに追加する
    for s in range(n_sources):
        clean_data[s] /= np.std(clean_data[s])
        # たまに「ValueError: The source must be added inside the room.」が出る
        room.add_source(source_locations[:, s], signal=clean_data[s])
    # RIRのシミュレーション生成と音源信号への畳み込みを実行
    room.simulate(snr=SNR)
    
#     # 残響時間（RT60）を知りたい場合
#     impulse_responses = room.rir
#     rt60 = pa.experimental.measure_rt60(impulse_responses[0][0], fs=sample_rate)
#     print("残響時間:{} [sec]".format(rt60))

    # 室内インパルス応答を畳み込んだ波形データを取得
    convolved_wave = room.mic_array.signals.T
    """convolved_wave: (num_samples, num_channels)"""

    return convolved_wave


if __name__ == '__main__':
    # 各パラメータを設定
    sample_rate = 16000 # 作成するオーディオファイルのサンプリング周波数を指定
    audio_length = 3 # 単位は秒(second) → fft_size=1024,hop_length=768のとき、audio_length=6が最適化かも？
    train_val_ratio = 0.9 # trainデータとvalidationデータの割合
    fft_size = 512 # 短時間フーリエ変換のフレーム長
    hop_length = 160 # 短時間フーリエ変換においてフレームをスライドさせる幅
    gain_decay = 0.8 # 音量調整のためのパラメータ（雑音が大きすぎるため）
    
    # RIR生成用のパラメータ
    # 音声と雑音の比率 [dB]
    SNR = None
    # 音源とマイクロホンの距離 [m]
    distance_mic_to_source=2
    # 部屋（シミュレーション環境）の設定
    room_width = 5.0
    room_length = 5.0
    room_height = 5.0
    # 部屋の残響を設定
    max_order = 0 # 部屋の壁で何回音が反射するか（反射しない場合0）
    absorption = None # 部屋の壁でどの程度音が吸収されるか （吸収されない場合None）
    # Nakbot上に載せたTAMAGO-03マイクロホンアレイで取得した音声をシミュレートする場合、以下は固定
    # 部屋の３次元形状を表す（単位はm）
    room_dim = np.r_[room_width, room_length, room_height]
    print("部屋の3次元形状：", room_dim)
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
    # 各マイクロホンの空間的な座標
    R = mic_alignments.T + mic_array_loc[:, None]
    """R: (3D coordinates [m], num_microphones)"""
    
    # データセットを格納するディレクトリを作成
    save_dataset_dir = "./save_dir/"
    os.makedirs(save_dataset_dir, exist_ok=True)

    # 学習・評価用
    # 目的音が格納されたディレクトリを指定
    target_data_dir = "./sample_audio/jp/target/"
    # 雑音が格納されたディレクトリを指定
    noise_data_dir = "./sample_audio/jp/noise/"
    # 目的音のリスト
    target_data_path_template = os.path.join(target_data_dir, "*.wav")
    target_list = glob.glob(target_data_path_template)
    # データセットをシャッフル
    random.shuffle(target_list)

    # 目的音と雑音、混合音をシミュレーションによって作成
    for idx, target_path in enumerate(tqdm(target_list)):
        file_num = os.path.basename(target_path).split('.')[0] # (例) p232_001
        target_file_name = file_num + "_target.wav" # (例) p232_001_target.wav
        
        # 音声にRIRを畳み込みながらマルチチャンネルに拡張
        # 目的音の畳み込み（評価用）
        # 音源方向（音源が複数ある場合はリストに追加、目的音の音源方向は固定）
        azimuth = [0] # 方位角（1個目の音源, 2個目の音源）
        elevation = [np.pi/6] # 仰角（1個目の音源, 2個目の音源）
        # 音源の位置（HARK座標系に対応） [仰角θ, 方位角φ]
        doas = np.array(
        [[elevation[0], azimuth[0]], # １個目の音源 
#         [elevation[1], azimuth[1]] # ２個目の音源
        ])
        convolved_target_data = rir_convolve([target_path], sample_rate, audio_length, doas, distance_mic_to_source, \
                 mic_array_loc, R, room_dim, max_order=0, absorption=None, SNR=None)
        # RIRの長さ-1サンプル分オーディオデータが長くなるので、元に戻す
        convolved_target_data = convolved_target_data[:sample_rate*audio_length, :]
        """convolved_target_data: (num_samples, num_channels=8)"""
        
        # 雑音の畳み込み（評価用）
        noise_path = os.path.join(noise_data_dir, file_num + ".wav")
        # 雑音の到来方向を指定（0°, 15°, 30°, 45°, 60°, 75°, 90°の7分割）→自分で自由に変更してください
        noise_azimuth = int(idx / (len(target_list) / 7)) * (np.pi / 12)
        # 音源方向（音源が複数ある場合はリストに追加、目的音の音源方向は固定）
        azimuth = [noise_azimuth] # 方位角（1個目の音源, 2個目の音源）
        elevation = [np.pi/6] # 仰角（1個目の音源, 2個目の音源）
        # 音源の位置（HARK座標系に対応） [仰角θ, 方位角φ]
        doas = np.array(
        [[elevation[0], azimuth[0]], # １個目の音源 
#         [elevation[1], azimuth[1]] # ２個目の音源
        ])
        # 音声にRIRを畳み込みながらマルチチャンネルに拡張
        convolved_noise_data = rir_convolve([noise_path], sample_rate, audio_length, doas, distance_mic_to_source, \
                 mic_array_loc, R, room_dim, max_order=0, absorption=None, SNR=None)
        # RIRの長さ-1サンプル分オーディオデータが長くなるので、元に戻す
        convolved_noise_data = convolved_noise_data[:sample_rate*audio_length, :]
        """convolved_noise_data: (num_samples, num_channels=8)"""
#         convolved_noise_data = convolved_noise_data * gain_decay
        
        # 目的音と雑音を空間的に配置（混合音声が欲しい場合は以下を実行すればよい）
        # 畳み込む音声をリストに格納
        wave_files = [target_path, noise_path]
        # 音源方向（音源が複数ある場合はリストに追加、目的音の音源方向は固定）
        azimuth = [0, noise_azimuth] # 方位角（1個目の音源, 2個目の音源）
        elevation = [np.pi/6, np.pi/6] # 仰角（1個目の音源, 2個目の音源）
        # 音源の位置（HARK座標系に対応） [仰角θ, 方位角φ]
        doas = np.array(
        [[elevation[0], azimuth[0]], # １個目の音源 
        [elevation[1], azimuth[1]] # ２個目の音源
        ])
        # 音声にRIRを畳み込みながらマルチチャンネルに拡張
        convolved_mixed_data = rir_convolve(wave_files, sample_rate, audio_length, doas, distance_mic_to_source, \
                 mic_array_loc, R, room_dim, max_order, absorption, SNR)
        # RIRの長さ-1サンプル分音声データが長くなるので、元に戻す
        convolved_mixed_data = convolved_mixed_data[:sample_rate*audio_length, :]
        """convolved_mixed_data: (num_samples, num_channels=8)"""
    
        # 混合音声の最大振幅で正規化（RIRを畳み込むと音が大きくなるので音割れを防ぐ）
        normalized_convolved_target_data = convolved_target_data / convolved_mixed_data.max()
        normalized_convolved_noise_data = convolved_noise_data / convolved_mixed_data.max()
        normalized_convolved_mixed_data = convolved_mixed_data / convolved_mixed_data.max()

        # 音声を保存
        # 目的音
        target_file_path = os.path.join(save_dataset_dir, target_file_name)
        sf.write(target_file_path, normalized_convolved_target_data, sample_rate)
        # 雑音
        noise_file_name = file_num + "_noise.wav" # (例) p232_001_noise.wav
        noise_file_path = os.path.join(save_dataset_dir, noise_file_name)
        sf.write(noise_file_path, normalized_convolved_noise_data, sample_rate)
        # 混合音声
        mixed_file_name = file_num + "_mixed.wav" # (例) p232_001_mixed.wav
        mixed_file_path = os.path.join(save_dataset_dir, mixed_file_name)
        sf.write(mixed_file_path, normalized_convolved_mixed_data, sample_rate)
        

    print("データ作成完了 保存先：{}".format(save_dataset_dir))