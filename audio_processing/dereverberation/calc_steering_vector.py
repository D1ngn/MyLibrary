import numpy as np

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

if __name__ == "__main__":
    # サンプリングレート [Hz]
    sample_rate = 16000
    # フレームサイズ
    N = 1024
    # 周波数の数
    Nk = N / 2 + 1
    # 各ビンの周波数
    freqs = np.arange(0, Nk, 1) * sample_rate / N
    # マイクロホンアレイのマイクロホン配置
    mic_alignments = np.array(
        [
            [-0.01, 0.0, 0.0],
            [0.01, 0.0, 0.0],
        ]
    ).T
    # 音源の方向
    doas = np.array(
        [[np.pi/2, 0],
        [np.pi/2, np.pi]
        ])
    # 音源とマイクロホンの距離
    distance=1.0
    # 音源の位置ベクトル
    source_locations = np.zeros((3, doas.shape[0]), dtype=doas.dtype)
    source_locations[0, :] = np.cos(doas[:, 1]) * np.sin(doas[:, 0])
    source_locations[1, :] = np.sin(doas[:, 1]) * np.sin(doas[:, 0])
    source_locations[2, :] = np.cos(doas[:, 0])
    source_locations *= distance

    # Near-field仮定に基づくステアリングベクトルを計算
    near_steering_vectors = calculate_steering_vector(mic_alignments, source_locations, freqs, is_use_far=False)
    # Far-field仮定に基づくステアリングベクトルを計算
    far_steering_vectors = calculate_steering_vector(mic_alignments, source_locations, freqs, is_use_far=True)

    # 内積を計算（Near-field仮定とFar-field仮定それぞれで求めたステアリングベクトルにどの程度違いがあるか評価）
    inner_product = np.einsum('ksm,ksm->ks', np.conjugate(near_steering_vectors), far_steering_vectors)
    print(np.average(np.abs(inner_product)))