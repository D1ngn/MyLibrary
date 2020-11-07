import numpy as np
import scipy.signal as scipy
import matplotlib.pyplot as plt

from calc_steering_vector import calculate_steering_vector

# サンプリングレート [Hz]
sample_rate = 16000
# フレームサイズ
N = 1024
# 周波数の数
Nk = N / 2 + 1
# 各ビンの周波数
freqs = np.arange(0, Nk, 1) * sample_rate / N

# # マイクロホンアレイのマイクロホン配置（2個）
# mic_alignments = np.array(
#         [[x, 0.0, 0.0] for x in np.arange(-0.01, 0.02, 0.02)]
# )
# マイクロホンアレイのマイクロホン配置（32個）
mic_alignments = np.array(
        [[x, 0.0, 0.0] for x in np.arange(-0.31, 0.32, 0.02)]
)

# マイクロホン数
n_channels = np.shape(mic_alignments)[0]
print(n_channels)

# 音源の場所
doas  = np.array(
    [[np.pi/2, theta] for theta in np.arange(-np.pi, np.pi, 1./180.*np.pi)]
    )

# 音源とマイクロホンの距離
distance = 1.
source_locations = np.zeros((3, doas.shape[0]), dtype=doas.dtype)
"""source_locations: (xyz, num_sources)"""
source_locations[0,  :] = np.cos(doas[:, 1]) * np.sin(doas[:, 0]) 
source_locations[1,  :] = np.sin(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[2,  :] = np.cos(doas[:, 0])
source_locations *= distance

# Near仮定に基づくステアリングベクトルを計算: steering_vectors(Nk × Ns × M)
near_steering_vectors = calculate_steering_vector(mic_alignments.T, source_locations, freqs,  is_use_far=False)
"""near_steering_vectors: (freq_bins, num_sources, num_microphones)"""

# theta=0に最も近いステアリングベクトルを取り出す
desired_index = np.argmin(np.abs(doas[:, 1]), axis=0)

# 所望音のステアリングベクトル
desired_steering_vector = near_steering_vectors[:, desired_index, :]

# 内積計算
directivity_pattern = np.square(np.abs(np.einsum("km,ksm->ks", np.conjugate(desired_steering_vector), near_steering_vectors)))

# スタイル
plt.style.use("grayscale")

# 音声データをプロットする
fig = plt.figure(figsize=(7,7))

# plot
ax = plt.subplot(111, projection="polar")

# グラフの向き、グリッドの線種を指定
ax.set_theta_zero_location('N')
ax.set_theta_direction('clockwise')
ax.grid(linestyle="--")

# y軸のラベルを調整
ax.yaxis.labelpad = -250
ylabel = plt.ylabel("Response [dB]")
ylabel.set_position((0, 0.6))
ylabel.set_rotation(0)
plt.yticks([-20, -10, 0])
plt.ylim([-30, 0])

# x軸のラベル
plt.xlabel("Azimuth [degree]")

# 描画する周波数
draw_freqs = np.array([1000, 2000, 3000, 4000])

draw_freq_list = np.argmin(np.abs(freqs[:, None]-draw_freqs[None, :]), axis=0)

for draw_freq_index in draw_freq_list:
    # 周波数ごとに指向性を描画
    plt.plot(doas[:, 1], 10.*np.log10(directivity_pattern[draw_freq_index, :]), lw=3, label="{} [Hz]".format(freqs[draw_freq_index]))

plt.legend(loc=(0.2, 0.6))

plt.show()