# import numpy as np
# from scipy import signal
# import pyroomacoustics as pa
# import time
# import librosa

# import sys
# sys.path.append('../..')
# # print(sys.path)
# from MyFunc import load_audio_file, wave_to_spec


# wav_1 = "../../../MaskBeamformer/test/p257_006/p257_006_target.wav"
# wav_2 = "../../../MaskBeamformer/test/p257_006/p257_006_interference_azimuth60.wav"

# wav_1 = load_audio_file(wav_1, length=3, sample_rate=16000)
# wav_2 = load_audio_file(wav_2, length=3, sample_rate=16000)

# # _, spec_1 = signal.stft(wav_1, fs=16000, window='hann', nperseg=512, noverlap=352)
# # _, spec_2 = signal.stft(wav_2, fs=16000, window='hann', nperseg=512, noverlap=352)

# # amps_pec_1, phase_spec_1 = wave_to_spec(wav_1[:, 0], fft_size=512, hop_length=160, win_length=None)
# # amp_spec_2, phase_spec_2 = wave_to_spec(wav_2[:, 0], fft_size=512, hop_length=160, win_length=None)

# # f, t, complex_spec1 = signal.stft(wav_1, fs=16000, window='hann', nperseg=1024, noverlap=256)
# # f, t, complex_spec2 = signal.stft(wav_2, fs=16000, window='hann', nperseg=1024, noverlap=256)

# complex_spec1 = librosa.stft(wav_1[:, 0], n_fft=512, hop_length=160, win_length=None, window='hann')
# complex_spec2 = librosa.stft(wav_2[:, 0], n_fft=512, hop_length=160, win_length=None, window='hann')

# spec = np.concatenate(complex_spec1[:, np.newaxis], complex_spec2[:, np.newaxis])
# # spec = np.concatenate((spec_1, spec_2))

# start_time = time.perf_counter()
# y = pa.bss.auxiva(spec)
# finish_time = time.perf_counter()
# # y = pa.bss.ilrma(spec)
# print("処理時間：", finish_time-start_time)

# _, output_data = signal.istft(y, fs=16000)


import time
from scipy.io import wavfile
import pyroomacoustics as pra
import soundfile as sf

wav_1 = "../../../MaskBeamformer/test/p257_006/p257_006_target.wav"
wav_2 = "../../../MaskBeamformer/test/p257_006/p257_006_interference_azimuth60.wav"
wav_mix = "../../../MaskBeamformer/test/p257_006/p257_006_mixed_azimuth60.wav"

# read multichannel wav file
# audio.shape == (nsamples, nchannels)
fs, audio = wavfile.read(wav_mix)

# STFT analysis parameters
fft_size = 512  # `fft_size / fs` should be ~RT60
# hop == fft_size // 2  # half-overlap
hop = 160
win_a = pra.hann(fft_size)  # analysis window
# optimal synthesis window
win_s = pra.transform.compute_synthesis_window(win_a, hop)

# STFT
# X.shape == (nframes, nfrequencies, nchannels)
X = pra.transform.analysis(audio, fft_size, hop, win=win_a)

# Separation
start_time = time.perf_counter()
# Y = pra.bss.auxiva(X, n_iter=20)
Y = pra.bss.ilrma(X, n_iter=20)
finish_time = time.perf_counter()
print("処理時間：", finish_time-start_time)

# iSTFT (introduces an offset of `hop` samples)
# y contains the time domain separated signals
# y.shape == (new_nsamples, nchannels)
y = pra.transform.synthesis(Y, fft_size, hop, win=win_s)

sf.write("test.wav", y, 16000)




from scipy.io import wavfile
import pyroomacoustics as pra
fs, audio = wavfile.read("input.wav") # マルチチャンネル音声の読み込み
win_a = pra.hann(fft_size)  # 短時間フーリエ変換の窓関数
win_s = pra.transform.compute_synthesis_window(win_a, hop) # 逆短時間フーリエ変換の窓関数
X = pra.transform.analysis(audio, fft_size=512, hop=160, win=win_a) # 短時間フーリエ変換
Y = pra.bss.auxiva(X, n_iter=20) # ブラインド音源分離（独立ベクトル分析）
y = pra.transform.synthesis(Y, fft_size=512, hop=160, win=win_s) # 逆短時間フーリエ変換で音源信号を復元