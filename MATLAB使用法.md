## MATLAB使用法

#### 基本情報

・MATLABのインデックスは1から始まる(他のプログラミング言語は0から始まる)



#### ベクトルまたは行列同士の垂直(列)方向の連結

・セミコロン(;)で連結する場合

```
>> A = [1 2 3; 4 5 6]
A =
     1     2     3
     4     5     6
```

・vertcatを使う場合

```
>> B = [7 8 9]
>> C = vertcat(A,B)
C =
     1     2     3
     4     5     6
     7     8     9
```

#### ベクトルまたは行列同士の水平(行)方向の連結

・直接連結する場合

```
>> A = [1 2]
>> B = [3 4]
>> C = [A B]
C =
     1     2
     3     4
```

#### ベクトルまたは行列の転置

・`.'`または`transpose`を使用

```
>> A = magic(4)
A = 4×4
    16     2     3    13
     5    11    10     8
     9     7     6    12
     4    14    15     1
>> B = A.' または B =  transpose(A)
B = 4×4
    16     5     9     4
     2    11     7    14
     3    10     6    15
    13     8    12     1
```







#### 音声ファイルの読み込み

```
>> filename = "audio.wav";
>> [audio_data, Fs] = audioread(filename); # audio_dataはサンプリング音声、Fsはサンプリング周波数
```



#### 音声ファイル再生

```
>> sound(audio_data, Fs)
```



#### 音声評価

参考サイト：「[http://bass-db.gforge.inria.fr/bss_eval/#ref1](http://bass-db.gforge.inria.fr/bss_eval/#ref1)」

・瞬時混合(音の残響・反響がない)でオーディオが1チャンネルの場合

1. 以下の式に従って混合音声を分解する

   ![image-20200811003923333](/Users/nagano.daichi/Library/Application Support/typora-user-images/image-20200811003923333.png)

   seはサンプリングした推定音声を行ベクトル(「shape:(1:num_samples)の形式」)にしたもの、Sは目的の音声の行ベクトルをi行目S(i, :)に, そのほかの音声の行ベクトルをj行目S(j, :)に並べた行列、iは行列Sにおいて目的音声が何列目にあるかを指定するインデックス

   ```
   >> [s_target, e_interf, e_noise, e_artif] = bss_decomp_gain(se, i, S, N)
   ```

   内部雑音Nを考慮しない場合は

   ```
   >> [s_target, e_interf, e_artif] = bss_decomp_gain(se, i, S)
   ```

   ※ `audio_read`で読み込んだ関数は列ベクトルの形なので、転置して行ベクトルにする必要がある

   (実行例)

   ```
   # 目的音
   >> [target, Fs] = audioread("target.wav"); 
   >> target = target.' # 転置
   
   # 別の音源
   >> [interference, Fs] = audioread("interference.wav"); 
   >> interference = interference.' # 転置
   
   # 混合音（評価したい音声と比較する場合）
   >> [mixed, Fs] = audioread("mixed.wav"); 
   >> mixed = mixed.' # 転置
   
   # 評価したい音声(深層学習モデルなどが推定した音声)
   >> [predicted, Fs] = audioread("predicted.wav"); 
   >> predicted = predicted.' # 転置
   
   # 行列Sの作成（１行目：目的音声のデータ, ２行目：別の音源のデータ）
   S = [target;interference]
   
   # 音声の分解
   # 行列Sの1行目が目的の音声のデータなので第２引数に１を指定
   >> [s_target, e_interf, e_artif] = bss_decomp_gain(predicted, 1, S) # 推定音声を分解
   ```

   

2. 分解した要素のエネルギー比率を計算することで、各音声評価指標を算出

   ```
   >> [SDR, SIR, SNR, SAR] = bss_crit(s_target, e_interf, e_noise, e_artif)
   ```

   内部雑音Nを考慮しない場合は

   ```
   >> [SDR, SIR, SAR] = bss_crit(s_target, e_interf, e_artif)
   ```




[このサイト](http://sisec2008.wiki.irisa.fr/tiki-index026a.html?page=Under-determined+speech+and+music+mixtures)のEvaluation Critereria







#### 信号アナライザーの起動

```
>> signalAnalyzer
```

