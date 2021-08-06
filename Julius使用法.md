## Juliusについて

#### インストール方法(Macbook)

- Juliusのインストール

  ```
  $ cd 
  $ git clone https://github.com/julius-speech/julius
  $ cd julius
  $ ./configure
  $ make -j4
  （以下のコマンドを実行することでどのディレクトリからもjuliusコマンドを実行できるようになる）
  $ sudo make install
  ```

- dictationキットのインストール(wgetがない場合は`brew install wget`でインストール)

  ```
  $ cd ~/julius
  $ wget https://ja.osdn.net/projects/julius/downloads/71011/dictation-kit-4.5.zip
  $ unzip dictation-kit-4.5.zip
  ```



#### インストール方法(jetson xavier)

- Juliusのインストール

  ```
  $ cd
  $ sudo apt-get install build-essential zlib1g-dev libsdl2-dev libasound2-dev
  $ git clone https://github.com/julius-speech/julius
  $ cd julius
  $ ./configure --build=aarch64-unknown-linux-gnu
  $ make -j4
  （下記lsコマンドで確認）
  $ ls -l julius/julius
  （以下のコマンドを実行することでどのディレクトリからもjuliusコマンドを実行できるようになる）
  $ sudo make install
  ```

- dictationキットのインストール

  ```
  $ cd ~/julius
  $ wget https://ja.osdn.net/projects/julius/downloads/71011/dictation-kit-4.5.zip
  $ unzip dictation-kit-4.5.zip
  ```

  

#### 実行方法

- 音声ファイルに対して音声認識

  - 混合ガウスモデル（GMM）ベースの音響モデルを用いる場合（精度は若干低下するが、処理速度は大きい）→今は「前に進め」、「後ろに退がれ」などを認識

    ```
    $ ~/julius/julius/julius -C ~/julius/dictation-kit-4.5/main.jconf -C ~/julius/dictation-kit-4.5/am-gmm.jconf -nostrip -input rawfile
    ```

  - DNNベースの音響モデルを用いる場合（処理速度は若干低下するが、精度は高い）→今はさまざまな日本語を認識

    ```
    $ ~/julius/julius/julius -C ~/julius/dictation-kit-4.5/main.jconf -C ~/julius/dictation-kit-4.5/am-dnn.jconf -dnnconf ~/julius/dictation-kit-4.5/julius.dnnconf -nostrip -input rawfile
    ```

- マイクで取得した音声に対して音声認識

  ```
  $ ~/julius/julius/julius -C ~/julius/dictation-kit-4.5/main.jconf -C ~/julius/dictation-kit-4.5/am-gmm.jconf -nostrip -input mic
  ```

- 使用するオプションの詳細
  - `-quiet`：出力が単純な結果だけになる
  - 





#### 辞書の変更方法

`~/julius/dictation-kit-4.5/main.jconf`内の`-v`で指定されている.dictファイルの部分を自分で作成した.dictファイルに変更する。現在は`~/julius/dictation-kit-4.5/LM/nakbot_words.dict`という.dictファイルが指定されているのでそれを参考に中身を書き換えればよい