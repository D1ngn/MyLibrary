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
    $ julius -C ~/julius/dictation-kit-4.5/main.jconf -C ~/julius/dictation-kit-4.5/am-gmm.jconf -nostrip -input rawfile
    ```

    - 自分で定義した辞書を用いて認識する場合

      ```
      $ julius -C ~/julius/dictation-kit-4.5/am-gmm.jconf -gram ~/julius/dict/test -nostrip -input file -outfile
      ```

  - DNNベースの音響モデルを用いる場合（処理速度は若干低下するが、精度は高い）→今はさまざまな日本語を認識

    ```
    $ julius -C ~/julius/dictation-kit-4.5/main.jconf -C ~/julius/dictation-kit-4.5/am-dnn.jconf -dnnconf ~/julius/dictation-kit-4.5/julius.dnnconf -nostrip -input rawfile
    ```

- マイクで取得した音声に対して音声認識

  ```
  $ julius -C ~/julius/dictation-kit-4.5/main.jconf -C ~/julius/dictation-kit-4.5/am-gmm.jconf -nostrip -input mic
  ```

- TCP/IPを活用して音声認識（Juliusモジュールモード）

  - サーバ側

    ```
    $ julius -C ~/julius/dictation-kit-4.5/main.jconf -C ~/julius/dictation-kit-4.5/am-gmm.jconf -module
    ```

  - クライアント側（サーバ側から音声認識結果を受け取り標準出力）

    ```
    #!/usr/bin/python3
    # -*- coding: utf-8 -*-
    import socket
    import logging
    
    # create logger
    logger = logging.getLogger('simple_example')
    logging.basicConfig(filename='example.log', level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    host = '127.0.0.1'   # IPアドレス
    port = 10500         # Juliusとの通信用ポート番号
    
    # Juliusにソケット通信で接続
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    # sock.send("RESUME".encode('utf-8'))
    
    data = ""
    try:
        while True:
            # 音声認識結果のみをXMLで取得
            while (data.find("</RECOGOUT>\n.") == -1):
                soc = sock.recv(1024)
                data = data + soc.decode('utf-8')
    
            # 音声認識結果のXMLから単語部分のみを抜き出して連結
            recog_text = ""
            for line in data.split('\n'):
                index = line.find('WORD="')
                if index != -1:
                    line = line[index+6:line.find('"', index+6)]
                    recog_text = recog_text + line
    
            logging.info("認識結果: " + recog_text)
            print("認識結果: " + recog_text)
            data = ""
    
    # Command + C などで強制終了した場合、JuliusサーバーをDIEコマンドにより落とす
    # https://julius.osdn.jp/juliusbook/ja/desc_module.html
    except KeyboardInterrupt:
        print('finished')
        sock.send("DIE".encode('utf-8'))
        sock.close()
    ```

    

  

  

- 使用するオプションの詳細

  - `-outfile`：認識結果を「.out」拡張子付きファイルに出力
  - `-nostrip`：音声波形中に存在する振幅が0となるフレームを取り除かないようにする（デフォルトでは取り除く）
  - `-quiet`：出力が単純な結果だけになる
  - `-module`：Juliusをモジュールモード（TCP/IP接続待ち状態）で起動
  - `-realtime, -norealtime`：`-realtime`を指定するとストリーミング処理を行い、`-norealtime`を指定するとブロック処理を行う
  - `-cutsilence`：発話の音声区間検出を有効にする





#### 辞書の変更方法

- `~/julius/dictation-kit-4.5/main.jconf`内の`-v`で指定されている.dictファイルの部分を自分で作成した.dictファイルに変更

- 現在は`~/julius/dictation-kit-4.5/LM/nakbot_words.dict`という.dictファイルが指定されているのでそれを参考に中身を書き換えればよい

  ```
  2	[前に進め]	m a e n i s u s u m e
  2	[後ろに退がれ]	u sh i r o n i s a g a r e
  2	[回れ右]	m a w a r e m i g i
  2	[回れ左]	m a w a r e h i d a r i
  2	[こっちにおいで]	k o q ch i n i o i d e
  2	[ナックボット]	n a q k u b o q t o
  2	[声を覚えて]	k o e o o b o e t e
  <s> []  silB
  </s>    []  silE
  ```

  - 上記のように記述する
  - 各行の左端にある2と*<*s*>*、*<*/s*>*はおまじない

- 自分で定義した辞書を使うことで辞書ファイルのサイズを小さくすることができ、辞書の探索範囲を減らすことができるため、応答速度を速くすることができる



#### 辞書の作成方法

1. 読みファイルの作成

   まず以下のコマンドで`~/julius/dict`内に`test.yomi`を作成

   ```
   $ cd ~/julius/dict
   $ touch test.yomi
   ```

   その後、`test.yomi`に以下を記述

   ```
   おはよう    おはよう
   ございます ございます
   こんにちは こんにちわ
   こんばんは こんばんわ
   ```

   注意点は下記の通り

   - 読みはひらがなで定義
   - 中央のスペースはTabキー1回で空ける
   - 読みは実際の発音で記載する（例）こんにち''わ""

2. 音素ファイルの作成

   以下のコマンドを実行

   ```
   $ sudo perl ~/julius/gramtools/yomi2voca/yomi2voca.pl test.yomi > test.phone
   ```

   `test.phone`の中身が以下のようになれば成功

   ```
   おはよう    o h a y o u
   ございます g o z a i m a s u
   こんにちは k o N n i ch i w a
   こんばんは k o N b a N w a
   ```

3. 構文ファイルの作成

   以下のコマンドを実行して`test.grammar`を作成

   ```
   $ touch test.grammar
   ```

   その後、`test.grammar`に以下を記述

   ```
   S : NS_B TEST NS_E
   TEST : OHAYOU
   TEST : OHAYOU GOZAIMASU
   TEST : KONNICHIWA
   TEST : KONBANWA
   ```

   詳細は下記の通り

   - 1行目は構文定義で、`NS_B`は開始位置、`NS_E`は終了位置を表すシンボル
   - 2行目以降の`TEST`部分は1行目の`TEST`部分の参照として用いられる
   - 2行目以降の「:」の右側の文字列は読みファイルをもとに単語を組み合わせた文字列

4. 語彙ファイルの作成

   以下のコマンドを実行して`test.voca`を作成

   ```
   $ touch test.voca
   ```

   その後、`test.voca`に以下を記述

   ```
   % OHAYOU
   おはよう        o h a y o u
   % GOZAIMASU
   ございます      g o z a i m a s u
   % KONNICHIWA
   こんにちは      k o N n i ch i w a
   % KONBANWA
   こんばんは      k o N b a N w a
   % NS_B
   [s]             silB
   % NS_E
   [/s]            silE
   ```

5. 辞書ファイルの作成

   まず、以下のコマンドを実行して`dfa_minimize`と`mkfa`を`~/julius/gramtools/mkdfa/`内に移動（`Warning: dfa_minimize not found in the same place as mkdfa.pl`などが出て、.dictファイルが生成されないため）

   ```
   $ cp ~/julius/gramtools/dfa_minimize/dfa_minimize ~/julius/gramtools/mkdfa/dfa_minimize
   $ cp ~/julius/gramtools/mkdfa/mkfa-1.44-flex/mkfa ~/julius/gramtools/mkdfa/mkfa
   ```

   その後、以下を実行すると`~/julius/dict`内に`test.dfa`と`test.dict`、`test.term`が作成される

   ```
   $ sudo perl ~/julius/gramtools/mkdfa/mkdfa.pl ~/julius/dict/test
   ```

   

