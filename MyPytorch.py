import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchtext


"""
PyTorch用
"""
# モデルのパラメータ数をカウント
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 既存のチェックポイントファイルをロード
def load_checkpoint(model, optimizer, checkpoint_path, device):
    # チェックポイントファイルがない場合エラー
    assert os.path.isfile(checkpoint_path)
    # チェックポイントファイルをロード
    checkpoint = torch.load(checkpoint_path, map_location=device)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    log_epoch = checkpoint['log_epoch']
    print("{}からデータをロードしました。エポック{}から学習を再開します。".format(checkpoint_path, start_epoch))
    return start_epoch, model, optimizer, log_epoch

# モデルの重みを部分的にロード
def load_partial_param(model, pretrained_path):
    checkpoint = torch.load(pretrained_path)
    pretrained_param = checkpoint['model_state_dict']
    removed_layer_list = ['you_want_remove_layer_name']
    for removed_layer in removed_layer_list:
        pretrained_param = {k: v for k, v in pretrained_partial_param.items() if removed_layer not in k}
    state_dict = model.state_dict()
    for k, v in pretrained_param.items():
        state_dict.update({k: v})
    model.load_state_dict(state_dict)

# モデルの一部の重みをフリーズ
def freeze_param(model):
    for name, param in model.named_parameters():
        if 'param_name' in name:
            param.requires_grad = False



def train(model, dataloaders_dict, criterion, optimizer, num_epochs):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    print('-----start-------')
    # ネットワークをGPUへ
    model.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # epochのループ
    for epoch in range(num_epochs):
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # モデルを訓練モードに
            else:
                model.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # データローダーからミニバッチを取り出すループ
            for batch in (dataloaders_dict[phase]):
                # batchはTextとLableの辞書オブジェクト

                # GPUが使えるならGPUにデータを送る
                texts = batch.Text[0].to(device)  # 文章
                labels = batch.Label.to(device)  # ラベル

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):

                    # # mask作成
                    # input_pad = 1  # 単語のIDにおいて、'<pad>': 1 なので
                    # input_mask = (inputs != input_pad)
                    #
                    # # Transformerに入力
                    # outputs, _, _ = net(inputs, input_mask)
                    # loss = criterion(outputs, labels)  # 損失を計算

                    scores = model(texts)
                    loss = criterion(scores, labels) # 損失を計算

                    pred_values, pred_label = torch.max(scores, 1)  # ラベルを予測 tensor([1, 0, 1, 0])


                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # 結果の計算
                    epoch_loss += loss.item() * texts.size(0)  # lossの合計を更新
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(pred_label == labels.data)

            # epochごとのlossと正解率
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs,
                                                                           phase, epoch_loss, epoch_acc))

    return model





class LstmTextClassify(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, seq_len, target_size, text_embedding_vecors=None):
        # 親クラスのコンストラクタ
        super(LstmTextClassify, self).__init__()
        # 入力した単語ID列(sequence)をベクトル化するEmbedderを定義
        # self.embedding = nn.Embedding(seq_len, embedding_dim)
        # word2vecやfasttextで学習済みのembeddingを使用する場合は以下を使用 freeze=Trueにより、誤差逆伝播で更新されなくなる
        self.embedding = nn.Embedding.from_pretrained(embeddings=text_embedding_vecors, freeze=True)
        # LSTMの隠れ層を定義　batch_first=Trueで入力を[batch, seq_len, embedding_dim]の形に
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # LSTMの出力を受け取って全結合するLinearを定義
        self.linear = nn.Linear(hidden_dim, target_size)
        # softmaxのLog版。dim=0で列、dim=1で行方向を確率変換。
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, sequence):

        # 入力した単語ID列(sequence)をベクトル化する
        embeds = self.embedding(sequence) # [batch, seq_len, embedding_dim]
        # LSTMの出力　hn_allは全隠れ層の出力, hnは最後の隠れ層の出力、cnは最後の隠れ層セルの値
        hn_all, (hn, cn) = self.lstm(embeds) # hn_all:[batch, seq_len, hidden_dim], hn,cn:[1, batch, hidden_dim]
        # LSTMの出力を受け取って全結合する(最終層の出力だけ使用するので-1を指定)
        linear_output = self.linear(hn_all[:, -1, :]) # [batch, hidden_dim] → [batch, target_size]
        # softmaxに入力し、確率として表現
        score = self.softmax(linear_output) # [batch, target_size]

        return score





if __name__ == "__main__":

    from MyFunc import mecab_wakati

    max_seq_len = 25
    embedding_dim = 300
    hidden_dim = 128
    target_size = 2 # Positive or Negative
    num_epochs = 2
    batch_size = 2

    # Mecabを用いた形態素解析用
    # text = "【人工知能】は「人間」の仕事を奪った"
    # wakati = mecab_wakati(text)
    # print(wakati)

    # torchtextを用いたデータセットの作成
    # Fieldオブジェクトの作成
    # tsvやcsvデータを読み込んだときに、読み込んだ内容に対して行う処理を定義
    # sequential: データの長さが可変か？文章は長さがいろいろなのでTrue.ラベルはFalse
    # tokenize: 文章を読み込んだときに、前処理や単語分割をするための関数を定義
    # use_vocab：単語をボキャブラリーに追加するかどうか
    # lower：アルファベットがあったときに小文字に変換するかどうか
    # include_length: 文章の単語数のデータを保持するか
    # batch_first：ミニバッチの次元を先頭に用意するかどうか
    # fix_length：全部の文章を指定した長さと同じになるように、paddingする
    TEXT = torchtext.data.Field(sequential=True, tokenize=mecab_wakati, \
    use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_seq_len)
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
    # csvファイルを読み込み、TabularDatasetオブジェクトを作成
    # 1行がTEXTとLABELで区切られていることをfieldsで指示
    train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(
    path='../datasets/sample_data/', train='train.csv',
    validation='val.csv', test='test.csv', format='csv',
    fields=[('Text', TEXT), ('Label', LABEL)])

    # torchtextで単語ベクトルとして読み込む
    from torchtext.vocab import Vectors
    # fasttextの学習済みモデルは「https://qiita.com/Hironsan/items/8f7d35f0a36e0f99752c」を参照
    japanese_fasttext_vectors = Vectors(name='../datasets/fasttext/model_vector_neologd.vec')
    # 単語ベクトルの中身を確認
    print("1単語を表現する次元数：", japanese_fasttext_vectors.dim)
    print("学習済みfasttextに含まれる全単語数：", len(japanese_fasttext_vectors.itos))

    # ベクトル化したバージョンのボキャブラリーを作成 min_freqより小さい出現頻度の単語は登録されない
    # ID:0はknown, ID:1はpadding
    TEXT.build_vocab(train_ds, vectors=japanese_fasttext_vectors, min_freq=1)
    # ボキャブラリーのベクトルを確認
    print(TEXT.vocab.vectors.shape)  # 52個の単語が300次元のベクトルで表現されている
    # ボキャブラリーの単語の順番を確認します
    print(TEXT.vocab.stoi)
    # LSTMを用いたテキスト分類
    model = LstmTextClassify(embedding_dim, hidden_dim, max_seq_len, target_size, text_embedding_vecors=TEXT.vocab.vectors)

    # DataLoaderを作成
    train_dl = torchtext.data.Iterator(train_ds, batch_size=batch_size, train=True)
    val_dl = torchtext.data.Iterator(val_ds, batch_size=batch_size, train=False, sort=False)
    test_dl = torchtext.data.Iterator(test_ds, batch_size=batch_size, train=False, sort=False)
    # batch = next(iter(val_dl))
    # print(batch.Text)
    # print(batch.Label)
    dataloaders_dict = {'train': train_dl, 'val': val_dl}
    # 損失関数の設定
    criterion = nn.CrossEntropyLoss()
    # 最適化手法の設定
    learning_rate = 2e-5
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    train(model, dataloaders_dict, criterion, optimizer, num_epochs)
