import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data


class LstmTextClassify(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, seq_len, target_size):
        # 親クラスのコンストラクタ
        super(LstmTextClassify, self).__init__()
        # 入力した単語ID列(sequence)をベクトル化するEmbedderを定義
        self.embedding = nn.Embedding(seq_len, embedding_dim)
        # word2vecやfasttextで学習済みのembeddingを使用する場合は以下を使用 freeze=Trueにより、誤差逆伝播で更新されなくなる
        # self.embedding = nn.Embedding.from_pretrained(embeddings=text_embedding_vecors, freeze=True)
        # LSTMの隠れ層を定義　batch_first=Trueで入力を[batch, seq_len, embedding_dim]の形に
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # LSTMの出力を受け取って全結合するLinearを定義
        self.linear = nn.Linear(hidden_dim, target_size)
        # softmaxのLog版。dim=0で列、dim=1で行方向を確率変換。
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, sequence):
        print(sequence.shape)
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

    sequence = [[0, 4, 3, 7, 6, 2, 5, 1], [0, 4, 3, 7, 6, 2, 5, 1]]
    sequence_tensor = torch.tensor(sequence)


    embedding_dim = 300
    hidden_dim = 128
    seq_len = len(sequence)
    target_size = 2 # Positive or Negative

    # LSTMを用いたテキスト分類
    model = LstmTextClassify(embedding_dim, hidden_dim, seq_len, target_size)
    output = model(sequence_tensor)
    print(output)
