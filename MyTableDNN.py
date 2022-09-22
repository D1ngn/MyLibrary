# 必要モジュールのimport
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from glob import glob
from xfeat import SelectNumerical, SelectCategorical
from sklearn.preprocessing import LabelEncoder # カテゴリ変数をラベルエンコーディングするためのライブラリ
from sklearn.preprocessing import StandardScaler # 標準化用
# PyTorch関連モジュールのimport
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data


# EarlyStopping用（参考：「https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py」）
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# モデル定義（シンプルな線形層）
def init_weights(layer):
    # 全結合層の場合
    if isinstance(layer, (nn.Linear)):
        nn.init.xavier_normal_(layer.weight) # xavierの重みで初期化
        # バイアスがある場合
        if layer.bias is not None:
            layer.bias.data.fill_(0.0) # 0.0で初期化

class LinearNorm(nn.Module):
    def __init__(self, lstm_hidden, emb_dim):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(lstm_hidden, emb_dim)

    def forward(self, x):
        return self.linear_layer(x)

class FCBlock(nn.Module):
    def __init__(self, in_feature_dim, out_feature_dim):
        """
        in_feature_dim: last dimension of input vectors
        out_feature_dim: last dimension of output vectors
        """
        super(FCBlock, self).__init__()
        self.fc_relu = nn.Sequential(
            nn.Linear(in_feature_dim, out_feature_dim),
            nn.BatchNorm1d(out_feature_dim),
            nn.Dropout2d(p=0.5), # 50%の割合でドロップアウトを実行
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.fc_relu(x)
        return x

class TableFC(nn.Module):
    def __init__(self, in_feature_dim, out_feature_dim, emb_dims):
        """
        in_feature_dim: last dimension of input vectors
        out_feature_dim: last dimension of output vectors
        emb_dims: list of tuple representing a pair of total and the embedding dimension of a categorical variable
        """
        super(TableFC, self).__init__()
        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims]) # x:num_uniques, y:embed_dim
        self.emb_dropout_layer = nn.Dropout(0.04)
        self.fc_layers = nn.Sequential(
            FCBlock(in_feature_dim, int(in_feature_dim/2)),
            FCBlock(int(in_feature_dim/2), int(in_feature_dim/4)),
            FCBlock(int(in_feature_dim/4), out_feature_dim),
        )
    def forward(self, X_num, X_cat):
        """X_num: (batch_size, num_features=24)"""
        """X_cat: (batch_size, num_features=12)"""
        # カテゴリ変数がラベルエンコーディングによって0以上の整数になっていないとエラーが出る（負の整数は不可）
        X_cat = [emb_layer(X_cat[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
        X_cat = torch.cat(X_cat, dim=1)
        X_cat = self.emb_dropout_layer(X_cat)
        """X_cat: (batch_size, num_features=278)"""
        X = torch.cat([X_num, X_cat], dim=1) # チャンネル方向に結合
        """X: (batch_size, num_features=302)"""
        X = self.fc_layers(X)
        return X

    
# データセットのクラス
class TableDataset(data.Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = X_num # 学習用数値説明変数
        self.X_cat = X_cat # 学習用カテゴリ説明変数
        self.y = y # 学習用目的変数
    
    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, index):
        # ファイルパスを取得
        X_num = self.X_num[index]
        """X_num: (num_features,)"""
        X_cat = self.X_cat[index]
        """X_cat: (num_features,)"""
        y = self.y[index]
        """y: double"""
        # PyTorchではfloat32型にしないとエラーが出る
        X_num = torch.from_numpy(X_num).float()
        X_cat = torch.from_numpy(X_cat).long()
        y = torch.tensor(y, dtype=torch.float32)
        return X_num, X_cat, y

    
# モデルを学習させる関数を作成
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, param_save_dir, checkpoint_path=None):
    # GPUが使える場合あはGPUを使用、使えない場合はCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：" , device)

    # モデルがある程度固定(イテレーションごとの入力サイズが一定)であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # 各カウンタを初期化
    start_epoch = 0
    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []

    # 学習を再開する場合はパラメータをロード、最初から始める場合は特に処理は行われない
    if checkpoint_path is not None:
        start_epoch, model, optimizer, log_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)
        # GPU環境で学習したOptimizerを再度GPU環境で学習させる場合は逐一値をdeviceへ送る
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    else:
        print("checkpointファイルがありません。最初から学習を開始します。")

    # ネットワークをGPUへ
    model.to(device)

    # 学習データと検証データの数、バッチサイズを取得
    num_train_data = len(dataloaders_dict['train'].dataset)
    num_val_data = len(dataloaders_dict['val'].dataset)
    batch_size = dataloaders_dict['train'].batch_size

    print("num_train_data:", num_train_data)
    print("num_val_data:", num_val_data)
    print("batch_size:", batch_size)

    # epochごとのループ
    for epoch in range(start_epoch, num_epochs):

        # 開始時刻を記録
        epoch_start_time = time.time()
        iter_start_time = time.time()

        print("エポック {}/{}".format(epoch+1, num_epochs))

        # モデルのモードを切り替える(学習 ⇔ 検証)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # 学習モード
            else:
                model.eval() # 検証モード

            # データローダーからミニバッチずつ取り出すループ
            for X_num, X_cat, y in dataloaders_dict[phase]:
                """
                X_num: 数値説明変数 (batch_size, num_features)
                X_cat: カテゴリ説明変数 (batch_size, num_features)
                y: 目的変数 (batch_size,)
                """
                # GPUが使える場合、データをGPUへ送る
                X_num = X_num.to(device)
                X_cat = X_cat.to(device)
                y = y.to(device)
                # optimizerを初期化
                optimizer.zero_grad()
                # 順伝播
                with torch.set_grad_enabled(phase == 'train'):
                    # 説明変数をモデルに入力し、予測
                    predicted = model(X_num, X_cat)
                    # 損失を計算
                    loss = criterion(predicted, y)
                    # 学習時は誤差逆伝播(バックプロパゲーション)
                    if phase == 'train':
                        # 誤差逆伝播を行い、勾配を算出
                        loss.backward()
                        # パラメータ更新
                        optimizer.step()
                        # 10iterationごとにlossと処理時間を表示
                        if (iteration % 10 == 0):
                            iter_finish_time = time.time()
                            duration_per_ten_iter = iter_finish_time - iter_start_time
                            # 0次元のテンソルから値を取り出す場合は「.item()」を使う
                            print("イテレーション {} | Loss:{:.4f} | 経過時間:{:.4f}[sec]".format(iteration, loss.item()/batch_size, duration_per_ten_iter))
                            epoch_train_loss += loss.item()

                        epoch_train_loss += loss.item()
                        iteration += 1
                    # 検証時
                    else:
                        epoch_val_loss += loss.item()

        # epochごとのlossと正解率を表示
        epoch_finish_time = time.time()
        duration_per_epoch = epoch_finish_time - epoch_start_time
        print("=" * 30)
        print("エポック {} | Epoch train Loss:{:.4f} | Epoch val Loss:{:.4f}".format(epoch+1, epoch_train_loss/num_train_data, epoch_val_loss/num_val_data))
        print("経過時間:{:.4f}[sec/epoch]".format(duration_per_epoch))
        
#         # 学習率の管理
#         scheduler.step(epoch_val_loss)

        # 学習経過を分析できるようにcsvファイルにログを保存 → tensorboardに変更しても良いかも
        log_epoch = {'epoch': epoch+1, 'train_loss': epoch_train_loss/num_train_data, 'val_loss': epoch_val_loss/num_val_data}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        log_save_path = os.path.join(param_save_dir, "log.xlsx")
        df.to_excel(log_save_path, index=False)

        # エポックごとのタイムログをファイルに追記
        time_log = os.path.join(param_save_dir, "time_log.txt")
        with open(time_log, mode='a') as f:
            f.write("エポック {} | {}\n".format(epoch+1, datetime.datetime.now()))

        # epochごとの損失を初期化
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # 学習したモデルのパラメータを保存
        if ((epoch+1) % 10 == 0):
            param_save_path = os.path.join(param_save_dir, "ckpt_epoch{}.pt".format(epoch+1))
            # torch.save(net.state_dict(), param_save_path) # 推論のみを行う場合
            # 学習を再開できるように変更
            torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'log_epoch': log_epoch
            }, param_save_path)


if __name__ == "__main__":
    # PyTorch 以外のRNGを初期化
    # random.seed(0)
    np.random.seed(0)
    # PyTorch のRNGを初期化
    torch.manual_seed(0)

    # 数値データを取り出し
    num_df = SelectNumerical().fit_transform(df)
    # 標準化
    scaler = StandardScaler()
    num_df = pd.DataFrame(scaler.fit_transform(num_df))

    # カテゴリデータを取り出し
    cat_df = SelectCategorical().fit_transform(df)
    # 特徴量（列）ごとの繰り返し処理
    label_encoders = {}
    for cat_col in cat_df.columns:
        # ラベルエンコーダーのインスタンスを作成
        label_encoders[cat_col] = LabelEncoder()
        # カテゴリ変数を整数にマッピング
        cat_df[cat_col] = label_encoders[cat_col].fit_transform(cat_df[cat_col])

    # 数値データとカテゴリデータを結合
    feat_df = pd.concat([num_df,cat_df], axis=1)
    # 学習データと検証データ、テストデータに分割
    train_df = feat_df.iloc[:val_min_idx, :]
    val_df = feat_df.iloc[val_min_idx:test_min_idx, :]
    test_df = feat_df.iloc[test_min_idx:, :]
    print(train_df.shape, val_df.shape, test_df.shape)
    # 特徴量の列名（IDと目的変数を除く）
    feat_cols = [col for col in train_df.columns if col not in rm_cols+[ID, TARGET]]
    # カテゴリ変数の列名
    cat_cols = list(cat_df.columns)
    # 説明変数と目的変数に分ける
    train_X = train_df[feat_cols]
    train_y = train_df[TARGET]
    val_X = val_df[feat_cols]
    val_y = val_df[TARGET]
    test_X = test_df[feat_cols]
    test_y = test_df[TARGET]
    # 説明変数をカテゴリ変数と数値変数に分ける
    train_X_cat = train_X[cat_cols]
    train_X_num = train_X.drop(cat_cols, axis=1)
    val_X_cat = val_X[cat_cols]
    val_X_num = val_X.drop(cat_cols, axis=1)
    test_X_cat = test_X[cat_cols]
    test_X_num = test_X.drop(cat_cols, axis=1)

    ####################################以下学習用##########################################
    # 各パラメータを設定
    batch_size = 1024 # バッチサイズ
    # カテゴリ変数のユニーク数のリストを取得し、Embedding用の情報を取得
    # ラベルエンコーディングによって得られる値は0以上でなければならない（-1等は不可）
    #（参考：「https://yashuseth.wordpress.com/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/」）
    max_emb_dim = 50
    num_dims = 24
    cat_dims = [int(feat_df[col].nunique()) for col in cat_cols] # 例：[15, 5, 2, 4, 112]
    emb_dims = [(x, min(max_emb_dim, (x + 1) // 2)) for x in cat_dims] # 例：[(15, 8), (5, 3), (2, 1), (4, 2), (112, 50)]
    # 全結合層へ入力すると特徴量の次元数は「数値変数の数」＋「カテゴリ変数のEmbeddingの次元数の合計」
    in_feature_dim = num_dims + sum([min(max_emb_dim, (x + 1) // 2) for x in cat_dims])
    # ネットワークモデルの定義
    model = TableFC(in_feature_dim=in_feature_dim, out_feature_dim=1, emb_dims=emb_dims)
    # データセットのインスタンスを作成
    train_dataset = TableDataset(train_X_num, train_X_cat, train_y)
    val_dataset = TableDataset(val_X_num, val_X_cat, val_y)
    # データローダーを作成
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloaders_dict = {'train':train_dataloader, 'val':val_dataloader} # データローダーを格納するリスト
    # 損失関数を定義
    criterion = nn.L1Loss(reduction='mean') # inputとtargetの各要素の差の絶対値の平均
    # 最適化手法を定義
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Default
    # optimizer = optim.Adam(model.parameters(), lr=0.0001) # val lossが振動してしまうので学習率を下げる
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) # val loss が3エポックの間下がらなかった場合学習率を0.5倍にする
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5) # 10エポックごとに学習率を0.5倍にする
    # 各種設定 → いずれ１つのファイルからデータを読み込ませたい
    num_epochs = 30 # epoch数を指定
    # 学習済みモデルのパラメータを保存するディレクトリを作成
    param_save_dir = "./output/ckpt/ckpt_linear_model_0921" # 学習済みモデルのパラメータを保存するディレクトリのパスを指定
    os.makedirs(param_save_dir, exist_ok=True)
    # モデルを学習
    train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, param_save_dir)
    

    ####################################以下推論用##########################################
    # GPUが使える場合はGPUを使用、使えない場合はCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：" , device)
    # 指定したチェックポイントに保存されたモデルのパラメータをロード
    checkpoint_path = os.path.join(param_save_dir, "ckpt_epoch30.pt")
    model_params = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_params['model_state_dict'])
    # numpy配列からPyTorchのテンソルに変換
    test_X_num_tensor = torch.from_numpy(test_X_num).float()
    test_X_cat_tensor = torch.from_numpy(test_X_cat).long()
    test_pred = model(test_X_num_tensor, test_X_cat_tensor)
    # PyTorchのテンソルからnumpy配列に変換
    test_pred = test_pred.cpu().detach().numpy().copy() # CPU
    test_df[TARGET] = test_pred
    sub_df = pd.merge(sub_df[['ID']], test_df[['ID', TARGET]], on='ID') # IDが一致する行同士を結合
    sub_df.to_csv(BASE_PATH + 'output/submission_baseline.csv', index=False)