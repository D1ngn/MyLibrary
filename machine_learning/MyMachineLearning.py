import os

# 機械学習用
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d # 3次元描画用
# データセット
from sklearn.datasets import make_blobs # クラスタリング用の等方性ガウス点群を生成
from sklearn.datasets import make_circles # 大きい円と小さい円を描くように２次元の点群を生成
# 特徴量エンジニアリング
from sklearn.feature_extraction.text import TfidfVectorizer
# データ作成用
from sklearn.model_selection import train_test_split, GridSearchCV # データ分割用, グリッドサーチ用
# モデル
from sklearn.pipeline import make_pipeline # 複数モデルを連続で実行するためのパイプライン
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC # Support Vector Classifier
from sklearn.decomposition import PCA # Principal Component Analysis
#評価用
from sklearn.metrics import accuracy_score



"""
データ理解（EDA: Exploratory Data Analysis）
"""
# カテゴリ変数の分析
def analyze_categorical(df, column_name):
    """
    df: pandasのデータフレーム （num_samples, num_features）
    column_name: 列名 str型
    """
    # ユニークな要素の値のリスト（numpy array形式）を取得
    unique_list = df[column_name].unique()
    # ユニークな要素の値（indexに格納）とその出現回数（dataに格納）を取得
    unique_freq = df[column_name].value_counts()
    # ユニークな要素の個数を取得（drop=FalseでNaNを含んだ値を返す）
    unique_num = df[column_name].nunique(drop=True)
    return unique_list, unique_freq, unique_num

# 特徴量の列ごとに欠損値の数を算出
def missing_values_table(df):
    """
    df: pandasのデータフレーム （num_samples, num_features）
    mis_val_table_ren_columns: 欠損値情報を表すデータフレーム （num_samples, num_features）
    """
    # 各列ごとの欠損値の数
    mis_val = df.isnull().sum()
    # 行数に対する欠損値の割合
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    # 欠損値の数と欠損値の割合を横方向に結合し、表を作成
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # 列名を変更
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : "Missing Values", 1 : "% of Total Values"})
    # 欠損値の割合が降順になるようにソート（round(1)で小数点1桁表示）
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    # 欠損値のある特徴量（列）の割合を表示
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    return mis_val_table_ren_columns



"""
学習・検証用データの生成
"""
# 学習データと検証データの分割
def train_valid_sep_kfold(X_train, y_train, FOLDS=5, SEED=0):
    from sklearn.model_selection import KFold
    """
    X_train: 学習データに含まれる説明変数 （num_samples, num_features）
    y_train: 学習データに含まれる目的変数 （num_samples, 1)
    FOLDS: 交差検証に用いるホールド数
    """
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    idx_tr, idx_va = list(kf.split(X_train))[0] # 交差検証による分割の1つを使用して、学習用と検証用に分ける
    X_tr, X_va = X_train.iloc[idx_tr], X_train.iloc[idx_va]
    y_tr, y_va = y_train.iloc[idx_tr], y_train.iloc[idx_va]
    return X_tr, X_va, y_tr, y_va

# 学習データと検証データの分割（分類問題でクラス間の比率が大きく異なる場合に使用）
def train_valid_sep_skfold(X_train, y_train, FOLDS=5, SEED=0):
    from sklearn.model_selection import StratifiedKFold
    """
    X_train: 学習データに含まれる説明変数 （num_samples, num_features）
    y_train: 学習データに含まれる目的変数 （num_samples, 1)
    FOLDS: 交差検証に用いるホールド数
    """
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    idx_tr, idx_va = list(skf.split(X_train))[0]
    X_tr, X_va = X_train.iloc[idx_tr], X_train.iloc[idx_va]
    y_tr, y_va = y_train.iloc[idx_tr], y_train.iloc[idx_va]
    return X_tr, X_va, y_tr, y_va

# 交差検証
def apply_kfold(X_train, y_train, FOLDS=5, SEED=0):
    from sklearn.model_selection import KFold
    """
    X_train: 学習データに含まれる説明変数 （num_samples, num_features）
    y_train: 学習データに含まれる目的変数 （num_samples, 1)
    FOLDS: 交差検証に用いるホールド数
    """
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (idx_train, idx_valid) in enumerate(kf.split(X_train)):
        X_tr, X_va = X_train.iloc[idx_tr], X_train.iloc[idx_va]
        y_tr, y_va = y_train.iloc[idx_tr], y_train.iloc[idx_va]

    return X_tr, X_va, y_tr, y_va




from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc # メモリ管理用（ガベージコレクション：必要なくなったメモリ領域を自動的に開放する機能）
# warningsを非表示
import warnings
warnings.filterwarnings('ignore')

def model(train_features, test_features, id_column, target_column, encoding = 'ohe', n_folds=5):
    
    """
    Train and test a light gradient boosting model using cross validation. 
    
    Parameters
    --------
        train_features (pd.DataFrame): 学習特徴量のデータフレーム（目的変数の列を含む） （num_samples, num_features）
        test_features (pd.DataFrame): 評価特徴量のデータフレーム （num_samples, num_features）
        id_column: IDに相当する列名 （str）
        target_column: 目的変数の列名 （str）
        encoding (str, default = 'ohe'): 
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation
        
    Return
    --------
        submission (pd.DataFrame): id_columnとモデルが推定したtarget_columnを含む提出用のデータフレーム
        feature_importances (pd.DataFrame): モデルを構築する際の各特徴量の重要度を含んだデータフレーム
        valid_metrics (pd.DataFrame): 各フォールドとそれらを総合した学習スコア・検証スコアを含むデータフレーム
                                        評価指標（メトリクス）にはROC AUCを使用
    """
    
    # IDの抽出
    train_ids = train_features[id_column]
    test_ids = test_features[id_column]
    # 学習時に用いる正解ラベル（目的変数）を抽出
    labels = train_features[target_column] 
    # IDと目的変数を学習特徴量から削除
    train_features = train_features.drop(columns = [id_column, target_column])
    # IDと目的変数を評価特徴量から削除
    test_features = test_features.drop(columns = [id_column])
    
    # One-Hotエンコーディング
    if encoding == 'ohe':
        train_features = pd.get_dummies(train_features)
        test_features = pd.get_dummies(test_features)
        # 特徴量（列）によって、学習特徴量とテスト特徴量を整列（内部結合）
        train_features, test_features = train_features.align(test_features, join = 'inner', axis = 1)
        
        # No categorical indices to record
        cat_indices = 'auto'
    
    # 整数ラベルエンコーディング
    elif encoding == 'le':
        # ラベルエンコーダーのインスタンスを作成
        label_encoder = LabelEncoder()   
        # カテゴリ変数のインデックスを格納するリスト
        cat_indices = []   
        # Iterate through each column
        for i, col in enumerate(train_features):
            if train_features[col].dtype == 'object':
                # Map the categorical features to integers
                train_features[col] = label_encoder.fit_transform(np.array(train_features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)
    
    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    print('Training Data Shape: ', train_features.shape)
    print('Testing Data Shape: ', test_features.shape)

    # 学習に用いる特徴量名のリストを取得
    feature_names = list(train_features.columns)
    # numpy配列に変換
    train_features = np.array(train_features)
    test_features = np.array(test_features)
    # kフォールド交差検証のためのインスタンスを生成
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)
    # 特徴量の重要度を格納するnumpy配列
    feature_importance_values = np.zeros(len(feature_names)) 
    # テスト時の推定結果を格納するnumpy配列
    test_predictions = np.zeros(test_features.shape[0])
    # 検証データ（Out-of-Fold）に対する予測を格納するnumpy配列
    out_of_fold = np.zeros(train_features.shape[0])
    # 学習時のスコアと検証時のスコアを格納するリスト
    train_scores = []
    valid_scores = []
    # フォールドの繰り返し
    for train_indices, valid_indices in k_fold.split(train_features):
        # フォールドにおける学習データ
        train_X, train_y = train_features[train_indices], labels[train_indices]
        # フォールドにおける検証データ
        valid_X, valid_y = train_features[valid_indices], labels[valid_indices]
        # モデルの作成
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
        # モデルの学習
        model.fit(train_X, train_y, eval_metric='auc',
                  eval_set = [(valid_X, valid_y), (train_X, train_y)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)
        # 最も良いスコアをマークしたイテレーション
        best_iteration = model.best_iteration_
        # モデルの特徴量の重要度を記録
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        # テストデータに対する予測
        test_predictions += model.predict_proba(test_features, num_iteration=best_iteration)[:, 1] / k_fold.n_splits
        # 検証データ（Out-of-Fold）に対する予測（フォールドごとに検証データに対する結果が格納されていく）
        out_of_fold[valid_indices] = model.predict_proba(valid_X, num_iteration = best_iteration)[:, 1]
        """out_of_fold: (num_train_sample,)"""
        # 最も良いスコアを記録
        train_score = model.best_score_['train']['auc']
        valid_score = model.best_score_['valid']['auc']
        train_scores.append(train_score)
        valid_scores.append(valid_score)
        # モデルとデータフレームを削除してメモリー解放（処理を軽くするため）
        gc.enable()
        del model, train_X, valid_X
        gc.collect()
    # データフレームを削除してメモリー解放
    gc.enable()
    del train_features
    gc.collect()
        
    # 提出用のデータフレーム
    submission = pd.DataFrame({id_column: test_ids, target_column: test_predictions})
    # 特徴量の重要度を示すデータフレーム
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    # 検証データ（Out-of-Fold）全体に対するスコア
    valid_auc = roc_auc_score(labels, out_of_fold)
    # 全体に対する（最終的な）学習スコアと検証スコアを追加
    train_scores.append(np.mean(train_scores))
    valid_scores.append(valid_auc)
    # 学習スコアと検証スコアを確認するためのデータフレームを作成
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics





"""
機械学習手法
"""
# LightGBM
import lightgbm as lgb
def lgbm(X_train, X_valid, X_test, y_train, y_valid, categorical_features):
    """
    X_train: 学習特徴量のデータフレーム （num_samples, num_features）
    X_valid: 検証特徴量のデータフレーム （num_samples, num_features）
    X_valid: 検証特徴量のデータフレーム （num_samples, num_features）
    categorical_features: カテゴリ変数 （Python List）
    """
    params = {
    'objective': 'binary',
    'max_bin': 427,
    'learning_rate': 0.05,
    'num_leaves': 79
    }
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features)

    model = lgb.train(params, lgb_train, 
                  valid_sets=[lgb_train, lgb_eval], 
                  verbose_eval=10, 
                  num_boost_round=1000, 
                  early_stopping_rounds=10,
                  random_state=0
                  )
    
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    return y_pred


# 回帰(教師あり学習)
# 線形回帰()
def linear_regression(X_train, X_test, y_train, y_test):
    """
    args:
        X_train, X_test: Features matrix (shape: [n_samples, n_features])
        y_train, y_test: Target array (shape: [n_samples,])

    return:
        y_model: predicted result of model
    """
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train) # モデルの学習
    y_model = model.predict(X_test) # モデルの推論
    plt.scatter(X_train, y_train) # 学習データをプロット
    plt.plot(X_test, y_model) # テスト結果を直線でつないでプロット
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    return y_model


# 分類(教師あり学習)
# ナイーブベイズ分類(各ラベルからのデータが単純なガウス分布や多項分布に基づいていると仮定)
# 多項分布に基づくテキスト分類
class NaiveBayes():
    def __init__(self):
        self.model = make_pipeline(TfidfVectorizer(), MultinomialNB()) # TF-IDFによる単語の重み付け(ベクトル化)+多項分布ナイーズベイズ

    def fit(self, data, target):
        self.model.fit(data, target) # モデル学習

    def text_category_classification(self, text, labels):
        pred = self.model.predict([text]) # 予測したラベルのインデックスを取得
        category = labels[pred[0]]
        return category


# サポートベクターマシンによる点群データの分類
class SupportVectorClassify():
    def __init__(self):
        # モデルのインスタンスを生成　
        # C：ソフトマージン用のパラメータ(マージンの中にポイントがどれだけ入れるかを指定する 大きいほど許容数が少なく、小さいほど許容数が多い)
        self.model = SVC(kernel='linear', C=1E10) # linear:線形分類
        # self.model = SVC(kernel='rbf', C=1E10) # rbf(radial basis function)：放射基底関数(カーネルトリックを用いたカーネル変換)

    def fit(self, X, y):
        self.model.fit(X, y) # モデル学習

    # サポートベクターマシンの決定直線を描画
    def plot_svc_decision_function(self, X, y, ax=None, plot_support=True):
        plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
        if ax is None:
            ax = plt.gca() # get current axis

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x = np.linspace(xlim[0], xlim[1], 30)
        y = np.linspace(ylim[0], ylim[1], 30)
        Y, X = np.meshgrid(y, x) # 格子点の座標データを生成

        xy = np.vstack([X.ravel(), Y.ravel()]).T

        P = self.model.decision_function(xy).reshape(X.shape)

        ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--']) # 決定境界を描画

        if plot_support:
            ax.scatter(self.model.support_vectors_[:, 0],self.model.support_vectors_[:, 1],
             s=300, linewidth=1, facecolors='none', edgecolor='black') # サポートベクターを黒丸で囲む
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        plt.show()


# ランダムフォレストによる分類or回帰



# クラスタリング(教師なし学習)



# 次元削減(教師なし学習)
class DimReductionPCA():
    def __init__(self, n_components):
        """
        n_components: Number of components
        """
        self.model = PCA(n_components)

    def fit(self, X):
        self.model.fit(X)

    def cal_cumulative_contribuntion_rate(self, X):
        """
        データを記述するのに必要は成分数を推定するために
        成分数に対する累積寄与率を表示
        """
        pca = PCA().fit(X)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()


# グリッドサーチによるハイパーパラメータの探索(主成分分析とサポートベクターマシンを用いた顔認識)
class GridSearch_PCA_SVC():
    def __init__(self, n_components, param_grid):
        pca = PCA(n_components=n_components, whiten=True, random_state=42, svd_solver='randomized')
        svc = SVC(kernel='rbf', class_weight='balanced')
        self.model = make_pipeline(pca, svc)
        self.grid = GridSearchCV(self.model, param_grid)

    def fit(self, X, y):
        self.grid.fit(X, y)
        print("best_params:", self.grid.best_params_)

    def predict(self, X):
        result = self.grid.best_estimator_.predict(X)
        return result

"""
図に各関数のグラフをプロット
"""
# 任意のグラフを表示する
def fig_plot(x, y, x_scale=1.0, y_scale=0.10, ylim_min=-0.5, ylim_max=0.5, fig_title=None, x_label_name='x', y_label_name='y'):
    # 図に描画
    sns.set() # スタイルをきれいにする
    fig = plt.figure(facecolor='w', linewidth=1, edgecolor='black')
    # ax = fig.add_subplot(1, 1, 1, title=fig_title) # 図を1行目1列の1番目に表示(図を1つしか表示しない場合)
    ax = fig.add_subplot(1, 1, 1, title=fig_title, ylim=(ylim_min, ylim_max)) # 図を1行目1列の1番目に表示(図を1つしか表示しない場合)
    ax.set_xlabel(x_label_name) # x軸名を設定
    ax.set_ylabel(y_label_name) # y軸名を設定
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(x_scale)) # x軸の主目盛を1.0ごとに表示
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(y_scale)) # y軸の主目盛を0.10ごとに表示
    ax.scatter(x, y) # データをプロット
    plt.show()
    # ax.legend(edgecolor="black") # 凡例を追加
    # file_name = os.path.basename(output_path).split('.')[0] # データの名前を設定
    # fig.savefig(output_path) # グラフを保存

# 混同行列の作成
def mk_confusion_matrix(label, predicted):
    """
    label: true label of data
    predicted: predicted result of data
    """
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(label, predicted)

    sns.heatmap(matrix, square=True, annot=True, cbar=False, cmap='RdPu')
    plt.xlabel('predicted value')
    plt.ylabel('true value')
    plt.show()


# 正弦波(sin関数)を描画
def sin(x):
    """
    plot sin wave
    """
    sin_x = np.sin(x)
    plt.plot(x, sin_x)
    plt.xlabel("x")
    plt.ylabel("sin_x")
    plt.show()

# 正規分布(ガウス分布を描画)
def gaussian(x, mean, std):
    """
    plot gaussian distibution
    """
    px = (1/np.sqrt(2*np.pi*std**2))*np.exp(-(x-mean)**2/(2*std**2))

    sns.set()
    plt.plot(x, px)
    plt.xlabel("x")
    plt.ylabel("px")
    plt.show()

# 3次元点群を描画
def plot_3D(X, y, elev=30, azim=30):
    r = np.exp(-(X**2).sum(1)) # 中央の点群を中心とする放射基底関数を使用
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')
    plt.show()

if __name__ == "__main__":
    # 図のスタイルを変更
    sns.set()

    # ガウス分布用
    # x = np.arange(-5, 5, 0.1)
    # mean = 0
    # std = 1
    #
    # gaussian(x, mean, std)

    #　三角関数用
    # x = np.arange(-5, 5, 0.1)
    # sin(x)

    # 3次元点群描画用
    # X, y = make_circles(100, factor=.1, noise=.1)
    # plot_3D(X, y)


    # 線形回帰用
    # x = 10 * np.random.rand(50)
    # X = x[:, np.newaxis] # shape: [n_samples,] → [n_samples, n_features]の形式へ
    # y = 3 * x + np.random.randn(50)
    # X1, X2, y1, y2 = train_test_split(X, y, random_state=0, test_size=0.33) # データをtrain用とtest用に2:1で分ける
    # output = linear_regression(X1, X2, y1, y2)

    # 多項分布ナイーブベイズに基づくテキストカテゴリ分類用
    # from sklearn.datasets import fetch_20newsgroups
    # categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space', 'comp.graphics']
    # train = fetch_20newsgroups(subset='train', categories=categories)　# データ取得
    # test = fetch_20newsgroups(subset='test', categories=categories)　# データ取得
    # naive_bayes = NaiveBayes() # モデル初期化
    # naive_bayes.fit(train.data, train.target) # モデル学習
    # predicted_result = naive_bayes.text_category_classification('sending a payload to the ISS', train.target_names) #　テキスト分類
    # print(predicted_result)

    # サポートベクターマシンによる点群分類用
    # X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
    # points_svc = SupportVectorClassify() # モデル初期化
    # points_svc.fit(X, y)
    # points_svc.plot_svc_decision_function(X, y)

    # # グリッドサーチによるハイパーパラメータの探索(主成分分析とサポートベクターマシンを用いた顔認識)用
    # from sklearn.datasets import fetch_lfw_people
    # faces = fetch_lfw_people(min_faces_per_person=60)
    # Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)
    # n_components = 150 # 主成分分析の次元数
    # param_grid = {'svc__C': [1, 5, 10, 50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]} # グリッドサーチで最適値を探索するパラメータと範囲を指定
    # gps = GridSearch_PCA_SVC(n_components, param_grid)
    # gps.fit(Xtrain, ytrain)
    # yfit = gps.predict(Xtest)
    # # print(yfit)
    # fig, ax = plt.subplots(4, 6)
    # for i, axi in enumerate(ax.flat):
    #     axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    #     axi.set(xticks=[], yticks=[])
    #     axi.set_ylabel(faces.target_names[yfit[i]].split()[-1], color='black' if yfit[i] == ytest[i] else 'red')
    # fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)
    # plt.show()

    # # 主成分分析による次元削減
    # # ランダムな点群
    # n_components = 2
    # rng = np.random.RandomState(1)
    # X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
    # pca = DimReductionPCA(n_components=n_components)
    # pca.fit(X) # 学習（次元削減）
    # print(pca.model.components_) # 各成分のベクトル
    # print(pca.model.explained_variance_) # 各成分の寄与率
    # fig_plot(X[:, 0], X[:, 1])
    # # 手書き数字画像
    # from sklearn.datasets import load_digits
    # digits = load_digits()
    # """
    # digits: (n_samples=1797, n_features=64)
    # """
    # pca = DimReductionPCA(n_components=2)
    # projected = pca.model.fit_transform(digits.data) # 次元削減
    # """
    # projected: (n_samples=1797, n_features=2)
    # """
    # pca.cal_cumulative_contribuntion_rate(digits.data) # 累積寄与率の表示（n_componentsの最適数を定めるため）

    # # 混同行列描画用
    # label = [1, 2, 3, 4, 5, 6, 7, 8]
    # predicted = [1, 3, 3, 5, 5, 8, 7, 6]
    # mk_confusion_matrix(label, predicted)

    # # ランダムフォレストによる回帰
    # from sklearn.ensemble import RandomForestRegressor
    # x = 10 * np.random.rand(200)
    # def sin_model(x, sigma=0.2):
    #     """大きな波＋小さな波＋ノイズからなるダミーデータ"""
    #     noise = sigma * np.random.randn(len(x))
    #     return np.sin(5 * x) + np.sin(0.5 * x) + noise
    # y = sin_model(x)
    # plt.figure(figsize=(16,8))
    # plt.errorbar(x, y, 0.1, fmt='o') # 元のデータをプロット
    # plt.show()
    # # 確認用に0〜10の1000個のデータを用意
    # xfit = np.linspace(0, 10, 1000)
    # # ランダムフォレスト実行
    # rfr = RandomForestRegressor(100)  # インスタンスの生成　木の数を100個に指定
    # rfr.fit(x[:, None], y) # モデルをデータに適合　x:(n_samples) → x[:, None]:(n_samples, n_features=1)
    # yfit = rfr.predict(xfit[:, None])
    # # 結果比較用に実際の値を取得。
    # ytrue = sin_model(xfit,0) # xfitを波発生関数に食わせて、その結果を取得
    # # 結果確認
    # plt.figure(figsize = (16,8))
    # plt.errorbar(x, y, 0.1, fmt='o')
    # plt.plot(xfit, yfit, '-r')                # 予測値のplot（赤）
    # plt.plot(xfit, ytrue, '-k', alpha = 0.5)  # 正解値のplot（黒）
    # plt.show()

    # # 学習データと検証データの分割
    # train = pd.read_csv("../kaggle/titanic/train.csv")
    # test = pd.read_csv("../kaggle/titanic/test.csv")
    # X_train = train.drop("Survived", axis=1)
    # y_train = train["Survived"]
    # X_tr, X_va, y_tr, y_va = train_valid_sep_kfold(X_train, y_train)



    train = pd.read_csv("../kaggle/titanic/train.csv")
    test = pd.read_csv("../kaggle/titanic/test.csv")
    submission, feature_importances, metrics = model(train, test, id_column='PassengerId', target_column='Survived')
    print('Baseline metrics')
    print(metrics)

