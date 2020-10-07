import os

# 機械学習用
import numpy as np
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
機械学習手法
"""
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
    y_model = model.predict(X_test) #　モデルの推論
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
