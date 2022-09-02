
# 機械学習用
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


"""
特徴量エンジニアリング
"""
# 欠損値の処理
def missing_values_processing(df, fill_value=None, mode=None):
    # 欠損値を特定の値で補完
    if mode == 'fill' and fill_value != None:
        df = df.fillna(fill_value) 
    return df

# 列に何らかの処理を加えて得られた列をdfに追加
def apply_column(df, base_column_name, add_column_name):
    """
    df: pandasのデータフレーム （num_samples, num_features）
    base_column_name: 基となる列名 str型
    add_column_name: 新たに追加する列名 str型（列を上書きする場合は基となる列名（base_column_name）を指定）
    """
    # 列に何らかの処理を加えて得られた列を追加
    df[add_column_name] = df[base_column_name].apply(lambda x: 2*x) # サンプル：2をかけたものを返す例

    return df

# 複数の列に何らかの処理を加えて得られた列をdfに追加
def apply_multiple_column(df, base_column_name_1, base_column_name_2, add_column_name):
    """
    df: pandasのデータフレーム （num_samples, num_features）
    base_column_name: 基となる列名 str型
    add_column_name: 新たに追加する列名 str型（列を上書きする場合は基となる列名（base_column_name）を指定）
    """
    # 行（row）ごとに処理
    def apply_each_row(row):
        new_val = row[base_column_name_1] + row[base_column_name_2] # サンプル：同じ行に含まれる2つの列の値を足し合わせる例
        return new_val
    # データフレーム全体にapplyを適用（axis=1で行ごとに適用）
    df[add_column_name] = df.apply(apply_each_row, axis=1)
    return df

# 列をグループ化した後、グループごとに特定の特徴量の値を集計して要約統計量を作成
def agg_features_groupby(df, groupby_column, features_columns):
    """
    df: pandasのデータフレーム （num_samples, num_features）
    groupby_column: グループ化に使用する列名 str型
    features_columns: 集計を行う列名 list型（リストにstr型の列名を格納したもの）
    """
    # 平均、標準偏差、最小値、最大値、最後尾の値、中央値
    agg_df = df.groupby(groupby_column)[features_columns].agg(['mean', 'std', 'min', 'max', 'last', 'median'])
    agg_df.columns = ['_'.join(x) for x in agg_df.columns]
    return agg_df

from sklearn.preprocessing import LabelEncoder # カテゴリ変数をラベルエンコーディングするためのライブラリ
# カテゴリ変数のエンコード
def cat_encoding(train_features, test_features, encoding='ohe'):
    # One-Hotエンコーディング
    if encoding == 'ohe':
        train_features = pd.get_dummies(train_features)
        test_features = pd.get_dummies(test_features)
        # 特徴量（列）によって、学習特徴量とテスト特徴量を整列（内部結合）
        train_features, test_features = train_features.align(test_features, join = 'inner', axis = 1)
        # カテゴリ変数は記録しない（デフォルトのまま）
        cat_indices = 'auto'
    # 整数ラベルエンコーディング
    elif encoding == 'le':
        # ラベルエンコーダーのインスタンスを作成
        label_encoder = LabelEncoder()   
        # カテゴリ変数のインデックスを格納するリスト
        cat_indices = []   
        # 特徴量（列）ごとの繰り返し処理
        for i, col in enumerate(train_features):
            if train_features[col].dtype == 'object':
                # カテゴリ変数を整数にマッピング
                train_features[col] = label_encoder.fit_transform(np.array(train_features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))
                # カテゴリ変数のインデックスを保存
                cat_indices.append(i)
    # エンコーディングのモードが正しくない場合
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
    return train_features, test_features, cat_indices


# # データ加工
# def data_processing(train_features, test_features, encoding):

#     df = cat_encoding(train_features, test_features, encoding)

#     return train_features, test_features, cat_indices


"""
モデルの定義
"""
# LightGBM（Training APIバージョン）
import lightgbm as lgb
def my_lgbm(train_X, train_y, valid_X, valid_y, metrics, cat_indices):
    """
    train_X: 学習特徴量のデータフレーム （num_samples, num_features）
    valid_X: 検証特徴量のデータフレーム （num_samples, num_features）
    X_valid: 検証特徴量のデータフレーム （num_samples, num_features）
    cat_indices: カテゴリ変数のインデックスのリスト （'auto' or Python List）
    """
    # ハイパーパラメータの設定
    params = {
    'objective': 'binary',
    'seed': 0,
    'verbose': 0,
    'metrics': metrics,
    'max_bin': 427,
    'learning_rate': 0.05,
    'num_leaves': 79
    }
    # 使用データセットの指定
    lgb_train = lgb.Dataset(train_X, train_y)
    lgb_eval = lgb.Dataset(valid_X, valid_y, reference=lgb_train)
    # モデルの作成と学習学習
    model = lgb.train(params, lgb_train,
                    categorical_feature=cat_indices, 
                    valid_names = ['train', 'valid'],
                    valid_sets=[lgb_train, lgb_eval], 
                    verbose_eval=10, 
                    num_boost_round=1000, 
                    early_stopping_rounds=10
                    )
    return model


# LightGBM（Scikit-learn APIバージョン）
def my_lgbm_sklearn_api(train_X, train_y, valid_X, valid_y, metrics, cat_indices):
    """
    train_X: 学習特徴量のデータフレーム （num_samples, num_features）
    valid_X: 検証特徴量のデータフレーム （num_samples, num_features）
    X_valid: 検証特徴量のデータフレーム （num_samples, num_features）
    cat_indices: カテゴリ変数のインデックスのリスト （'auto' or Python List）
    """
    # モデルの作成
    model = lgb.LGBMClassifier(n_estimators=10000, objective='binary', 
                                class_weight='balanced', learning_rate = 0.05, 
                                reg_alpha = 0.1, reg_lambda = 0.1, 
                                subsample = 0.8, n_jobs = -1, random_state = 50)
    # モデルの学習
    """
    categorical_feature (default='auto')：整数のリストを渡した場合は特徴量（列）のインデックス、
                                            文字列のリストを渡した場合は特徴量（列）名と判断される
                                            'auto'を渡した場合はデータ型が'category'のものが用いられる
                                            負の値は欠損値と判断される
    """
    model.fit(train_X, train_y, eval_metric=metrics,
                eval_set=[(valid_X, valid_y), (train_X, train_y)],
                eval_names=['valid', 'train'], categorical_feature=cat_indices,
                early_stopping_rounds=100, verbose=200)

    return model




"""
モデルの学習・評価
"""
import lightgbm as lgb
import gc # メモリ管理用（ガベージコレクション：必要なくなったメモリ領域を自動的に開放する機能）
import warnings # warningsを非表示
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, log_loss

# モデルの学習・交差検証・評価
def model(train_features, test_features, id_column, target_column, metrics='binary_logloss', encoding='ohe', n_folds=5, 
            model_api='training_api'):
    
    """
    Train and test a light gradient boosting model using cross validation. 
    
    Args:
        train_features (pd.DataFrame): 学習特徴量のデータフレーム（目的変数の列を含む） （num_samples, num_features）
        test_features (pd.DataFrame): 評価特徴量のデータフレーム （num_samples, num_features）
        id_column （str）: IDに相当する列名
        target_column （str）: 目的変数の列名 
        metrics（str）: 交差検証時に用いる評価指標
        encoding (str, default = 'ohe'): カテゴリ変数をエンコード方法を指定
                                        ・'ohe': ワンホットエンコード
                                        ・'le': ラベルエンコーディング
        n_folds (int, default = 5): 交差検証に用いるホールド数
        
    Return:
        submission (pd.DataFrame): id_columnとモデルが推定したtarget_columnを含む提出用のデータフレーム
        feature_importances (pd.DataFrame): モデルを構築する際の各特徴量の重要度を含んだデータフレーム
        metrics_results (pd.DataFrame): 各フォールドとそれらを総合した学習スコア・検証スコアを含むデータフレーム
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

    # 特徴量エンジニアリング（欠損値の処理、カテゴリ変数の処理、基本統計量を用いた特徴量の作成など）
    # カテゴリ変数のエンコード
    train_features, test_features, cat_indices = cat_encoding(train_features, test_features, encoding)
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
    # 交差検証における各フォールドの繰り返し
    for train_indices, valid_indices in k_fold.split(train_features):
        # フォールドにおける学習データ
        train_X, train_y = train_features[train_indices], labels[train_indices]
        """train_X: (num_samples, num_features), train_y: （num_samples, 1)"""
        # フォールドにおける検証データ
        valid_X, valid_y = train_features[valid_indices], labels[valid_indices]
        """valid_X: (num_samples, num_features), valid_y: （num_samples, 1)"""
        # LightGBM（Training APIバージョン）の学習
        model = my_lgbm(train_X, train_y, valid_X, valid_y, metrics, cat_indices)
        # LightGBM（Scikit-learn APIバージョン）の学習
        # model = my_lgbm_sklearn_api(train_X, train_y, valid_X, valid_y, metrics, cat_indices)
        # 使用するAPIごとの処理
        if model_api == 'training_api':
            # 最も良いスコアをマークしたイテレーション
            best_iteration = model.best_iteration
            # モデルの特徴量の重要度を記録（importance_typeはデフォルトよりもgainの方が良い？）
            feature_importance_values += model.feature_importance(importance_type='gain') / k_fold.n_splits
            # テストデータに対する予測
            test_predictions += model.predict(test_features, num_iteration=best_iteration) / k_fold.n_splits
            # 検証データ（Out-of-Fold）に対する予測（フォールドごとに検証データに対する結果が格納されていく）
            out_of_fold[valid_indices] = model.predict(valid_X, num_iteration=best_iteration)
            """out_of_fold: (num_train_sample,)"""
            # 最も良いスコアを記録
            train_score = model.best_score['train'][metrics]
            valid_score = model.best_score['valid'][metrics]
        elif model_api == 'sklearn_api':
            # 最も良いスコアをマークしたイテレーション
            best_iteration = model.best_iteration_
            # モデルの特徴量の重要度を記録
            feature_importance_values += model.feature_importances_ / k_fold.n_splits
            # テストデータに対する予測
            test_predictions += model.predict_proba(test_features, num_iteration=best_iteration)[:, 1] / k_fold.n_splits
            # 検証データ（Out-of-Fold）に対する予測（フォールドごとに検証データに対する結果が格納されていく）
            out_of_fold[valid_indices] = model.predict_proba(valid_X, num_iteration=best_iteration)[:, 1]
            """out_of_fold: (num_train_sample,)"""
            # 最も良いスコアを記録
            train_score = model.best_score_['train'][metrics]
            valid_score = model.best_score_['valid'][metrics]
        else:
            print("Please specify correct 'model_api'")
        # 学習スコアと検証スコアを格納
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
    # 検証データ（Out-of-Fold）全体に対するスコア（評価指標は交差検証時とは異なる場合あり）
    valid_auc = log_loss(labels, out_of_fold)
    # 全体に対する（最終的な）学習スコアと検証スコアを追加
    train_scores.append(np.mean(train_scores))
    valid_scores.append(valid_auc)
    # 学習スコアと検証スコアを確認するためのデータフレームを作成
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    metrics_results = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics_results



"""
評価・可視化
"""
# モデルが返却した特徴量の重要度を可視化（重要度は大きいほど良いことを表す）
def plot_feature_importances(df, num_display_features=15):
    """
    Args:
        df (pd.dataframe): 特徴量の重要度を表すデータフレーム
        num_display_features (int): 重要度を表示する特徴量の数
        
    Returns:
        df (dataframe): 重要度順にソートされた特徴量の重要度 (降順) 
                        特徴量は正規化される
    """
    # 重要度順に特徴量を並び替え
    df = df.sort_values('importance', ascending=False).reset_index()
    # 特徴量の重要度を正規化
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    # 水平方向の棒グラフを作成
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    # 重要度が最大のものが上方にくるようにインデックスを逆転
    ax.barh(list(reversed(list(df.index[:num_display_features]))), 
            df['importance_normalized'].head(num_display_features), 
            align='center', edgecolor='k')
    # タイトルとラベルを表示
    ax.set_yticks(list(reversed(list(df.index[:num_display_features]))))
    ax.set_yticklabels(df['feature'].head(num_display_features))
    plt.title('Feature Importances')
    plt.xlabel('Normalized Importance')
    plt.show()
    return df


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


if __name__ == "__main__":
    # 図のスタイルを変更
    sns.set()

    # # 学習データと検証データの分割
    # train = pd.read_csv("../kaggle/titanic/train.csv")
    # test = pd.read_csv("../kaggle/titanic/test.csv")
    # X_train = train.drop("Survived", axis=1)
    # y_train = train["Survived"]
    # X_tr, X_va, y_tr, y_va = train_valid_sep_kfold(X_train, y_train)

    # 機械学習フローのテスト
    train = pd.read_csv("../../kaggle/titanic/train.csv")
    test = pd.read_csv("../../kaggle/titanic/test.csv")
    submission, feature_importances, metrics = model(train, test, id_column='PassengerId', target_column='Survived', model_api='training_api')
    print('Baseline metrics')
    print(metrics)
    plot_feature_importances(feature_importances)
    submission['Survived'] = (submission['Survived'] > 0.5).astype(int) # 0.5より大きい場合1、0.5以下の場合0に変換
    submission.to_csv("../../kaggle/titanic/submission.csv", index=False)

