import os
import h5py
import numpy as np

from sklearn.model_selection import GridSearchCV # グリッドサーチ用
from sklearn.pipeline import make_pipeline # 複数モデルを連続で実行するためのパイプライン
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor


# グリッドサーチによる交差検証用のパラメータ（ランダムフォレスト用）
def param_grid_rfr():
    ret = {
    'n_estimators': [100, 200],
    'max_depth': [2, 3]
    }

def main():
    pred_resp_path_list = ["pred_resp_1.h5", "pred_resp_2.h5"] # 脳活動情報（１ファイル600秒分）
    imp_path = "./impress.h5" # 印象度の正解ラベル
    meta_path = "./meta.h5" # 好感度の正解ラベル
    save_dir = "./pred_result" # モデルが推測した印象度と好感度を格納するディレクトリ
    os.makedirs(save_dir, exist_ok=True)

    # 印象度の正解ラベルを格納したhdfファイルの読み込み
    with h5py.File(imp_path,'r') as f:
        # hdfファイルの中身を確認
        impression_rating = f['Impress'][...] # 印象度
        """impression_rating: (time=8400, num_features=30)"""

    # 好感度の正解ラベルを格納したhdfファイルの読み込み
    with h5py.File(meta_path,'r') as f:
        # hdfファイルの中身を確認
        favorability_rating = f['Meta'][...] # 好感度
        """favorability_rating: (time=8400, num_features=19)"""

    # 動画データを10分(600秒)ごとに分割したデータの処理
    for idx, pred_resp_path in enumerate(pred_resp_path_list):
        # 脳活動情報を格納したhdfファイルの読み込み
        with h5py.File(pred_resp_path,'r') as f:
            # print('(1)----- 1st level -----')
            # ファイルオブジェクトをイテレートするとファイル直下のオブジェクト名を返す
            # for k in f:
            #     print(k)
            # hdfファイルの中身を確認
            pred_response = f['PredResp'][...] # 脳活動情報
            """pred_response: (time=610, feature_dim=1000)"""

        # # リッジ回帰
        # X = pred_response[10:, :] # 脳活動情報の最初の10秒分を削除
        # """X: (time=600, feature_dim=1000)"""
        # X_test = np.random.randn(600, 1000) # 脳活動情報：0~1までの正規分布に従う乱数
        # """X_test: (time=600, feature_dim=1000)"""
        # # 印象度
        # y_imp = impression_rating[600*idx:600*(idx+1), :] # 600秒分ごとのデータ
        # """y_imp: (time=600, feature_dim=30)"""
        # ridge_imp = Ridge(alpha=1.0).fit(X, y_imp)
        # predicted_impression_rating = ridge_imp.predict(X_test)
        # """predicted_impression_rating: (time=600, feature_dim=30)"""
        # # 好感度
        # y_meta = favorability_rating[600*idx:600*(idx+1), :] # 600秒分ごとのデータ
        # """y_meta: (time=600, feature_dim=19)"""
        # ridge_meta = Ridge(alpha=1.0).fit(X, y_meta)
        # predicted_favorability_rating = ridge_meta.predict(X_test)
        # """predicted_favorability_rating: (time=600, feature_dim=19)"""
        #
        # # モデルの予測した印象度と好感度をhdfファイルに保存
        # save_path = os.path.join(save_dir, "pred_result_{}.h5".format(str(idx).zfill(3)))
        # with h5py.File(save_path, 'w') as f:
        #     dataset1 = f.create_dataset(name='PredImpress', data=predicted_impression_rating)
        #     dataset2 = f.create_dataset(name='PredMeta', data=predicted_favorability_rating)

        # ランダムフォレスト回帰
        num_trees = 100
        X = pred_response[10:, :] # 脳活動情報の最初の10秒分を削除
        """X: (time=600, feature_dim=1000)"""
        X_test = np.random.randn(600, 1000) # 脳活動情報：0~1までの正規分布に従う乱数
        """X_test: (time=600, feature_dim=1000)"""
        # 印象度
        y_imp = impression_rating[600*idx:600*(idx+1), :] # 600秒分ごとのデータ
        """y_imp: (time=600, feature_dim=30)"""
        rfr_imp = RandomForestRegressor(n_estimators=num_trees).fit(X, y_imp)
        # self.grid_imp = GridSearchCV(rfr_imp, param_grid_rfr()).fit(X, y_imp) # グリッドサーチによる交差検証
        # predicted_impression_rating = self.grid_imp.best_estimator_.predict(X_test) # 最高性能のモデルでテスト
        predicted_impression_rating = rfr_imp.predict(X_test)
        """predicted_impression_rating: (time=600, feature_dim=30)"""
        # 好感度
        y_meta = favorability_rating[600*idx:600*(idx+1), :] # 600秒分ごとのデータ
        """y_meta: (time=600, feature_dim=19)"""
        rfr_meta = RandomForestRegressor(n_estimators=num_trees).fit(X, y_meta)
        # self.grid_meta = GridSearchCV(rfr_meta, param_grid_rfr()).fit(X, y_meta) # グリッドサーチによる交差検証
        # predicted_impression_rating = self.grid_meta.best_estimator_.predict(X_test) # 最高性能のモデルでテスト
        predicted_favorability_rating = rfr_meta.predict(X_test)
        """predicted_favorability_rating: (time=600, feature_dim=19)"""

        # モデルの予測した印象度と好感度をhdfファイルに保存
        save_path = os.path.join(save_dir, "pred_result_{}.h5".format(str(idx).zfill(3)))
        with h5py.File(save_path, 'w') as f:
            dataset1 = f.create_dataset(name='PredImpress', data=predicted_impression_rating)
            dataset2 = f.create_dataset(name='PredMeta', data=predicted_favorability_rating)


if __name__ == "__main__":
    main()
