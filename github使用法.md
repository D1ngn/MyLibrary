## github使用法



#### リポジトリの作成と登録

1. ブラウザでgithubにログインし、新しいリポジトリを作成

2. 自分のローカルPCにて以下を実行し、gitのローカルリポジトリを初期化

   ```
   $ git init
   $ git config --global user.name [ユーザ名]
   $ git config --global user.email [メールアドレス]
   $ git remote add origin https://github.com/[ユーザ名]/[リポジトリ名]
   ```

3. 以下のコマンドで`~/.ssh/id_rsa.pub`にsshキーが生成されるので、その中身(ssh-rsaから始まる部分)をgithubの公開鍵のページ([https://github.com/settings/keys](https://github.com/settings/keys))に登録する。鍵名は`id_rsa.pub`で良い。(2個目以降のリポジトリを作る際には必要ない)

   ```
   $ ssh-keygen -t rsa
   ```

   次に、以下のコマンドで接続の確認を行う。

   ```
   $ ssh -T git@github.com
   ```

   「Hi (account名)! You've successfully authenticated, but GitHub does not provide shell access.」と返ってきたら成功

4. GitHubと実際にやりとりするときにID・パスワードを聞かれるときssh接続がうまくいっていないので、

   ```
   $ git config remote.origin.url
   ```

   で確認し　`https://github.com/[ユーザID]/[リポジトリ].git`となっていたら、

   ```
   $ git remote set-url origin git@github.com:[ユーザID]/[リポジトリ].git
   ```

   とする。



#### githubを使用した作業

・リモートリポジトリの変更をローカルに反映

```
$ git pull origin master
```





#### gitで管理したくないファイルの登録

・作業ディレクトリ内に`.gitignore`というファイルを作成し、テキストエディタなどを利用し、管理したくないファイル名やディレクトリ名を書き込む

(例)
ファイルの場合
file.txt

ディレクトリの場合
dir/

というように書き込む





#### リポジトリ名変更

1. 名前を変更したいgithubのリポジトリをブラウザで開き、SettingからRenameを行う

2. githubのリポジトリと結びついているローカルディレクトリ内にある`.git/config`をテキストエディタで開き、以下のように変更する

   ```
   [remote "origin"]
   	url = git@github.com:D1ngn/[新しいリポジトリ名].git
   	fetch = +refs/heads/*:refs/remotes/origin/*
   ```
