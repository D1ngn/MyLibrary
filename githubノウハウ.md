# github使用方法


## リポジトリ操作


### リポジトリの作成と登録（※2021年8月13日にパスワード認証が廃止されたため一部手順はスキップ）

1. ブラウザでgithubにログインし、新しいリポジトリを作成

2. 自分のローカルPCにて以下を実行し、gitのローカルリポジトリを初期化

   ```
   $ git init
   $ git config --global user.name [ユーザ名]
   $ git config --global user.email [メールアドレス]
   $ git remote add origin https://github.com/[ユーザ名]/[リポジトリ名]
   ```

3. 以下のコマンドで`~/.ssh/id_rsa.pub`にsshキー（公開鍵）が生成されるので、その中身(ssh-rsaから始まる部分)をgithubの公開鍵のページ([https://github.com/settings/keys](https://github.com/settings/keys))に登録する。鍵名は`id_rsa.pub`で良い(2個目以降のリポジトリを作る際には必要ない)。（秘密鍵は自分のPCに保存し、公開しない）

   ```
   $ ssh-keygen -t rsa
   ```

   次に、以下のコマンドで接続の確認を行う。

   ```
   $ ssh -T git@github.com
   ```

   「Hi (account名)! You've successfully authenticated, but GitHub does not provide shell access.」と返ってきたら成功

4. GitHubと実際にやりとりするときにID・パスワードを聞かれるときや、`remote: Permission to [ユーザID]/[リポジトリ].git denied to D1-ngn.ssh`といったエラーが出るときは接続がうまくいっていないので、

   ```
   $ git config remote.origin.url
   ```

   で確認し　`https://github.com/[ユーザID]/[リポジトリ].git`となっていたら、

   ```
   $ git remote set-url origin git@github.com:[ユーザID]/[リポジトリ].git
   ```

   とする。


### 個人アクセストークンの作成（パスワード認証から個人アクセストークン認証に変更されたため）

[このページ](https://zenn.dev/yuri0427/articles/9587ae6a578ee9)を参考に、[githubの個人アクセストークン作成手順](https://docs.github.com/ja/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)を開き、手順に従って個人アクセストークンを作成（個人アクセストークンに適切な権限が付与されていないとpullやpushができなくなるため、個人アクセストークンに付与する権限に注意する必要あり）

※ユーザネームはgithubのユーザネーム（D1ngnなど）、パスワードには上記で生成した個人アクセストークンを入力（求められた場合のみ）


### リモートリモジトリの変更を今いるローカルリポジトリにpull

反映させたいローカルリポジトリに移動し、以下のコマンドを実行

```
git pull [リモートリポジトリ名] [リモートブランチ名]
```

（例）リモートリポジトリのdevelopブランチの内容をローカルリポジトリの今いるブランチにpull

```
git pull --rebase origin develop
```

※`--rebase`オプションをつけることで不要なマージコミットを発生させずに履歴を綺麗に保つことができる

※`git pull`のみを使う際は今いる「ローカルブランチ」とpullしたい「リモートブランチ」が一致していることを確認する必要あり


### ローカルリポジトリの変更内容をcommit

以下コマンドを実行し、`git add -A`で修正したファイル、削除したファイル、新規に追加したファイルなど変更のあった全てのファイルをaddした上で、commitを実行して変更内容を確定させる

```
git add -A
git commit -m "任意のコミットメント"
```

※認証情報などはマスクしてからcommitするよう注意


### ローカルリポジトリの変更内容をリモートリポジトリにpush

以下のコマンドを実行

```
$ git push リポジトリ名 ローカルブランチ名:リモートブランチ名
```

（例）ローカルリポジトリの作業用ブランチ（feature/func1）の内容をリモートリポジトリの作業用ブランチにpush

```
git push origin feature/func1
```

※リモートリポジトリに「feature/func1」というブランチがなくても`git push`コマンドで指定すれば自動で作成される


### gitで管理したくないファイルの登録

作業ディレクトリ内に`.gitignore`というファイルを作成し、テキストエディタなどを利用し、管理したくないファイル名やディレクトリ名を書き込む

(例)`.gitignore`の記述内容
```
ファイルの場合
file.txt

ディレクトリの場合
dir/
```


### リポジトリ名変更

1. 名前を変更したいgithubのリポジトリをブラウザで開き、SettingからRenameを行う

2. githubのリポジトリと結びついているローカルディレクトリ内にある`.git/config`をテキストエディタで開き、以下のように変更する

   ```
   [remote "origin"]
   	url = git@github.com:D1ngn/[新しいリポジトリ名].git
   	fetch = +refs/heads/*:refs/remotes/origin/*
   ```

あるいは、以下のコマンドを実行して直接`.git/config`を変更

```
$ git remote set-url origin https://github.com/D1ngn/[新しいリポジトリ名]
```


### リモートリポジトリの複製方法

cloneしたリポジトリを（コミット履歴を消した状態で）別のリポジトリにpushする方法は下記の通り

1. Githubでコピー先リポジトリを新規作成

2. コピー元リポジトリをローカル端末にクローン

   ```
   git clone https://github.com/username/コピー元リポジトリ
   ```

3. コピー元リポジトリの.git（コミット履歴などの情報）を削除

   ※コミット履歴の削除を行う必要がない場合はスキップ

- Linuxの場合
   ```
   cd コピー元リポジトリのディレクトリ
   rm -rf .git
   ```

- Windowsの場合
   ```
   cd コピー元リポジトリのディレクトリ
   Remove-Item .git -Recurse -Force
   ```

4. ローカルリポジトリの再設定

   ```
   cd コピー元リポジトリのディレクトリ
   git init
   git add -A
   git commit -m ""任意のコミットメント"
   git remote add origin git clone https://github.com/username/コピー先リポジトリ
   ```

5. コピー先リポジトリにpush

   ```
   git push origin [リモートブランチ名]
   ```

**参考**
- https://yuito-blog.com/repository-change/
- https://zenn.dev/akst/articles/github-duplication-commit


## ブランチ操作

- ローカルブランチの一覧を表示
   ```
   git branch
   ```

- リモートブランチの一覧を表示
   ```
   git branch -r
   ``` 

- リモートブランチを含んだブランチの一覧を表示
   ```
   git branch -a
   ``` 

- ローカルブランチを新規作成し、作成したブランチに切り替え
   ```
   git checkout -b <ローカルブランチ名>
   ``` 

- 特定のリモートブランチの内容を特定のローカルブランチ（新規作成）にpull
   ```
   git checkout -b <ローカルブランチ名> origin/<リモートブランチ名>
   ``` 

   （例）リモートのdevelopブランチをローカルのdevelopブランチ（新規作成）にpull
   ```
   git checkout -b develop origin/develop
   ```

   ※リモートリポジトリをクローンした時、pullされるブランチがmainブランチだけなので、他のブランチ（develop）をpullしたい場合に使用
   ※リモートブランチの最新情報がローカルに反映さえれていない場合は以下のコマンドを実行して最新化する
   ```
   git fetch
   ```

- ローカルブランチを削除
   ```
   git branch -d <ローカルブランチ名>
   ```

- リモートブランチを削除
   ```
   git push origin --delete <リモートブランチ名>
   ``` 


## ブランチ戦略


### メインブランチ (main branch)

Gitリポジトリの作成時に作成されるブランチ

Git Flowではメインブランチに直接コミットすることはない

システム開発で、実際にシステムが稼働している環境を本番環境 (production environment) というが、メインブランチは本番環境でシステムが稼働しているプログラムのみが反映されるブランチ

### 開発ブランチ (development branch)

開発ブランチは、本番環境に移行される前のプログラムを保管するためのブランチであり、開発の中心となる

メインブランチと並行しており、削除されることはない

### 機能ブランチ (feature branch)

主に、新しい機能を加える作業を行うためのブランチであり、開発ブランチから派生させる

変更を加えるごとに作成されるため、最も頻繁に作成されるブランチ

### 修正ブランチ (fix branch)

バグの修正を行う際に用いるブランチであり、開発ブランチから派生させる

**参考**
- https://komatsuna4747.github.io/how-to-use-git/branch-strategy.html
