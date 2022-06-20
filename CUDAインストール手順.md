# CUDA インストール手順





```
$ sudo apt update
$ sudo apt upgrade
```



- CUDA10.2をインストールする場合

  ```
  $ sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
  $ wget wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
  $ sudo dpkg -i cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
  $ sudo apt update
  $ sudo apt install cuda-10-2 cuda-drivers
  $ sudo reboot
  $ rm cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
  ```

  `apt install`で`cuda`を指定すると最新版がインストールされてしまうので、`cuda-10-2`のようにバージョン指定を行うことに注意



最後に`~/.bashrc`の末尾に以下を追加

```
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

その後、一度ログオフして再度ログイン













## 参考サイト

- https://qiita.com/yukoba/items/4733e8602fa4acabcc35
- https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=deb_network

