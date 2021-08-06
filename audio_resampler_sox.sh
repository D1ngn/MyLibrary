#!/bin/bash

input_files="./input_dir_name/*.wav"
output_dir="./output_dir_name/"

for input_file in ${input_files}; do
  # ファイル名取得
  file_name=$(echo | basename ${input_file})
  # 出力ファイルパス
  output_files=${output_dir}${file_name}
  echo ${output_files}
  # 指定したビット数とサンプリングレートでリサンプル
  # ビット数：16、サンプリングレート：16000、チャンネル数：8、時間：0秒から3秒間
  sox ${input_file} -b 16 -r 16000 -c 8 ${output_files} trim 0 3
done
