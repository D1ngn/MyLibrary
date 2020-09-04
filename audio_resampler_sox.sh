#!/bin/bash

input_files="./input_dir_name/*.txt"
output_dir="./output_dir_name/"

for input_file in ${input_files}; do
  # ファイル名取得
  file_name=$(echo | basename ${input_file})
  # 出力ファイルパス
  output_files=${output_dir}${file_name}
  echo ${output_files}
  # 指定したビット数とサンプリングレートでリサンプル
  sox ${input_file} -b 16 -r 16000 -c 4 ${output_files}
done
