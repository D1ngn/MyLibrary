-- 基本操作

-- NULLも含めたテーブルのレコード数(行数)をカウント
SELECT COUNT(*) 
FROM tbl
;

-- 特定の列においてNULLも除いたレコード数(行数)をカウント
SELECT COUNT(column_name) 
FROM tbl
;

-- 特定の列においてデータのユニーク件数をカウント
SELECT COUNT(DISTINCT column_name) 
FROM tbl
;

-- カテゴリごとにカウント
SELECT COUNT(*) OVER(PARTITION BY category)
FROM tbl
;

-- 特定の列における欠損値（NULL値）を0に置き換える
SELECT COALESCE(column_name, 0)
FROM tbl
;

-- 列ごとの欠損値数を確認
SELECT 
    SUM(CASE WHEN column_name_1 IS NULL THEN 1 ELSE 0 END) AS column_name_1_null_num,
    SUM(CASE WHEN column_name_2 IS NULL THEN 1 ELSE 0 END) AS column_name_2_null_num
FROM product
LIMIT 10
;

-- 特定の列における中央値を取得
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY column_name)
FROM tbl

 
-- データ（母集団）からランダムに1%のデータをサンプリング
-- random()関数は、行ごとに0.0以上1.0未満の値をランダムに返す
-- 0.01（1%）未満のランダム値が返された行のみを取得することで、ランダムサンプリングを実行
SELECT * 
FROM tbl
WHERE RANDOM() <= 0.01
;


-- 日付・日時データの処理

-- 数値UNIX秒（エポックタイム）からtimestamp型への変換
SELECT TO_TIMESTAMP(unix_time)
FROM tbl
;

-- YYYYMMDD形式の文字列から数値UNIX秒（エポックタイム）への変換
SELECT EXTRACT(EPOCH FROM TO_TIMESTAMP(string_date, 'YYYYMMDD'))
FROM tbl
;

-- timestamp型からdate型への変換
SELECT CAST(time_stamp AS DATE)
FROM tbl
;

-- YYYYMMDD形式の文字列からdate型への変換
SELECT TO_DATE(string_date, 'YYYYMMDD')
FROM tbl
;

-- date型からYYYYMMDD形式の文字列に変換
SELECT TO_CHAR(date_time, 'YYYYMMDD')
FROM tbl
;

-- 'YYYYMMDD'形式の数値型からdate型への変換
SELECT TO_DATE(CAST(ymd AS VARCHAR), 'YYYYMMDD') AS date
FROM tbl
;

-- 数値型から文字列への変換
SELECT CAST(num_date AS integer)
FROM tbl
;

-- 日付型から年・月・日・時・分・秒・曜日を抽出
-- date_timeはdate型またはtimestamp型
SELECT 
    EXTRACT(year FROM date_time),
    EXTRACT(month FROM date_time),
    EXTRACT(day FROM date_time),
    EXTRACT(hour FROM date_time),
    EXTRACT(minute FROM date_time),
    EXTRACT(second FROM date_time),
    EXTRACT(dow FROM date_time)
FROM 
    tbl
;

-- YYYYMMDD形式の文字列からtimestamp型に変換し、年数差（1年未満は切り捨て）を算出
SELECT EXTRACT(year FROM AGE(TO_TIMESTAMP(string_date_1, 'YYYYMMDD'), TO_TIMESTAMP(string_date_2, 'YYYYMMDD'))) AS elapsed_years
FROM tbl

-- 日付の0埋め
SELECT
    TO_CHAR(year, 'FM0000') -- 4桁の場合
    TO_CHAR(month, 'FM00') -- 2桁の場合
    TO_CHAR(day, 'FM00') -- 2桁の場合
FROM 
    tbl
;

-- 日付の差分を算出
-- date_timeはdate型またはtimestamp型
SELECT
    "ID",
    date_time AS "決済日時",
    LEAD("決済日時", 1) OVER(PARTITION BY "ID" ORDER BY "決済日時") AS "次回決済日時",
    DATEDIFF(day, "決済日時", "次回決済日時") AS "決済間隔"
FROM 
    tbl
;


-- 正規表現

-- 「エラー」という文字列が含まれている項目を表示
SELECT * 
FROM store
WHERE status_message LIKE '%エラー%'
;

-- 電話番号（tel_no）が3桁-3桁-4桁のデータを表示
-- 「{n}」は直前の文字をn回繰り返すことを表す
SELECT *
FROM table
WHERE tel_no ~ '[0-9]{3}-[0-9]{3}-[0-9]{4}'
;

-- 先頭がA~Zのアルファベットで始まり、末尾が1~9で終わるデータを表示
-- 「^」は直後の文字が文字列の先頭、「$」は直前の文字が文字列の末尾であることを表す
-- 「.」は任意の1文字、「*」は直前の文字を0回以上繰り返すことを表す（正規表現では「%」は使わない）
SELECT *
FROM tbl
WHERE status_cd ~ '^[A-Z].*[1-9]$' 
;

-- 住所から都道府県名のみを抽出
SELECT SUBSTRING(address, '^.*[都道府県]')
FROM address_tb
;


-- 連番
-- データに連番を振る
SELECT ROW_NUMBER() OVER(ORDER BY column_name) AS rn, *
FROM tbl
;

-- カテゴリごとに連番を振る
SELECT ROW_NUMBER() OVER(PARTITION BY category ORDER BY column_name) AS rn, *
FROM tbl
;


-- テーブルを作成
CREATE TABLE tbl2 AS (
    SELECT * 
    FROM tbl1
);

-- テーブルを削除（IF EXISTSを用いるとテーブルが存在しない場合でもエラーにならない）
DROP TABLE IF EXISTS tbl;
