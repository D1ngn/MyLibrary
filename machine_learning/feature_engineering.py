





#############################テキスト特徴量##################################

# N-gram（英単語分割）
def word_ngram(text, n):
    """
    text: input text (ex: "I am a Japanese.")
    """
    ngram = []
    word_list = text.split() # 単語に分割
    for idx in range(len(word_list)):
        ngram_word_list = word_list[idx:idx+n] # n単語ごとにリストにまとめる
        if len(ngram_word_list) == n:
            ngram_list = ' '.join(ngram_word_list)
            ngram.append(ngram_list)
        return ngram

# N-gram（文字分割）
def char_ngram(char, n):
    """
    char: input char (ex: "Supermarket")
    """
    ngram = []
    char_list = list(char) # 単語に分割
    for idx in range(len(char_list)):
        ngram_word_list = char_list[idx:idx+n] # n単語ごとにリストにまとめる
        if len(ngram_word_list) == n:
            ngram_list = ''.join(ngram_word_list)
            ngram.append(ngram_list)
        return ngram


##############################################################################

