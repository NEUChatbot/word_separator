# word_separator
This repo includes two kinds of Chinese word separator based on Bi-LSTM and maximum matching.

# Requirements
Everything works normally with following packages.
+ python3
+ tensorflow 1.6.0
+ numpy 1.14.2
+ sklearn 0.19.1
+ pandas 0.22.0


# Useage
## To use Bi-LSTM separator
1. import lstm
2. separator = lstm.BiLSTMWordSeparator()
3. result = separator.separate(str)
## To use maximum matching separator
1. import maximum-matching
2. separator = lstm.MatchingSeparator()
3. result = separator.separate(str)
