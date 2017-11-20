# LSTM_WordsMaker

## Introduction

 - Use LSTM network to judge and generate words. But there is something wrong with `` tf.contrib.rnn.MultiRNNCell `` function so I don't use multiple cells.

 - It can shorten the wordlist if your training data has too many kinds of words. Set `` Shorten `` as _True_.

 - Here is the hyperparameter:

 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/247.jpg?raw=true)

When you train the data or use the trained model I've made, you can order the length of the words you want to generate, and then give the program a start, it can either be a single word or a sentence.

The result examples are shown below:

(Trained from _Buble_)
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/241.jpg?raw=true)
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/242.jpg?raw=true)
  
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/243.jpg?raw=true)
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/246.jpg?raw=true)
 
 ```
As farbhaol wheal has been given to him; and in the same way the priest's poor arm is, 
so that the words of the prophets have put in triet among him.
1Sa 2:3 The same words have given to him, saying that he has been position in his fear of the Lord; 
and the priest went out to her.
Luk 2:37 And at these words, Jesus had given the same.
Rev 1:2 And he went in to her through a tree, so that the ways made a request to another, and saying,
Mat 24:2 What is tourne?
 ```
 
(Trained from _The Romance of the Three Kingdoms_(《_三国演义_》))
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/234.jpg?raw=true)
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/235.jpg?raw=true)
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/236.jpg?raw=true)
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/237.jpg?raw=true)
 
 ```
斬之。遂命解孟達至此觀看。
只見承頓首級，大驚曰：“將軍欲降我獻劉備，我早來見我等報公矣。”
表曰：“汝有心腹之言，何不早降？”
操曰：“吾已死，汝當先生。”
即便傳令差人往長安李傕等赴許都。曹真知兵，急將馬來，奉報不濟。
曹操引兵追來，先被殺條路走，裏遇袁術。殺衆軍奔回，中得三路，驅兵齊進。
 ```
 
(Trained from _Collected works of Lu Xun_(《鲁迅文集》))

 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/248.jpg?raw=true)
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/249.jpg?raw=true)
 
 ```
我们是这样的。

　　我想，是不知道的是一件事，我们的不能有人是一个大家的，我也不能不知道我的话
，我不知道这是因为我是我不愿意的。”

　　他在上海的一个报章上，我在他说的，我的话已经有些：

　　“你看不懂我们中国人的一个小说，我也不是这里面，我们的话，是不可能的。我是
在这时候，我就不能不写他的话。”
 ```
 
*********************

## How to use

 - If you want to train yourself, put the txt with the code and set `` if_Train `` as True. The model will be saved in `` lstm_model `` folder. 

 - If you want to test, put the __correct__ txt with the code too. Then put model in the `` lstm_model `` folder, notice the hyperparameter of the model to be right, then run it and see.
