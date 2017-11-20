#coding:utf-8
"""
python 3.6
tensorflow 1.3
By LiWenDi
"""
#num_layers暂时不进行使用
import time
from collections import namedtuple
from collections import Counter
import numpy as np

import tensorflow as tf

batch_size = 100
num_steps = 100
lstm_size = 512
num_layers = 1#重大问题！！！
learning_rate = 0.001
resume_training = True #继续上次的训练
if_Train = False #进行模型训练
if_Test = True #进行生成文字测试
Shorten = False #如果训练的文本文字种类太多请使用
epochs = 200
save_every_n = 50

with open("text.txt", 'r', encoding='gb18030') as f:
    text = f.read()
if Shorten:
    vocab = [each[0] for each in Counter(text).most_common(4000)]
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    chars = np.array([vocab_to_int[c] for c in text if c in vocab], dtype = np.int32)
else:
    vocab = sorted(set(text),key=text.index)
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    chars = np.array([vocab_to_int[c] for c in text], dtype = np.int32)


print(vocab)
print("字典长度：", len(vocab))

def split_data(chars, batch_size, num_steps, split_frac = 0.9):
    """
    将字符串分为 [训练集，验证集，输入，目标] 这四个。

    Arguments
    ------------------------------------------------------------------------------
    chars:字符串数组
    batch_size:每个batch中样本的个数
    num_steps:保持输入并将其传入网络的输入序列长度
    split_frac:batch中挑选在训练集中的概率
    """

    slice_size = batch_size * num_steps
    n_batches = int(len(chars) / slice_size)

    x = chars[:n_batches*slice_size]
    y = chars[1:n_batches*slice_size+1]

    x = np.stack(np.split(x, batch_size))
    y = np.stack(np.split(y, batch_size))

    split_idx = int(n_batches * split_frac)
    train_x, train_y = x[:, :split_idx*num_steps], y[:, :split_idx*num_steps]
    val_x, val_y = x[:, split_idx*num_steps:], y[:, split_idx*num_steps:]

    return train_x, train_y, val_x, val_y

def get_batch(arrs, num_steps):
    batch_size, slice_size = arrs[0].shape
    n_batches = int(slice_size/num_steps)
    for b in range(n_batches):
        yield [x[:, b*num_steps: (b+1)*num_steps] for x in arrs]

def build_rnn(num_classes, batch_size = 50, num_steps = 50, lstm_size = 120, num_layers = 1, learning_rate = 0.001, grad_clip = 5, sampling = False):
    if sampling == True:
        batch_size, num_steps = 1, 1

    tf.reset_default_graph()

    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name = "inputs")
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name = "targets")
    keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

    x_one_hot = tf.one_hot(inputs, num_classes)
    y_one_hot = tf.one_hot(targets, num_classes)

    lstm = tf.contrib.rnn.BasicLSTMCell(num_units = lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = keep_prob)
    #cell = tf.contrib.rnn.MultiRNNCell([drop]*num_layers)

    initial_state = lstm.zero_state(batch_size, tf.float32)

    rnn_inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(x_one_hot, num_steps, 1)]
    (outputs, state) = tf.contrib.rnn.static_rnn(cell = drop,inputs =  rnn_inputs, initial_state = initial_state)
    final_state = state

    seq_output = tf.concat(outputs, axis=1)
    output = tf.reshape(seq_output, [-1,lstm_size])

    with tf.variable_scope("softmax"):
        softmax_w = tf.Variable(tf.truncated_normal((lstm_size, num_classes), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(num_classes))
        
    logits = tf.matmul(output, softmax_w) + softmax_b
    preds = tf.nn.softmax(logits, name = "predictions")

    y_reshaped = tf.reshape(y_one_hot, [-1, num_classes])
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    cost = tf.reduce_mean(loss)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    export_nodes = ['inputs', 'targets', 'initial_state', 'final_state', 'keep_prob', 'cost', 'preds', 'optimizer']
    Graph = namedtuple('Graph', export_nodes)

    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def sample(checkpoint, n_samples, lstm_size, vocab_size, prime):
    samples = [c for c in prime]
    model = build_rnn(vocab_size, lstm_size=lstm_size, sampling= True)
    saver2 = tf.train.Saver()
    with tf.Session() as sess:
        saver2.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1,1))
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x, model.keep_prob: 1., model.initial_state: new_state}
            preds, new_state = sess.run([model.preds, model.final_state], feed_dict = feed)
            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x, model.keep_prob: 1., model.initial_state: new_state}
            preds, new_state = sess.run([model.preds, model.final_state], feed_dict = feed)
            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])
        sess.close()
    return ''.join(samples)


train_x, train_y, val_x, val_y = split_data(chars, batch_size, num_steps)
model = build_rnn(len(vocab), batch_size=batch_size, num_steps=num_steps, learning_rate=learning_rate, lstm_size=lstm_size,num_layers=num_layers)
saver = tf.train.Saver()
if if_Train:
    with tf.Session() as sess:
        if resume_training:
            saver.restore(sess, 'lstm_model/textMaker.ckpt')
        else:
            sess.run(tf.global_variables_initializer())
    
        n_batches = int(train_x.shape[1]/num_steps)
        iterations = n_batches * epochs
        for e in range(epochs):
            new_state = sess.run(model.initial_state)
            loss = 0
            for b, (x,y) in enumerate(get_batch([train_x,train_y], num_steps), 1):
                iteration = e * n_batches + b
                start = time.time()
                feed = {model.inputs: x, model.targets: y, model.keep_prob: 0.5, model.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([model.cost, model.final_state, model.optimizer], feed_dict = feed)
    
                loss += batch_loss
                end = time.time()
                print('总迭代 {}/{}'.format(e+1, epochs),
                      '子迭代 {}/{}'.format(iteration, iterations),
                      '训练loss：{:.4f}'.format(loss/b),
                      '{:.4f} 秒/批'.format((end-start)))
    
                if(iteration%save_every_n == 0) or (iteration == iterations):
                    val_loss = []
                    new_state = sess.run(model.initial_state)
                    for x,y in get_batch([val_x, val_y], num_steps):
                        feed = {model.inputs: x, model.targets: y, model.keep_prob: 1., model.initial_state: new_state}
                        batch_loss, new_state = sess.run([model.cost, model.final_state], feed_dict = feed)
                        val_loss.append(batch_loss)
    
                    print('验证loss：', np.mean(val_loss))
                    saver.save(sess, "lstm_model/textMaker.ckpt")
        sess.close()



if if_Test:
    checkpoint = "lstm_model/textMaker.ckpt"
    n_samples = 1
    print("\n===================下面开始进行语句生成===================：\n")
    while(n_samples != -1):
        n_samples = input("请输入生成的语句长度（-1表示结束程序）：\n")
        n_samples = int(n_samples)
        if n_samples == -1:
            break
        prime = input("请输入开头字：\n")
        re = False
        if len(prime) > 1:
            for each in prime:
                if each not in vocab:
                    print(each,"不在字典中，请重新输入！")
                    re = True
        else:
            if prime not in vocab:
                print(prime,"不在字典中，请重新输入！")
                re = True
            prime = [prime]
        if not re:
            samp = sample(checkpoint, n_samples, lstm_size,len(vocab), prime)
            print("-------------------------------------------------以下是生成结果-------------------------------------------------\n")
            print(samp)
            print("\n-------------------------------------------------以上是生成结果-------------------------------------------------")
