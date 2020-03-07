import tensorflow as tf
import numpy as np
from rnn_model import EvalModel
import utils
#指定验证时不用cuda,这样可以在用gpu训练的同时使用cpu进行验证
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
x_data = tf.placeholder(tf.int32,[1,None])
emb_keep = tf.placeholder(tf.float32)
rnn_keep = tf.placeholder(tf.float32)

#验证使用模型
model = EvalModel(x_data,emb_keep,rnn_keep)

saver = tf.train.Saver()

#单词映射到id
work_to_id = utils.read_word_to_id_dict()
#id映射到单词
id_to_work = utils.read_id_to_word_dict()

def generate_work(prob):
    """
    选择概率最高的前100个词，并用轮盘赌法选取最终结果
    :param prob:
    :return: 生成的词
    """
    prob = sorted(prob,reverse=True)[:100]
    index = np.searchsorted(np.cumsum(prob),np.random.rand(1) * np.sum(prob))
    return id_to_work[int(index)]
def generate_poem():
    """
    随机生成一首诗
    :return:
    """
    with tf.Session() as sess:
        #加载最新模型

        ckpt = tf.train.get_checkpoint_state('ckpt')
        saver.restore(sess,ckpt.model_checkpoint_path)
        #预测第一个词
        rnn_state = sess.run(model.cell.zero_state(1,tf.float32))
        x = np.array([[work_to_id['s']]],np.int32)
        prob, rnn_state = sess.run([model.prob, model.last_state],
                                   {model.data: x, model.init_state: rnn_state, model.emb_keep: 1.0,
                                    model.rnn_keep: 1.0})
        word = generate_work(prob)
        poem = ''
        #循环操作，直到出现结束符号‘e’
        while word != 'e':
            poem += word
            x = np.array([[work_to_id[word]]])
            prob,rnn_state = sess.run([model.prob, model.last_state],
                                       {model.data: x, model.init_state: rnn_state, model.emb_keep: 1.0,
                                        model.rnn_keep: 1.0})
            word = generate_work(prob)
        wither = tf.summary.FileWriter('./logs/', sess.graph)
        wither.close()
        #打印生成的诗
        print(poem)

def generate_acrostic(head):
    """
    生成藏头诗
    :param head:每行的第一个字符
    :return:
    """
    with tf.Session() as sess:
        # 加载最新模型
        ckpt = tf.train.get_checkpoint_state('ckpt')
        saver.restore(sess, ckpt.model_checkpoint_path)
        #进行预测
        rnn_state = sess.run(model.cell.zero_state(1,tf.float32))
        poem = ''
        for word in head:
            while word != '，' or word != '。':
                poem += word
                x = np.array([[work_to_id[word]]])
                prob, rnn_state = sess.run([model.prob, model.last_state],
                                           {model.data: x, model.init_state: rnn_state, model.emb_keep: 1.0,
                                            model.rnn_keep: 1.0})
                word = generate_work(prob)
                # print(word)
                if word == 'e':
                    break
        wither = tf.summary.FileWriter('./logs/', sess.graph)
        wither.close()
        # 打印生成的诗
        print(poem)


if __name__ == '__main__':
    #藏头诗
    #title = input("请输入藏头诗标题:")
    # title = '月亮'
    # generate_acrostic(title)
    #随机生成诗
    generate_poem()