import tensorflow as tf
import functools
import config

HIDDEN_SIZE = 128   #LSTM隐藏节点个数
NUM_LAYERS = 2      #RNN深度

#装饰器(功能函数)
def doubleqrap(function):
    @functools.wraps(function)
    def decorator(*args,**kwargs):
        if len(args) == 1 and len(kwargs) ==0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee:function(wrapee,*args,**kwargs)

    return decorator

@doubleqrap
def define_scope(function,scope=None,*args,**kwargs):
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self,attribute):
            with tf.variable_scope(name,*args,**kwargs):
                setattr(self,attribute,function(self))
        return getattr(self,attribute)
    return decorator
class TrainModel(object):
    """
    训练模型
    """
    def __init__(self,data,labels,emdedding_keep,rnn_keep):
        self.data = data
        self.labels =labels
        self.emdedding_keep = emdedding_keep    # embedding层dropout保留率
        self.rnn_keep = rnn_keep    # lstm层dropout保留率
        self.global_step    #步
        self.predict    #细胞
        self.loss   #损失
        self.optimize   #最优化

    @define_scope
    def cell(self):
        """
        rnn网络结构
        :return:
        """
        lstm_cell = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
                                          output_keep_prob = self.rnn_keep ) for
            _ in range(NUM_LAYERS)]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
        return cell
    @define_scope
    def predict(self):
        """
        定义前向传播
        :return:
        """
        #创建词嵌入矩阵权重
        embedding = tf.get_variable('embedding',shape=[config.VOCABULARY_SIZE,HIDDEN_SIZE])
        #创建softmax层参数
        if config.SHARE_END_WITH_SOFTMAX:
            softmax_weights = tf.transpose(embedding)
        else:
            softmax_weights = tf.get_variable('softmaweights',shape=[HIDDEN_SIZE,config.VOCABULARY_SIZE])
        softmax_bais = tf.get_variable('softmax_bais',shape=[config.VOCABULARY_SIZE])
        # 进行词嵌入
        emd = tf.nn.embedding_lookup(embedding,self.data)
        #dropout
        enb_dropout = tf.nn.dropout(emd,self.emdedding_keep)
        #计算循环神经网络的输出
        self.init_state = self.cell.zero_state(config.BATCH_SIZE,dtype=tf.float32)
        outputs,last_state = tf.nn.dynamic_rnn(self.cell,enb_dropout,scope='d_rnn',dtype=tf.float32,
                                               initial_state=self.init_state)
        outputs = tf.reshape(outputs,[-1,HIDDEN_SIZE])
        #计算logits
        logits = tf.matmul(outputs,softmax_weights) + softmax_bais
        return logits
    @define_scope
    def loss(self):
        """
        定义损失函数
        :return:
        """
        #定义交叉熵
        outputs_target = tf.reshape(self.labels,[-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict,labels=outputs_target,)
        #平均
        cost = tf.reduce_mean(loss)
        return cost
    @define_scope
    def global_step(self):
        """
        golbal_step
        :return:
        """
        golbal_step = tf.Variable(0,trainable=False)
        return golbal_step
    @define_scope
    def optimize(self):
        """
        定义反向传播过程
        :return:
        """
        #学习衰减率
        learn_rate = tf.train.exponential_decay(config.LEARN_RATE,self.global_step,config.LR_DECAY_STEP,
                                           config.LR_DECAY)
        #计算梯度防止梯度爆炸
        trainable_variables = tf.trainable_variables()
        grads,_ = tf.clip_by_global_norm(tf.gradients(self.loss,trainable_variables),config.MAX_GRADIENT)
        #创建优化器，进行反向传播
        optimizer = tf.train.AdamOptimizer(learn_rate)
        train_op = optimizer.apply_gradients(zip(grads,trainable_variables),self.global_step)
        return train_op
class EvalModel(object):
    """
    验证模型
    """
    def __init__(self,data,emb_keep,rnn_keep):
        self.data =data
        self.emb_keep = emb_keep    # embedding层dropout保留率
        self.rnn_keep = rnn_keep    #lstm层dropout保留率
        self.cell
        self.predict
        self.prob

    @define_scope
    def cell(self):
        """
        rnn网络结构
        :return:
        """
        lstm_cell = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),output_keep_prob=self.rnn_keep) for
                     _ in range(NUM_LAYERS)]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
        return cell

    @define_scope
    def predict(self):
        """
        定义前向传播过程
        :return:
        """
        embedding = tf.get_variable('embedding',shape=[config.VOCABULARY_SIZE,HIDDEN_SIZE])
        if config.SHARE_END_WITH_SOFTMAX:
            softmax_weights = tf.transpose(embedding)
        else:
            softmax_weights = tf.get_variable('softmaweights',shape=[HIDDEN_SIZE,config.VOCABULARY_SIZE])
        softmax_bais = tf.get_variable('softmax_bais',shape=[config.VOCABULARY_SIZE])
        emb = tf.nn.embedding_lookup(embedding,self.data)
        emb_droput = tf.nn.dropout(emb,self.emb_keep)
        # 与训练模型不同，这里只要生成一首古体诗，所以batch_size=1
        self.init_state = self.cell.zero_state(1, dtype=tf.float32)
        outputs,last_start = tf.nn.dynamic_rnn(self.cell, emb_droput, scope='d_rnn', dtype=tf.float32,
                                               initial_state=self.init_state)
        outputs = tf.reshape(outputs,[-1,HIDDEN_SIZE])

        logits = tf.matmul(outputs,softmax_weights) + softmax_bais
        # 与训练模型不同，这里要记录最后的状态，以此来循环生成字，直到完成一首诗
        self.last_state = last_start
        return logits
    @define_scope
    def prob(self):
        """
        softmax计算概率
        :return:
        """
        probs = tf.nn.softmax(self.predict)
        return probs
