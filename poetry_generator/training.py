import tensorflow as tf
import config
import dataset
from rnn_model import TrainModel
# g1 = tf.Graph()
ITERATION_TIMES = 30000 #迭代总次数
SHOW_STEP = 1   #显示loss频率
SAVE_STEP = 100 #保存模型参数频率
#placeholder占位符
x_data = tf.placeholder(tf.int32,[config.BATCH_SIZE,None])  #输入数据
y_data = tf.placeholder(tf.int32,[config.BATCH_SIZE,None])  #标签
embedding_keep = tf.placeholder(tf.float32)     #embedding层dropout保留率
rnn_keep = tf.placeholder(tf.float32)       #lstm层dropout保留率

#创建数据集
data = dataset.Dataset(config.BATCH_SIZE)   #创建数据集

model = TrainModel(x_data,y_data,embedding_keep,rnn_keep)    # 创建训练模型

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())     #初始化op
    wither = tf.summary.FileWriter('./logs', sess.graph())
    for step in range(ITERATION_TIMES):
        #获取训练batch
        x,y = data.next_batch()
        #计算loss
        loss, _ = sess.run([model.loss,model.optimize],
                           {model.data:x,model.labels:y,model.emdedding_keep:config.EMB_KEEP,
                            model.rnn_keep:config.RNN_KEEP})
        if step % SHOW_STEP ==0:


            print('步长 {}, 损失 {}'.format(step, loss))

        #保存模型
        if step % SAVE_STEP == 0:
            saver.save(sess,config.CKPT_PATH,global_step=model.global_step)

