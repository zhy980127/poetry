#词汇表大小
VOCABULARY_SIZE = 7539

# 是否在embedding层和softmax层之间共享参数
SHARE_END_WITH_SOFTMAX = True

#最大梯度防止梯度爆炸
MAX_GRADIENT = 5.0

#初始学习率
LEARN_RATE = 0.0005

#学习衰减率
LR_DECAY = 0.92

#衰减步数
LR_DECAY_STEP = 600

#batch大小
BATCH_SIZE = 16

#模型保存路径
CKPT_PATH = 'ckpt/model_ckpt'

#词表路径
VOCAB_PATH = 'data/poetry.vocab'

#embedding层dropout保留率
EMB_KEEP = 0.5

#lstm层dropout保留率
RNN_KEEP = 0.5