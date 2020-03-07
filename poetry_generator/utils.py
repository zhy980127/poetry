import config

def read_word_list():
    """
    从文件读取词汇表
    :return: 词汇列表
    """
    with open(config.VOCAB_PATH,'r') as f:
        word_list = [word for word in f.read().strip().split('\n')]
    return word_list

def read_word_to_id_dict():
    """
    生成单词到id的映射
    :return:
    """
    word_list = read_word_list()
    word_to_id = dict(zip(word_list, range(len(word_list))))
    return word_to_id

def read_id_to_word_dict():
    """
    生成id到单词的映射
    :return:
    """
    word_list = read_word_list()
    id_to_word = dict(zip(range(len(word_list)), word_list))
    return id_to_word

if __name__ == '__main__':
    read_id_to_word_dict()