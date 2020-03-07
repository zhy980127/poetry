from collections import Counter
INPUT_DATA = 'data/poetrys.txt'     #原始数据
OUTPUT_DATA = 'data/output_poetry.txt'  #输出向量
VOCAB_DATA = 'data/poetry.vocab'

def word_to_id(word,id_dict):
    if word in id_dict:
        return id_dict[word]
    else:
        return id_dict['<unknow>']

#唐诗数组
poetry_list = []

#读取唐诗源文件
with open(INPUT_DATA, 'r') as f:
    f_lines = f.readlines()
    print('唐诗总行数:',len(f_lines))
    #去除空行，空格
    for line in f_lines:
        strip_line = line.strip()
        if len(strip_line) <5 or len(strip_line) >70:
            #去除长度过大或过小的唐诗
            continue
        if strip_line.split():
            # 去除多余空格
            poetry_content = strip_line.strip().replace(' ', '')
            #加入列表
            poetry_list.append('s' + poetry_content + 'e')
        else:
            continue
print('用于训练的诗总数：',len(poetry_list))
#整理数组排序从小到大，sorted排序（可迭代类型都可以不只是list）
poetry_list = sorted(poetry_list,key=lambda x:len(x))

#字符列表
words_list = []
#获取唐诗内全部字符
for poetry in poetry_list:
    #extend追加
    words_list.extend([word for word in poetry])
#统计出现次数
counter = Counter(words_list)
#排序
sorted_words = sorted(counter.items(),key=lambda x:x[1],reverse=True)
#获得降序排列数组字符串列表
words_list = ['<unknow>']+[x[0] for x in sorted_words]
words_list = words_list[:len(words_list)]
print("词汇表大小:",len(words_list))
print('词汇表：',words_list)

#写入数据
with open(VOCAB_DATA,'w') as f:
    for word in words_list:
        f.write(word+'\n')

#生成词到id的映射
word_id_dict = dict(zip(words_list,range(len(words_list))))
#将poetry_list转换成向量
id_list = []
for poetry in poetry_list:
    id_list.append([str(word_to_id(word,word_id_dict))for word in poetry])

#将向量写入文件
with open(OUTPUT_DATA,'w') as f:
    for ids in id_list:
        f.write(' '.join(ids)+'\n')

