import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


'''声明一个 DataSet 类'''
class MyDataset(Dataset):

    def __init__(self, data) -> None:
        # data：使用词表映射之后的数据
        self.data = data

    def __len__(self):
        # 返回样例的数目
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
'''声明一个collate_fn函数，用于对一个批次的样本进行整理'''
def collate_fn(examples):
    # 从独立样本集合中构建各批次的输入输出
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    # 获取每个序列的长度
    inputs = [torch.tensor(ex[0]) for ex in examples]
    # 将输入inputs定义为一个张量的列表，每一个张量为句子对应的索引值序列
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # 目标targets为该批次所有样例输出结果构成的张量
    inputs = pad_sequence(inputs, batch_first=True)
    # 将用pad_sequence对批次类的样本进行补齐
    return inputs, lengths, targets

