import pandas as pd
from transformers import AutoTokenizer

def build_data_transformer(data_path):
    '''
    Args:
       data_path:待读取本地数据的路径 
    Returns:
       训练集、测试集、词表
    '''
    train_data = pd.read_csv(data_path + '/train.csv', names=['label', 'review'], 
                             sep=',', engine='python', on_bad_lines='skip')
    
    dev_data = pd.read_csv(data_path + '/dev.csv', names=['label', 'review'],
                           sep=',', engine='python', on_bad_lines='skip')
    
    test_data = pd.read_csv(data_path + '/test.csv', names=['label', 'review'],
                            sep=',', engine='python', on_bad_lines='skip')
    # print(train_data.head())
    # 读取数据为 DataFrame 类型
    tokenizer = AutoTokenizer.from_pretrained("./models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55", cache_dir="./", offline=True)
    # 构建词表
    # train_data = \
    # [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in whole_data[whole_data["label"] == 1][:cut1]["review"]] \
    # +[(vocab.convert_tokens_to_ids(sentence), 0) for sentence in whole_data[whole_data["label"] == 0][:cut0]["review"]]
    train_data = [(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(series['review'])), int(series['label'])) for _, series in train_data.iterrows()]
    # dev_data = \
    # [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in whole_data[whole_data["label"] == 1][cut1:]["review"]] \
    # +[(vocab.convert_tokens_to_ids(sentence), 0) for sentence in whole_data[whole_data["label"] == 0][cut0:]["review"]]
    dev_data = [(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(series['review'])), int(series['label'])) for _, series in dev_data.iterrows()]
    # test_data = \
    # [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in whole_data[whole_data["label"] == 1][cut1:]["review"]] \
    # +[(vocab.convert_tokens_to_ids(sentence), 0) for sentence in whole_data[whole_data["label"] == 0][cut0:]["review"]]
    test_data = [(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(series['review'])), int(series['label'])) for _, series in test_data.iterrows()]
    # 其余数据作为测试数据
    # print(tokenizer.vocab.__len__())
    return train_data, dev_data, test_data, tokenizer.vocab
