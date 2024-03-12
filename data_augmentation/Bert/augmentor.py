import argparse
import util
import bert_main as bert
import warnings
warnings.filterwarnings("ignore")

class Augmentor(object):
    def __init__(self, model_dir:str):
        #Bert
        self.mask_model = bert.BertAugmentor(model_dir)
        pass
    
    def bert_replace_augment(self, input_file:str, output_file:str):
        writer = open(output_file, 'w', encoding='utf-8')
        lines = open(input_file, 'r', encoding='utf-8').readlines()
        print("正在使用Bert_replace增强语句...")

        for i, line in enumerate(lines):
            #分割标签与文本
            parts = line[:-1].split('\t')    #使用[:-1]是把\n去掉了
            label = parts[0]
            if len(parts) > 1:
                sentence = parts[1]
            else:
                print("Warning: The line does not have enough elements.")
            sentence = parts[1]

            #得到Bert进行替换词后的结果
            replace_result = self.mask_model.replace_word2queries(sentence)
            #取出得分排名前五的结果
            top_results = sorted(replace_result, key=lambda x: x['score'], reverse=True)[:5]
            for result in top_results:
                augment_sentence = str(result['sequence']).replace(" ", "")
                writer.write(label + "\t" + augment_sentence + '\n')
        writer.close()
        print("已生成增强语句!")


    def bert_insert_augment(self, input_file:str, output_file:str):
        writer = open(output_file, 'a', encoding='utf-8')
        lines = open(input_file, 'r', encoding='utf-8').readlines()
        print("正在使用Bert_insert增强语句...")
   
        for i, line in enumerate(lines):
            #分割标签与文本
            parts = line[:-1].split('\t')    #使用[:-1]是把\n去掉了
            label = parts[0]
            if len(parts) > 1:
                sentence = parts[1]
            else:
                print("i is ",i)
                print("Warning: The line does not have enough elements.")
            sentence = parts[1]

            #得到Bert进行插入词后的结果
            replace_result = self.mask_model.insert_word2queries(sentence)
            #取出得分排名前五的结果
            top_results = sorted(replace_result, key=lambda x: x['score'], reverse=True)[:5]
            for result in top_results:
                augment_sentence = str(result['sequence']).replace(" ", "")
                writer.write(label + "\t" + augment_sentence + '\n')

        writer.close()
        print("已生成增强语句!")

    def augment(self, input_file, output_file):
        # bert replace
        self.bert_replace_augment(input_file, output_file)
        # bert insert
        self.bert_insert_augment(input_file, output_file)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./data/raw_data/raw_500.txt", type=str, help="input file of unaugmented data")
    ap.add_argument("--output", default="./data/bert_replace/bert_500.txt", type=str, help="output file of augmented data")
    ap.add_argument("--bert_dir", default="bert-base-chinese", type=str, help="input file of unaugmented data")
    args = ap.parse_args()
    augmentor = Augmentor(args.bert_dir)
    # Bert数据增强
    augmentor.augment(args.input, args.output)

