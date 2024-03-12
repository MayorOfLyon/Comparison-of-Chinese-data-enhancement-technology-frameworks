import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
from torch.optim.lr_scheduler import StepLR

from oldmodel import RNNModel
from tokenizer import build_data_transformer
from dataset import MyDataset, collate_fn

disable = True

def train_one(*args):
    dir, name, batch_size = args
    train_data, dev_data, test_data, vocab = build_data_transformer(dir)
    train_dataset = MyDataset(train_data)
    dev_dataset = MyDataset(dev_data)
    # test_dataset = MyDataset(test_data)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dev_data_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    # test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = 'cuda'
    num_epoch = 10
    model = RNNModel(embedding_dim=256, hidden_dim=128, num_class=2).to(device)
    nll_loss = nn.NLLLoss()
    # 负对数似然损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Adam优化器

    model.train()
    loss_before = 1000
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}", disable=disable):
            inputs, lengths, targets = [x.to(device) for x in batch]
            # print(inputs.size())
            log_probs = model(inputs, lengths)
            loss = nll_loss(log_probs, targets)
            # print(log_probs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if loss_before > total_loss:
            with open(f'./model/{name}.pkl', 'wb') as f:
                torch.save(model.state_dict(), f)
            print('save!', end='\r')
        loss_before = total_loss
        
        acc = 0
        for batch in tqdm(dev_data_loader, desc=f"Validating Epoch {epoch}", disable=disable):
            inputs, lengths, targets = [x.to(device) for x in batch]
            with torch.no_grad():
                output = model(inputs, lengths).argmax(dim=1)
                acc += (output == targets).sum().item()
        print(f"Val acc:{acc / len(dev_dataset):.4f} Validating Epoch {epoch} "\
            f"Loss:{total_loss:.2f}", 
              end='\r' if disable else '\n')

    print(f'{dir} is Done!')
    # 测试
    # acc = 0
    # for batch in tqdm(test_data_loader, desc=f"Testing"):
    #     inputs, lengths, targets = [x.to(device) for x in batch]
    #     with torch.no_grad():
    #         output = model(inputs, lengths).argmax(dim=1)
    #         print(output, targets)
    #         acc += (output == targets).sum().item()
    # print(f"ACC:{acc / len(test_dataset):.4f}")


def test_one(*args):
    dir, name, _ = args
    train_data, dev_data, test_data, vocab = build_data_transformer(dir)
    # train_dataset = MyDataset(train_data)
    # dev_dataset = MyDataset(dev_data)
    test_dataset = MyDataset(test_data)

    batch_size = 16
    # train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # dev_data_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = 'cuda'
    model_dict = torch.load(f'./model/{name}.pkl')
    model = RNNModel(embedding_dim=256, hidden_dim=128, num_class=2).to(device)
    model.load_state_dict(model_dict)

    model.eval()
    # 测试
    acc = 0
    for batch in tqdm(test_data_loader, desc=f"Testing", disable=disable):
        inputs, lengths, targets = [x.to(device) for x in batch]
        with torch.no_grad():
            output = model(inputs, lengths).argmax(dim=1)
            # print(output, targets)
            acc += (output == targets).sum().item()
    print(f"ACC:{acc / len(test_dataset):.4f}", end='\r' if disable else '\n')
    return acc / len(test_dataset)


if __name__ == '__main__':
    # args = ('./data/raw/raw_shopping/raw_shopping_2000', 'raw_shopping_2000', 2000, False, 32)
    args = ('./data/raw/raw_shopping/raw_shopping_2000', 'raw_shopping_2000', 32)
    print(args)
    train_one(*args)
    # test_one(*args)