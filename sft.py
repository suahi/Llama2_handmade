import argparse
import torch
from Llama import Transformer as Model
from transformers import AutoTokenizer
import DataLoader


def pretrain():
    pass


if __name__ == '__main__':
    # args define
    parser = argparse.ArgumentParser(description="Tiny-LLM Pretraining")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--accumulate_grad", type=int, default=8, help="累积更新梯度")

    args = parser.parse_args()
    # model&tokenizer define
    llama = Model()
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer')

    # optimizer define
    scaler = scaler.scale(loss)
    optimizer = torch.optim.Adam(lr=args.lr)

    # data define
    train_dataset =  Dataset(dataLoader.PretrainData)

    # start to train
    for epoch in range(args.epochs):
        for i, (x, y) in enumerate(train_dataset):
            pred = Model(x)
            loss = scaler(pred, y)

            if (i % args.accumulate_grad == 0):
                # 梯度回传
                optimizer.step()
                # 梯度清空
                optimizer.zero_grad()







