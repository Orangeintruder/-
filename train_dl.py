# train_dl.py
import os
import argparse
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

# 导入预处理类和模型构建函数
from utils.preprocessor import DomainPreprocessor
from models.cnn_model import build_cnn
from models.lstm_model import build_lstm
from models.cnn_lstm_model import build_cnn_lstm

# 创建检查点目录（如果不存在）
os.makedirs('models/checkpoints', exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn_lstm',
                        choices=['cnn', 'lstm', 'cnn_lstm'])
    parser.add_argument('--level', type=str, default='char',
                        choices=['char', 'word'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--lstm_units', type=int, default=64)
    args = parser.parse_args()

    # 加载数据
    train_df = pd.read_csv('data/processed/train.csv')
    val_df   = pd.read_csv('data/processed/val.csv')
    test_df  = pd.read_csv('data/processed/test.csv')

    print(f"Train samples: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 预处理：根据 level 设置序列长度
    if args.level == 'char':
        max_len = 75
    else:  # word
        max_len = 20

    prep = DomainPreprocessor(max_len=max_len, level=args.level)
    # 用所有数据构建词汇表（确保词表覆盖完整）
    all_domains = pd.concat([train_df['domain'], val_df['domain'], test_df['domain']])
    vocab_size = prep.build_vocab(all_domains)
    print(f"Vocabulary size: {vocab_size}")

    # 编码数据集
    X_train = prep.encode(train_df['domain'])
    y_train = train_df['label'].values
    X_val   = prep.encode(val_df['domain'])
    y_val   = val_df['label'].values
    X_test  = prep.encode(test_df['domain'])
    y_test  = test_df['label'].values

    # 选择模型并构建
    model_dict = {
        'cnn':      build_cnn(max_len=max_len, vocab_size=vocab_size,
                              embedding_dim=args.embedding_dim),
        'lstm':     build_lstm(max_len=max_len, vocab_size=vocab_size,
                               embedding_dim=args.embedding_dim,
                               lstm_units=args.lstm_units),
        'cnn_lstm': build_cnn_lstm(max_len=max_len, vocab_size=vocab_size,
                                   embedding_dim=args.embedding_dim,
                                   lstm_units=args.lstm_units)
    }
    model = model_dict[args.model]
    model.summary()

    # 训练模型
    history = model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # 在测试集上评估
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")

    # 保存模型和预处理工具
    model_path = f'models/checkpoints/{args.model}_{args.level}.h5'
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # 保存 tokenizer（用于 predict 时对输入域名做相同编码）
    tokenizer_path = f'models/checkpoints/tokenizer_{args.level}.pkl'
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(prep.tokenizer, f)
    print(f"Tokenizer saved to {tokenizer_path}")

    # 保存训练历史（方便画图）
    history_path = f'models/checkpoints/{args.model}_{args.level}_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved to {history_path}")

if __name__ == '__main__':
    main()