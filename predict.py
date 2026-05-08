# predict.py
import sys
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocessor import DomainPreprocessor

# ============ 配置：根据你实际训练的模型选择 ============
MODEL_PATH = 'models/checkpoints/cnn_lstm_char.h5'   # 推荐使用表现最好的模型
TOKENIZER_PATH = 'models/checkpoints/tokenizer_char.pkl'
LEVEL = 'char'    # 'char' 或 'word'
MAX_LEN = 75      # 字符级 75, 单词级 20

# 加载模型和预处理器
print("正在加载模型...")
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
prep = DomainPreprocessor(max_len=MAX_LEN, level=LEVEL)
prep.tokenizer = tokenizer
prep.vocab_size = len(tokenizer.word_index) + 1
print("模型就绪，输入域名进行检测 (q 退出)\n")

# 交互循环
while True:
    domain = input("请输入域名: ").strip()
    if domain.lower() in ('q', 'quit', 'exit'):
        break
    if not domain:
        continue

    # 编码并预测
    encoded = prep.encode([domain])
    prob = model.predict(encoded, verbose=0)[0][0]
    pred_label = '恶意' if prob > 0.5 else '正常'
    print(f"  预测结果: {pred_label}")
    print(f"  恶意概率: {prob:.4f}\n")