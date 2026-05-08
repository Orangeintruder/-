# 基于深度学习的恶意域名检测系统

## 项目简介

本项目为毕业设计——**"基于深度学习的恶意域名检测"** 的完整代码实现。 
通过构建正常域名与 DGA（域名生成算法）恶意域名数据集，训练并对比字符级/单词级的 CNN、LSTM 及 CNN‑LSTM 混合模型，实现端到端的域名自动检测与风险评估。

系统支持：
- 多类型 DGA 恶意域名自动生成（Banjori、Ramnit、Murofet）
- 字符级与单词级两种输入粒度的预处理
- CNN、LSTM、CNN‑LSTM 三种深度学习模型训练
- 基于手工特征的传统机器学习基线（随机森林、SVM）
- 全面的模型评估（准确率、精确率、召回率、F1、ROC 曲线、混淆矩阵）
- 命令行交互检测 + Streamlit 网页演示

---

## 目录结构

```
Malicious_Domain_Detection/
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据（合法域名 & DGA 域名）
│   └── processed/             # 预处理后的训练/验证/测试集 CSV
├── utils/                     # 工具模块
│   ├── __init__.py
│   ├── data_loader.py         # 数据加载与划分
│   ├── preprocessor.py         # 字符级/单词级编码与 padding
│   ├── feature_engineer.py     # 传统 ML 手工特征提取
│   └── dga_generator.py        # DGA 域名生成算法
├── models/                    # 模型定义与保存
│   ├── __init__.py
│   ├── cnn_model.py           # 纯 CNN 模型
│   ├── lstm_model.py          # 纯 LSTM 模型
│   ├── cnn_lstm_model.py      # CNN‑LSTM 混合模型
│   ├── traditional_ml.py      # 传统 ML 模型训练（可选）
│   └── checkpoints/           # 保存的模型文件 (.h5, .pkl)
├── train_dl.py                # 深度学习模型训练主脚本
├── train_ml.py                # 传统机器学习基线训练脚本
├── evaluate.py                # 统一评估与可视化
├── predict.py                 # 命令行交互检测程序
├── app.py                     # Streamlit 网页演示
├── requirements.txt           # 依赖包列表
└── README.md                  # 项目说明
```

---

## 环境配置

### 1. 推荐 Python 版本
- **Python 3.10 或 3.11**（TensorFlow 稳定支持）
- 避免使用 Python 3.12 以上版本，可能不兼容。

### 2. 安装依赖
在项目根目录下：
```bash
python -m venv venv
# Windows 激活
venv\Scripts\activate
# Linux/macOS 激活
source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` 内容：
```
numpy
pandas
scikit-learn
matplotlib
seaborn
tensorflow
tldextract
streamlit
joblib
```

---

## 使用指南

### 第一步：准备原始域名数据

1. **合法域名采集**  
   - 从 [Alexa Top 1M](http://s3.amazonaws.com/alexa-static/top-1m.csv.zip) 或 Cisco Umbrella 下载
   - 提取域名，每行一个，保存为 `data/raw/legitimate_domains.txt`

2. **生成 DGA 恶意域名**  
   运行：
   ```bash
   python utils/dga_generator.py
   ```
   该脚本会基于 Banjori、Ramnit、Murofet 三种算法各生成 10000 个恶意域名，合并保存至 `data/raw/dga_domains.txt`。  
   *可以根据需要修改生成数量或添加更多算法。*

### 第二步：构建训练/验证/测试集
```bash
python utils/data_loader.py
```
脚本会读取 `data/raw/` 下的两个文件，打标签（0=正常，1=恶意），按 80/10/10 比例划分并保存为 `data/processed/train.csv`、`val.csv`、`test.csv`。

### 第三步：训练深度学习模型

**格式**：`python train_dl.py --model <模型名> --level <输入级别> [--epochs N] [--batch_size B]`

- **模型名**：`cnn`, `lstm`, `cnn_lstm`
- **输入级别**：`char`（字符级，长度75）或 `word`（单词级，长度20）

**示例训练命令**（按毕设要求必须覆盖字符级与单词级）：
```bash
# 字符级 CNN-LSTM（核心混合模型）
python train_dl.py --model cnn_lstm --level char --epochs 20

# 字符级基准模型
python train_dl.py --model cnn --level char --epochs 20
python train_dl.py --model lstm --level char --epochs 20

# 单词级模型（体现单词级研究点）
python train_dl.py --model lstm --level word --epochs 20
# 可选其他单词级模型
python train_dl.py --model cnn --level word --epochs 20
python train_dl.py --model cnn_lstm --level word --epochs 20
```

模型、分词器（tokenizer）及训练历史均自动保存在 `models/checkpoints/` 下。

### 第四步：训练传统机器学习基线
```bash
python train_ml.py
```
利用手工特征提取（长度、熵、元音/数字比例等），训练 **随机森林** 和 **SVM**，保存模型并输出性能指标。

### 第五步：综合评估与可视化
```bash
python evaluate.py
```
该脚本会：
- 加载所有已训练的深度模型和传统模型
- 在测试集上计算 Accuracy、Precision、Recall、F1
- 生成对比图表，保存至 `figures/` 文件夹：
  - `metrics_bar.png` – 准确率/F1 柱状图
  - `roc_curves.png` – 多模型 ROC 曲线
  - `confusion_matrix.png` – 最佳模型的混淆矩阵  
  - `all_models_results.csv` – 完整指标表格

### 第六步：交互式检测演示

#### 命令行版本
```bash
python predict.py
```
输入域名，实时输出是否恶意及概率，按 `q` 退出。

#### 网页演示（推荐答辩使用）
```bash
streamlit run app.py
```
浏览器自动打开 `http://localhost:8501`，提供简洁的 UI，支持单域名检测与批量文件上传分析。

---

## 技术实现要点

- **字符级编码**：每个字符（含标点）转为独立 token，序列定长 75，适合捕捉精细拼写模式。
- **单词级编码**：按 `.` 和 `-` 拆分成多级标签，序列定长 20，适合提取域名结构信息。
- **CNN 模型**：单层一维卷积 + 全局最大池化，提取局部 n-gram 特征。
- **LSTM 模型**：双向 LSTM，捕捉域名整体序列依赖。
- **CNN‑LSTM 混合模型**：多尺度卷积 (3,4,5) 与 LSTM 并行结合，同时捕获局部突变与全局依赖，为本课题核心创新架构。
- **传统基线**：随机森林/SVM 使用手工统计特征，用于验证深度学习方法优势。

---

## 常见问题

**Q: 训练时提示 `tensorflow.python` 找不到？**  
A: 请使用 Python 3.10/3.11，删除旧虚拟环境后重新创建并安装依赖。

**Q: 评估时某些模型报文件不存在？**  
A: 确保已运行对应的训练命令。`evaluate.py` 会跳过缺失模型，先训练再评估即可。

**Q: `dga_generator.py` 路径错误？**  
A: 请确认脚本在当前工作目录下运行，或调整文件保存路径。建议在项目根目录执行：  
```bash
cd Malicious_Domain_Detection
python utils/dga_generator.py
```

**Q: 如何打包为 EXE？**  
A: 使用 PyInstaller（需自行处理依赖路径）：
```bash
pip install pyinstaller
pyinstaller --onefile --add-data "models/checkpoints/cnn_lstm_char.h5;." --add-data "utils;utils" predict.py
```

---

## 作者与声明

本项目为毕业设计示例，仅供学习研究使用。数据集来源为公开的 Alexa/DGA 样本，模型代码由 TensorFlow/Keras 构建。