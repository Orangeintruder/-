# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from utils.preprocessor import DomainPreprocessor

# ---- 页面配置 ----
st.set_page_config(page_title="恶意域名检测系统", page_icon="🛡️", layout="centered")

# ---- 缓存加载模型 ----
@st.cache_resource
def load_assets(model_path, tokenizer_path, level, max_len):
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    prep = DomainPreprocessor(max_len=max_len, level=level)
    prep.tokenizer = tokenizer
    prep.vocab_size = len(tokenizer.word_index) + 1
    return model, prep

# 配置模型路径（按你训练好的最佳模型修改）
MODEL_PATH = 'models/checkpoints/cnn_lstm_char.h5'
TOKENIZER_PATH = 'models/checkpoints/tokenizer_char.pkl'
LEVEL = 'char'
MAX_LEN = 75

model, prep = load_assets(MODEL_PATH, TOKENIZER_PATH, LEVEL, MAX_LEN)

# ---- UI ----
st.title("🛡️ 基于深度学习的恶意域名检测系统")
st.markdown("输入一个域名，系统将自动判断它是**正常域名**还是**恶意域名**（如DGA生成）。")

domain = st.text_input("请输入待检测域名:", placeholder="例如: www.google.com 或 xyzmalware.net")

if st.button("开始检测", type="primary"):
    if not domain.strip():
        st.warning("请输入有效域名")
    else:
        with st.spinner("正在分析..."):
            encoded = prep.encode([domain.strip()])
            prob = model.predict(encoded, verbose=0)[0][0]
            is_malicious = prob > 0.5

        # 结果展示
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            if is_malicious:
                st.error("⚠️ 恶意域名")
            else:
                st.success("✅ 正常域名")
        with col2:
            st.metric("恶意概率", f"{prob:.2%}")
            if is_malicious:
                st.caption("该域名很可能是由DGA算法生成，或具备恶意特征。")
            else:
                st.caption("该域名来自常见顶级域，结构正常。")

        # 显示详细信息
        with st.expander("查看详细分析"):
            st.write(f"**原始域名:** {domain.strip()}")
            st.write(f"**处理级别:** {LEVEL}")
            st.write(f"**序列长度:** {MAX_LEN}")
            st.write(f"**模型:** {MODEL_PATH}")
            # 可视化概率
            st.progress(float(prob))

# 侧边栏：批量检测
st.sidebar.header("批量检测 (可选)")
uploaded_file = st.sidebar.file_uploader("上传文本文件 (每行一个域名)", type=["txt", "csv"])
if uploaded_file is not None:
    try:
        domains = uploaded_file.read().decode('utf-8').splitlines()
        # 如果是CSV，取第一列
        if uploaded_file.name.endswith('.csv'):
            import io
            df = pd.read_csv(io.BytesIO(uploaded_file.read()), header=None)
            domains = df[0].tolist()
        st.sidebar.write(f"共 {len(domains)} 个域名")
        if st.sidebar.button("批量分析"):
            results = []
            for d in domains:
                if d.strip():
                    encoded = prep.encode([d.strip()])
                    prob = model.predict(encoded, verbose=0)[0][0]
                    results.append((d.strip(), prob, '恶意' if prob>0.5 else '正常'))
            df_res = pd.DataFrame(results, columns=['域名', '恶意概率', '判定结果'])
            st.sidebar.dataframe(df_res)
    except Exception as e:
        st.sidebar.error(f"解析文件出错: {e}")

# 页脚
st.markdown("---")
st.caption("毕业设计项目 · 深度学习恶意域名检测 · 杨金粱")
#运行命令：streamlit run app.py
#恶意域名：apiw2xwwiv.org、nawaciqepucov.net、api5hh7.org