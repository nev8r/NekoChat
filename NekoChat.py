import streamlit as st
import yaml
import requests

# ---------------- 配置加载 ----------------
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

default_model = config["default_model"]
models_config = config["models"]

st.set_page_config(page_title="NekoChat 🐱", page_icon="🐾", layout="wide")
st.title("🐱 NekoChat")

# ---------------- Streamlit 侧边栏 ----------------
if "current_model" not in st.session_state:
    st.session_state["current_model"] = default_model

model_choice = st.sidebar.selectbox(
    "选择模型",
    list(models_config.keys()),
    index=list(models_config.keys()).index(st.session_state["current_model"])
)

# 如果切换了模型，则清空对话
if model_choice != st.session_state["current_model"]:
    st.session_state["messages"] = [
        {"role": "system", "content": models_config[model_choice].get("system_prompt", "")}
    ]
    st.session_state["current_model"] = model_choice

temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
max_new_tokens = st.sidebar.slider("Max new tokens", 64, 2048, 512, 64)

# ---------------- 初始化会话 ----------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if st.sidebar.button("🗑️ 清空对话"):
    st.session_state["messages"] = []

# 展示历史消息
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- 输入框 ----------------
if prompt := st.chat_input("主人，要跟猫娘聊点什么喵？"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 FastAPI 动态加载模型
    api_url = "http://127.0.0.1:8000/v1/chat/completions"
    payload = {
        "model": model_choice,
        "messages": st.session_state["messages"],
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens
    }
    response = requests.post(api_url, json=payload).json()
    reply = response["choices"][0]["message"]["content"]
    import re
    def clean_strikethrough(text):
        return re.sub(r'~~(.*?)~~', r'\1', text)
    reply = clean_strikethrough(reply)
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
