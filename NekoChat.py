import streamlit as st
import json
import time
import requests
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from datetime import datetime

# 页面配置
st.set_page_config(
    page_title="NekoChat - 虚拟猫娘助手",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded"
)

