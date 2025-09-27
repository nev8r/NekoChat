
# NekoChat

Base Model: internlm_chat_7B


Dataset : NekoQA-10K 🐱


sft tools: Xtuner(Qlora)
## Cmp
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# model_path = "./work_dirs/merged"
model_path = "./internlm2_5-7b-chat"
print(f"加载模型：{model_path}")

start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype="auto", device_map="auto"
)

def cmp(input):
    messages = [
        {"role": "system", "content": "你是一只哈基米（猫娘)，性格粘人，爱撒娇，喜欢喵喵叫"},
        {"role": "user", "content": input},
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,  
        do_sample=True,
        temperature=0.1,  # 降低温度提高确定性
        top_p=0.95,
        repetition_penalty=1.0,
    )


    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()
    
    return response


if __name__ == "__main__":
    input = "我现在有点生气！"
    result = cmp(input)
    print(result)


    end_time = time.time()
    total_time = end_time - start_time
    print(f"程序总耗时：{total_time:.2f}秒")
```

### Base Model

![base](./img/base.png)

### Sft Model
![sft](./img/sft.png)



