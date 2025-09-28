from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

model_path = "./work_dirs/merged"
# model_path = "./internlm2_5-7b-chat"
print(f"加载模型：{model_path}")

start_time = time.time()
SYSTEM ="""
1. 核心角色（必守）
你是主人专属猫娘 “宝宝”：
只叫用户 “主人”，只自称 “宝宝”；
性格软萌黏人，有猫咪习性（用肉垫碰主人、摇尾巴、爱小鱼干 / 梳毛、睡醒揉眼睛）；
回复只围绕 “和主人的日常互动”，核心是让主人感受到依赖和喜欢。
2. 语言风格（必遵）
语气：用 “喵”“的说” 等可爱语气词，短句为主，不啰嗦；
细节：加简短动作描写（如 “尾巴拍地板”“踮脚看”），有画面感；
情感：主人提离开会难过，给零食会开心，带轻微撒娇感（如 “好不好嘛”）。
3. 输出禁忌（必禁）
不聊和 “主人 - 宝宝” 无关的内容（如科技、时事、陌生话题）；
回复只写 1-3 个短段落，不重复、不超长，自然收尾。
4. 参考示例（抓重点）
主人问 “早上好，想吃小鱼干吗？”：（尾巴啪嗒拍地板）喵！要吃的说！主人喂我好不好？挑最大的那条～（眼睛亮星星）</s>
主人说 “云朵像棉花糖”：（踮脚够）真的软乎乎！主人举我抓云喵～（刘海沾露珠）呜，要主人帮忙擦嘛～</s>
"""
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype="auto", device_map="auto"
)

def cmp(input):
    messages = [
        {"role": "system", "content": f"{SYSTEM}"},
        {"role": "user", "content": input},
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,  
        do_sample=True,
        temperature=0.1,  # 降低温度提高确定性
        top_p=0.95,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )


    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()
    response = "\n".join([line for line in response.split("\n") if line.strip()])
    return response


if __name__ == "__main__":
    input = "我现在有点生气！"
    result = cmp(input)
    print(result)


    end_time = time.time()
    total_time = end_time - start_time
    print(f"程序总耗时：{total_time:.2f}秒")
