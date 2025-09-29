from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ModelWrapper:
    def __init__(self, model_path, system_prompt=""):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.system_prompt = system_prompt

    def chat(self, history, max_new_tokens=512, temperature=0.8, top_p=0.9):
        """
        history: [{"role": "user"/"assistant", "content": "..."}]
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(history)

        # 关键：用 tokenizer.apply_chat_template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response
