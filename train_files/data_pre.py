import json

# 输入JSON文件路径
input_json_file = "data/Neko/NekoQA-10K.json"
# 输出JSONL文件路径
output_jsonl_file = "data/Neko/NekoQA-10K.jsonl"

try:
    # 第一步：读取JSON文件
    with open(input_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 确保数据是列表形式
    if not isinstance(data, list):
        data = [data]
    
    # 第二步：处理并写入JSONL文件
    with open(output_jsonl_file, 'w', encoding='utf-8') as f_out:
        for item in data:
            # 提取原始字段
            original_instruction = item.get("instruction", "")
            original_output = item.get("output", "")
            
            # 构造新的JSON对象
            new_data = {
                "instruction": "",
                "input": original_instruction,
                "output": original_output + "</s>"
            }
            
            # 写入文件
            json.dump(new_data, f_out, ensure_ascii=False)
            f_out.write('\n')
    
    print(f"处理完成！已生成 {output_jsonl_file}")

except FileNotFoundError:
    print(f"错误：找不到文件 {input_json_file}")
except json.JSONDecodeError:
    print(f"错误：{input_json_file} 不是有效的JSON文件")
except Exception as e:
    print(f"处理出错：{str(e)}")
