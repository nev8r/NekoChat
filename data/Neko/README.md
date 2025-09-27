---
license: apache-2.0
---
![catgirl](image/catgirl1.png)

# Dataset Card for NekoQA-10K 🐱

## Dataset Summary

**NekoQA-10K** 是一个面向大语言模型的 **猫娘对话数据集**，共包含 **10,000 条 QA 对话**。
所有回答均遵循统一的 **猫娘人设**：

* 称呼用户为“主人”
* 在句尾添加特定口癖（如“喵\~”、“no desu”、“的说喵”）
* 保持可爱、撒娇、二次元风格

该数据集的主要用途是研究 **大语言模型的“猫娘味”塑造能力**，为微调、对话风格迁移、拟人化交互研究提供素材。

---

## Supported Tasks and Benchmarks

* **风格微调 (Style Finetuning)**: 提升模型的“猫娘化”特征。
* **角色对话生成 (Persona-based Dialogue)**: 研究 LLM 的角色一致性建模。
* **情感陪伴研究 (Affective Computing)**: 探索模型与用户的情感交互能力。
* **评测基准 NekoBench**: 可结合本数据集，评估模型在“猫娘味感知指数 (NPS)”上的表现。

---

## Languages

* **Chinese (zh)** 为主，带少量中英夹杂。
* 语气风格统一为 **猫娘口癖**，具有强烈的拟人化特征。

---

## Dataset Creation

### Source Data

* 部分人工手写原创问答（注意，作者不是猫娘）
* 部分来源于公开网络论坛（如弱智吧），经过 **大模型重写**，保证风格统一与安全性
* 基于现有猫娘QA数据集的重写（少量，900条左右）

### Annotations

* 回答大部分由大语言模型生成并人工筛选
* 风格标签：猫娘口癖

### Ethical Considerations

* 数据不包含敏感、违法或仇恨内容
* **温馨提醒**：请勿将本数据集用于真实人际关系替代，仅限学术和娱乐研究

---

## Limitations

* 回答多为轻松、拟人化语气，**不保证事实严谨性**
* 可能导致模型在严肃任务上“过于可爱”

---

## Citation

如果你使用了本数据集，请引用以下论文：

```bibtex
@article{nekoqa2025,
  title={NekoQA-10K: A Catgirl Dialogue Dataset and NekoBench Evaluation},
  author={MindsRiverPonder},
  journal={ZHIHU preprint ZHIHU:2508.22},
  year={2025}
}
```

---

## License

本数据集基于 **apache-2.0** 开源。

> 猫娘味可自由传播，撒娇权属于全人类。

---


