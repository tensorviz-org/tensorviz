"""LoRA supervised finetune of a pretrained checkpoint.

    python finetune_lora.py

Needs `transformers`, `datasets`, `peft` and `trl`, which the baseline does not —
install them only if you want to run this one. The graph reads it either way,
because reading is a parse rather than an import.

Nothing here subclasses anything: the pipeline is assembled by instantiating
library classes, which is how most current research code is written.
"""
import tensorviz
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

tensorviz.experiment("lora-sft")

BASE_MODEL = "sshleifer/tiny-gpt2"


def build_peft_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )


def main():
    raw = load_dataset("imdb", split="train[:1%]")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    peft_config = build_peft_config()

    args = TrainingArguments(
        output_dir="runs/lora-sft",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=0.0002,
        logging_steps=10,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=raw,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model("runs/lora-sft")


if __name__ == "__main__":
    main()
