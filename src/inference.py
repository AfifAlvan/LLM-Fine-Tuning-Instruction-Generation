import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "models/final_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)

def generate_steps(instruction):
    prompt = f"Generate step-by-step instructions for: {instruction}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.8,
        num_beams=1,
        no_repeat_ngram_size=2,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    test_instruction = " How  do  I  reset  my  password  in  the  e-commerce  “xx”  mobile app ?"
    print(generate_steps(test_instruction))
