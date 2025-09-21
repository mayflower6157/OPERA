from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/opt-350m"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

inputs = tok("Hello OPERA", return_tensors="pt")

out = model.generate(
    **inputs,
    max_length=30,
    opera_decoding=True,
    num_beams=2,
    do_sample=True
)
print(tok.decode(out[0]))
