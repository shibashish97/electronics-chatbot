from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-chat-hf"

# Download and save the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save the model and tokenizer locally
tokenizer.save_pretrained("./local_llama2_model")
model.save_pretrained("./local_llama2_model")

print("Model and tokenizer saved locally!")
