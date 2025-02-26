import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

time_1 = time.time()
hf_token = "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"  # Replace with your token

# Load the model in 4-bit
# Or "microsoft/Phi-3-mini-4k-instruct"
model_name = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    token=hf_token  # Pass token to access the model
)

# model.to("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

# Create a pipeline for text generation
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to generate answers based on a file


def ask_file(file_path, question, max_new_tokens=100):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Use a more structured prompt
    prompt = (
        f"Context:\n{text}\n\n"
        f"Based on the context above, answer the following question concisely:\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    response = qa_pipeline(
        prompt, max_new_tokens=max_new_tokens, do_sample=True)

    return response[0]["generated_text"].strip()


# Example usage
file_path = "example.txt"  # Change this to your file
question = "What is the main topic?"
answer = ask_file(file_path, question)
print(answer)
print(f"Time taken: {time.time() - time_1:.2f} seconds")
