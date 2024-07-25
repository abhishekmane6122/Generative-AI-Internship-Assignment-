from transformers import GPT2LMHeadModel, GPT2Tokenizer
from vector_database import setup_faiss, load_texts, search

def generate_response(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=500, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def chatbot_response(query):
    texts = load_texts()
    index, model, texts = setup_faiss(texts)
    retrieved_texts = search(index, model, texts, query, k=1)
    context = " ".join(retrieved_texts)
    prompt = f"Context: {context}\n\nQ: {query}\nA:"
    return generate_response(prompt)

# Example usage
if __name__ == "__main__":
    response = chatbot_response("What is Apple Vision Pro?")
    print(response)