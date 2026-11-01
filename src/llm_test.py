import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": "Explain machine learning in one simple sentence."
        }
    ]
)

print("AI Answer:")
print(response["message"]["content"])