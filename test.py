import requests

def chat_with_gemma(prompt):
    response = requests.post(
        "http://localhost:11434/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "gemma3:latest",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    return response.json()["choices"][0]["message"]["content"]

# Test it
print(chat_with_gemma("Tell me a fun fact about Bangalore."))
