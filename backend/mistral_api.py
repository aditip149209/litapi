from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hf-inference",
    api_key="",
)

completion = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    max_tokens=500,
)

print(completion.choices[0].message)