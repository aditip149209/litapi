import random
import torch 

from model_def import BigramLanguageModel
from model_def import stoi
from model_def import decode
from model_def import device

model = BigramLanguageModel()

import os
model_path = os.path.abspath("../models/model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))


model.to(device)

model.eval()

def generate_response(prompt):
    encoded_prompt = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
    num = random.randint(100, 1000)
    output = model.generate(encoded_prompt, max_new_tokens=num)
    generated_text = decode(output[0].tolist())
    return generated_text

