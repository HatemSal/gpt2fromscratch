from flask import Flask, request, jsonify
from src.gptmodel import GPTModel
import tiktoken
import torch
from pathlib import Path
BASE_CONFIG = {
    "vocab_size": 50257, 
    "context_length": 1024, 
    "drop_rate": 0.0, 
    "qkv_bias": True 
    }
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
model_choice = "gpt2-small (124M)"
BASE_CONFIG.update(model_configs[model_choice])
model = GPTModel(BASE_CONFIG)
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG['emb_dim'],
    out_features=num_classes
)
tokenizer = tiktoken.get_encoding("gpt2")
context_length = BASE_CONFIG['context_length']

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
models_path = Path("models")
model_name = "unfrozen_finaltrfblock_finalBN_classhead.pth"
model.load_state_dict(torch.load(models_path/model_name, weights_only=True))
model.to(device)
model.eval()

@app.route('/predict', methods = ['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text']
        encoded = tokenizer.encode(text)
        encoded = torch.tensor(encoded, dtype=torch.long).to(device)
        if len(encoded) > context_length:
            encoded = encoded[:-context_length]
        with torch.inference_mode():
            logits = model(encoded.unsqueeze(0))
            logits = logits[:,-1,:]
            predicted_label = torch.argmax(logits, dim=-1)
            predicted_label = predicted_label.item()
        predicted_label_text = 'Spam' if predicted_label == 1 else 'Ham'
        return jsonify({'prediction': predicted_label_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)