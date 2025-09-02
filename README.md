# GPT2-from-Scratch: Spam Classification

This project implements a GPT-2-like architecture from scratch in PyTorch and fine-tunes it for SMS spam classification. The workflow includes model development, experimentation with fine-tuning strategies, and deployment via a Flask API.

## Features
- Custom implementation of GPT-2 architecture
- Multi-head self-attention mechanism
- Fine-tuning for spam classification
- Experiment tracking with Weights & Biases (wandb)
- Flask API for local inference

## Project Structure
- `src/` — Core model, training, and utility scripts
- `data/` — Processed and raw SMS spam datasets
- `models/` — Saved model checkpoints
- `app.py` — Flask API for serving predictions
- Notebooks for exploration and experiments

## Key Challenges
### Model Architecture
- Extending the self-attention mechanism to multi-head attention was a significant challenge, requiring careful tensor manipulation and understanding of attention mechanics.

### Fine-tuning Strategy
- Deciding which layers to fine-tune was non-trivial. Two experiments were run using Weights & Biases:
	- Different combinations of trainable layers were tested.
	- Based on results, the best performance was achieved by fine-tuning the classifier head, the final transformer block, and the final normalization layer.
 <img width="1479" height="452" alt="image" src="https://github.com/user-attachments/assets/17e0cb42-d804-4a51-a3fc-eaa46d22163a" />


## Usage
1. **Install dependencies** (see `requirements.txt` or use the environment from wandb logs).
2. **Train or fine-tune the model** using scripts in `src/`.
3. **Run the Flask API locally:**
	 ```bash
	 python app.py
	 ```
4. **Send requests to the API** for spam prediction.

## Future Work
- Deploy the Flask API to a cloud service for production use.

## Credits
- Model inspired by OpenAI's GPT-2 architecture.
- Dataset: SMS Spam Collection.
- Experiment tracking: Weights & Biases.

