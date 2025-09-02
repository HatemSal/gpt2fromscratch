from src.utils import get_pretrained_gpt2, get_dataloaders
import torch
import tiktoken
import time
import wandb
from src.engine import train_classifier

tokenizer = tiktoken.get_encoding("gpt2")
data_path = "data/processed"
train_loader, val_loader, test_loader = get_dataloaders(data_path, tokenizer)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_choice = "gpt2-small (124M)"
models_dir = "gpt2"
model, config = get_pretrained_gpt2(model_choice, models_dir)

for param in model.parameters():
    param.requires_grad = False
    
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=config['emb_dim'],
    out_features=num_classes
)
model.to(device)

for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
    
wandb.init(project="gpt2-spam-finetuning", group="unfrozen-finaltrfblock-finalBN-classhead")

config = {
    "learning_rate":5e-5,
    "weight_decay":0.1,
    "num_epochs":5
}
wandb.config.update(config)

start_time = time.time()
optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=wandb.config.num_epochs, eval_freq=50,
        eval_iter=5
    )

end_time = time.time()
time_taken = (end_time - start_time)/60
wandb.log({"time_taken_mins":time_taken})
wandb.finish()
print(f"Training completed in {time_taken:.2f}")

torch.save(model.state_dict(),"models/unfrozen_classhead")