from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import torch
import pandas as pd

def get_embeddings(texts, tokenizer, model, device="cpu", batch_size=32):
    all_embeddings = []

    texts = list(map(str, texts))

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        encodings = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling

        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)

if __name__ == "__main__":
    # Load dataset
    print("Loading dataset...")
    path = f"data/preprocessed_raceform.csv"
    ds = pd.read_csv(path, low_memory=False)

    # Embed comments
    print("Embedding comments using SentenceTransformer...")
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    embeddings = get_embeddings(ds["comment"].tolist(), tokenizer, model, device, batch_size=128)
    
    # Save embeddings
    print("Saving embeddings...")
    emb_df = pd.DataFrame(embeddings.numpy(), columns=[f"topic_{i}" for i in range(embeddings.shape[1])])
    ds = pd.concat([ds, emb_df], axis=1)
    ds.to_csv("data/preprocessed_raceform_with_embeddings.csv", index=False)