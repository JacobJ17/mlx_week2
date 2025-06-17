import torch
from cbow_model import CBOW  # Adjust import if needed

def preprocess(text: str) -> list[str]:
    if not isinstance(text, str):
        return ""
    
    # Normalise some symbols
    text = text.lower()
    text = re.sub(r"&", "and", text)
    text = re.sub(r"%", "percent", text)

    # Replace specific punctuation with tokens
    # text = text.replace("-", " ")
    text = text.replace("-", ' <HYPHEN> ')
    text = text.replace('.',  ' <PERIOD> ')
    text = text.replace(',',  ' <COMMA> ')
    text = text.replace(';',  ' <SEMICOLON> ')
    text = text.replace('!',  ' <EXCLAMATION_MARK> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')

    # Remove any other punctuation.
    text = re.sub(r'[^\w\s<>]', '', text)
    words = text.split()
    stats = collections.Counter(words)
    words = [word for word in words if stats[word] > 5 or word == '<DELIMIT>']
    return words

def preprocess_for_inference(text: str) -> list[str]:
    text.replace("<", "").replace(">", "")
    return preprocess(text)

def load_cbow_for_inference(checkpoint_path, vocab_size=30000, emb_dim=128, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CBOW(vocab_size=vocab_size, emb_dim=emb_dim)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded CBOW model from {checkpoint_path}")
    return model

# Usage:
# model = load_cbow_for_inference('checkpoints/cbow_model.pth', vocab_size=30000, emb_dim=128)