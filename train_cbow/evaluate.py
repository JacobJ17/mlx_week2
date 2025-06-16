import torch
import pickle

vocab_to_int = pickle.load(open('./tkn_words_to_ids.pkl', 'rb'))
int_to_vocab = pickle.load(open('./tkn_ids_to_words.pkl', 'rb'))

def topk(cbow_model, query_word: str = 'computer', top_k: int = 5) -> None:
    """
    Print the top_k words most similar to query_word based on cosine similarity
    of embedding vectors from the CBOW model cbow_model.
    """
    if query_word not in vocab_to_int:
        print(f'Word "{query_word}" not found in vocabulary.')
        return
    
    idx = vocab_to_int[query_word]
    query_vec = cbow_model.embeddings.weight[idx].detach().unsqueeze(0)
    
    with torch.no_grad():
        query_vec = torch.nn.functional.normalize(query_vec, p=2, dim=1)
        embeddings = torch.nn.functional.normalize(cbow_model.embeddings.weight.detach(), p=2, dim=1)
        similarity_scores = torch.matmul(embeddings, query_vec.squeeze())
        top_vals, top_idxs = torch.topk(similarity_scores, top_k + 1)  # +1 to skip the query itself

        print(f'\nTop {top_k} words similar to "{query_word}":')
        rank = 0
        for val, idx in zip(top_vals, top_idxs):
            word = int_to_vocab[idx.item()]
            if word == query_word:
                continue  # skip the query word itself
            print(f'  {word}: {val.item():.4f}')
            rank += 1
            if rank == top_k:
                break
