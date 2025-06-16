import torch
from torch.utils.data import Dataset
import pickle
import bisect


class CBOW(Dataset):
    def __init__(self, 
                 vocab_to_int_path: str = './tkn_words_to_ids.pkl',
                 corpus_path: str = './corpus.pkl',
                 delimiter: str = '<DELIMIT>',  # ID that separates boundaries
                 context_size: int = 3,
                 vocab_size: int = None):  # Number of tokens before and after center
        self.vocab_to_int = pickle.load(open(vocab_to_int_path, 'rb'))
        # If vocab_size is set, keep only the first vocab_size items
        if vocab_size is not None:
            # Assume vocab_to_int is sorted by frequency (if not, sort it)
            items = list(self.vocab_to_int.items())[:vocab_size]
            self.vocab_to_int = dict(items)
        self.int_to_vocab = {v: k for k, v in self.vocab_to_int.items()}
        self.corpus_words = pickle.load(open(corpus_path, 'rb'))
        self.tokens = [self.vocab_to_int.get(word, 0) for word in self.corpus_words]

        self.delimiter = self.vocab_to_int[delimiter]
        self.delimiter_positions = [i for i, t in enumerate(self.tokens) if t == self.delimiter]
        self.context_size = context_size

        # Precompute valid indices that are not too close to boundaries and do not cross delimiters
        self.valid_indices = [
            i for i in range(len(self.tokens))
            if self._valid_context(i)
        ]

    def _valid_context(self, idx):
        start = idx - self.context_size
        end = idx + self.context_size
        # Ensure context window is within bounds
        if start < 0 or end >= len(self.tokens):
            return False
        # Find delimiters in [start, end] excluding center idx
        left = bisect.bisect_left(self.delimiter_positions, start)
        right = bisect.bisect_right(self.delimiter_positions, end)
        # If any delimiter in window => invalid
        return (right - left) == 0    

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        center_idx = self.valid_indices[idx]
        start = center_idx - self.context_size
        end = center_idx + self.context_size + 1

        context = self.tokens[start:center_idx] + self.tokens[center_idx + 1:end]
        target = self.tokens[center_idx]

        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


if __name__ == "__main__":
    delimiter = "<DELIMIT>"  # assuming 0 is reserved for delimiter or padding
    ds = CBOW(delimiter=delimiter, context_size=2)
    print("First 15 tokens:", ds.tokens[:15])
    print("Sample 5:", ds[5])

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=3, shuffle=True)
    batch = next(iter(dl))
    print("Batch inputs shape:", batch[0].shape)
    print("Batch targets shape:", batch[1].shape)
