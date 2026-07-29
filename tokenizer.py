"""Tokenizer with BPE support trained on the provided corpus.
Maintains byte-level fallback for arbitrary UTF-8 text.

load() -> tokenizer with .encode(str) -> list[int], .decode(list[int]) -> str, .vocab_size
"""
import json
import os


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "byte"}, f)


class BPETokenizer:
    """Byte-level BPE tokenizer. Fast encode using priority-ordered merges."""
    
    def __init__(self):
        self.merges = []  # list of (int, int) pairs
        self.vocab_size = 256
        self._build_vocab()
    
    def _build_vocab(self):
        """Build decode table from merges."""
        self.vocab = {}
        for i in range(256):
            self.vocab[i] = bytes([i])
        # Build merge priority: earlier merges have higher priority
        self.merge_map = {}
        for idx, (a, b) in enumerate(self.merges):
            new_id = 256 + idx
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            if (a, b) not in self.merge_map:
                self.merge_map[(a, b)] = new_id
        self.vocab_size = 256 + len(self.merges)
    
    def train(self, text, num_merges=256):
        """Train BPE on text. num_merges = number of merge operations."""
        from collections import Counter
        raw = list(text.encode("utf-8"))
        ids = list(raw)
        
        self.merges = []
        for i in range(num_merges):
            counts = Counter()
            for j in range(len(ids) - 1):
                counts[(ids[j], ids[j + 1])] += 1
            if not counts:
                break
            pair = counts.most_common(1)[0][0]
            new_id = 256 + i
            new_ids = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and ids[j] == pair[0] and ids[j + 1] == pair[1]:
                    new_ids.append(new_id)
                    j += 2
                else:
                    new_ids.append(ids[j])
                    j += 1
            ids = new_ids
            self.merges.append(pair)
            if (i + 1) % 50 == 0:
                print(f"  BPE merge {i+1}/{num_merges}: {len(ids):,} tokens remaining "
                      f"(compression ratio: {len(raw)/len(ids):.2f}x)")
        
        self._build_vocab()
        print(f"BPE training done: {num_merges} merges, vocab_size={self.vocab_size}, "
              f"compression: {len(raw):,} -> {len(ids):,} tokens "
              f"({len(raw)/len(ids):.2f}x)")
    
    def encode(self, text):
        """Encode text to token ids using trained merges — optimized."""
        ids = list(text.encode("utf-8"))
        # Apply merges in priority order (earlier merges first)
        for idx, (a, b) in enumerate(self.merges):
            new_id = 256 + idx
            i = 0
            new_ids = []
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids
    
    def decode(self, ids):
        """Decode token ids back to text."""
        raw = b""
        for tid in ids:
            raw += self.vocab[tid]
        return raw.decode("utf-8", errors="replace")
    
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "bpe", "merges": self.merges}, f)
    
    @classmethod
    def from_file(cls, path):
        tok = cls()
        with open(path, "r") as f:
            data = json.load(f)
        tok.merges = [tuple(m) for m in data["merges"]]
        tok._build_vocab()
        return tok


def train_bpe(corpus_path, num_merges=512):
    """Train a BPE tokenizer on the given corpus and save it."""
    print(f"Training BPE tokenizer with {num_merges} merges...")
    text = open(corpus_path, encoding="utf-8").read()
    tok = BPETokenizer()
    tok.train(text, num_merges=num_merges)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_vocab.json")
    tok.save(save_path)
    print(f"Saved BPE vocab to {save_path}")
    return tok


def load(path=None):
    """Return the tokenizer used by evaluate.py."""
    vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_vocab.json")
    if os.path.exists(vocab_path):
        return BPETokenizer.from_file(vocab_path)
    return ByteTokenizer()


if __name__ == "__main__":
    import sys
    corpus = sys.argv[1] if len(sys.argv) > 1 else "../data/train_corpus.txt"
    merges = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    train_bpe(corpus, merges)
