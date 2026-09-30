"""字符级分词器：中文情感分类任务上不依赖任何预训练词表。"""
from collections import Counter

import torch

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
CLS_TOKEN = "<cls>"

PAD_ID, UNK_ID, CLS_ID = 0, 1, 2


class CharTokenizer:
    """把一句话切成字符 id 序列，句首插入 <cls>。"""

    def __init__(self, vocab, max_len: int = 256, add_cls: bool = True):
        self.vocab = dict(vocab)
        self.max_len = max_len
        self.add_cls = add_cls

    @classmethod
    def build(cls, texts, max_len: int = 256, min_freq: int = 2, add_cls: bool = True):
        counter = Counter()
        for t in texts:
            counter.update(str(t))
        vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID, CLS_TOKEN: CLS_ID}
        for ch, freq in counter.most_common():
            if freq >= min_freq and ch not in vocab:
                vocab[ch] = len(vocab)
        return cls(vocab, max_len=max_len, add_cls=add_cls)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text, max_len=None) -> torch.Tensor:
        limit = (self.max_len if max_len is None else max_len) - (1 if self.add_cls else 0)
        ids = [self.vocab.get(ch, UNK_ID) for ch in str(text)][:limit]
        if self.add_cls:
            ids = [CLS_ID] + ids
        if not ids:
            ids = [UNK_ID]
        return torch.tensor(ids, dtype=torch.long)

    def __call__(self, text) -> torch.Tensor:
        return self.encode(text)

    def to_dict(self) -> dict:
        return {"vocab": self.vocab, "max_len": self.max_len, "add_cls": self.add_cls}

    @classmethod
    def from_dict(cls, d) -> "CharTokenizer":
        return cls(d["vocab"], max_len=d["max_len"], add_cls=d["add_cls"])


def pad_batch(seqs, pad_id: int = PAD_ID):
    """把一批不等长 id 序列补齐成 (B, T) 张量。"""
    max_len = max(int(s.size(0)) for s in seqs)
    batch = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, : s.size(0)] = s
    return batch
