"""Deterministic DNA BPE; preserves the historical notebook merge rules."""
from collections import Counter
from pathlib import Path
import json

class DNABytePairEncoder:
    def __init__(self):
        self.base_tokens = ["A", "C", "G", "T", "N"]
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[MASK]"]

        self.merges = []
        self.token_to_id = {}
        self.fitted = False

    @staticmethod
    def normalize(sequence):
        return "".join(
            char if char in "ACGT" else "N"
            for char in sequence.upper()
        )

    @staticmethod
    def merge_pair(tokens, pair):
        """从左到右合并，不重复使用已合并的位置。"""
        result = []
        i = 0

        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                result.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                result.append(tokens[i])
                i += 1

        return result

    def fit(self, sequences, vocab_size=128, min_frequency=2):
        minimum_size = len(self.special_tokens) + len(self.base_tokens)

        if vocab_size < minimum_size:
            raise ValueError(f"vocab_size 至少为 {minimum_size}")

        # 每条序列各自分词，避免跨序列边界统计
        corpus = [
            list(self.normalize(sequence))
            for sequence in sequences
        ]

        vocabulary = self.special_tokens + self.base_tokens
        self.merges = []

        while len(vocabulary) < vocab_size:
            pair_counts = Counter()

            for tokens in corpus:
                for pair in zip(tokens, tokens[1:]):
                    if "N" not in pair[0] and "N" not in pair[1]:
                        pair_counts[pair] += 1

            # 只考虑会生成新 token 的组合
            candidates = [
                (pair, count)
                for pair, count in pair_counts.items()
                if pair[0] + pair[1] not in vocabulary
            ]

            if not candidates:
                break

            # 频率相同时按字符顺序选，保证结果可重复
            pair, count = min(
                candidates,
                key=lambda item: (-item[1], item[0]),
            )

            if count < min_frequency:
                break

            self.merges.append(pair)
            vocabulary.append(pair[0] + pair[1])

            corpus = [
                self.merge_pair(tokens, pair)
                for tokens in corpus
            ]

        self.token_to_id = {
            token: index
            for index, token in enumerate(vocabulary)
        }
        self.fitted = True
        return self

    def tokenize(self, sequence):
        if not self.fitted:
            raise ValueError("请先使用训练集调用 fit()")

        tokens = list(self.normalize(sequence))

        # 按训练时学到的顺序应用合并规则
        for pair in self.merges:
            tokens = self.merge_pair(tokens, pair)

        return tokens

    def encode(self, sequence):
        tokens = self.tokenize(sequence)

        return [
            self.token_to_id.get(token, self.token_to_id["[UNK]"])
            for token in tokens
        ]
    def to_state(self):
        return {"token_to_id": self.token_to_id,
                "merges": [list(p) for p in self.merges],
                "base_tokens": self.base_tokens,"special_tokens": self.special_tokens}

    @classmethod
    def from_state(cls, state):
        obj = cls()
        obj.token_to_id = dict(state["token_to_id"])
        obj.merges = [tuple(pair) for pair in state["merges"]]
        obj.base_tokens = list(state["base_tokens"])
        obj.special_tokens = list(state["special_tokens"])
        ids = sorted(obj.token_to_id.values())
        if ids != list(range(len(ids))):
            raise ValueError("Invalid token IDs")
        required = obj.base_tokens + obj.special_tokens
        if any(token not in obj.token_to_id for token in required):
            raise ValueError("Incomplete vocabulary")
        for left, right in obj.merges:
            if any(token not in obj.token_to_id for token in [left,right,left+right]):
                raise ValueError("Invalid merge rule")
        obj.fitted = True
        return obj

    def save(self, path):
        if not self.fitted:
            raise ValueError("Tokenizer is not fitted")
        Path(path).write_text(json.dumps(self.to_state(),indent=2),encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_state(json.loads(Path(path).read_text(encoding="utf-8")))
