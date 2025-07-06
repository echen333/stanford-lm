from typing import Iterable, Iterator
import regex as re
import os
from collections import defaultdict


import regex as re
from collections import defaultdict
from multiprocessing import Pool


def bpe_get_freqs_per_chunk(input_path, start, end, split_pattern):
    freqs2: dict[tuple[bytes], int] = defaultdict(int)
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for split in re.split(split_pattern, chunk):
        matches = re.finditer(PAT, split)
        for m in matches:
            tmp = m.group().encode()
            tmp2 = [bytes([x]) for x in tmp]
            freqs2[tuple(tmp2)] += 1
    return freqs2


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
):
    vocab: dict[int, bytes] = {}
    end_of_text_token = "<|endoftext|>"
    vocab_cnt = 0

    def add_vocab(word: bytes):
        nonlocal vocab_cnt
        vocab[vocab_cnt] = word
        vocab_cnt += 1

    add_vocab(end_of_text_token.encode())
    for i in range(256):
        add_vocab(bytes([i]))

    freqs: dict[tuple[bytes], int] = defaultdict(int)
    merges: list[tuple[bytes, bytes]] = []

    from cs336_basics.pretokenization_example import find_chunk_boundaries

    split_pattern = "|".join(special_tokens)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, os.cpu_count() // 2, end_of_text_token.encode())

        tasks = [(input_path, s, e, split_pattern) for s, e in zip(boundaries[:-1], boundaries[1:])]
        with Pool(os.cpu_count()) as pool:
            partials: list[dict] = pool.starmap(bpe_get_freqs_per_chunk, tasks)

        for chunk_freqs in partials:
            for x, y in chunk_freqs.items():
                freqs[x] += y

    while vocab_cnt < vocab_size:
        pairs: dict[tuple[bytes, bytes], int] = defaultdict(int)
        for tup, cnt in freqs.items():
            for i in range(len(tup) - 1):
                pairs[(tup[i], tup[i + 1])] += cnt

        # Find max in pairs to merge
        max_freq = max(pairs.values())
        all_max_freq_pairs = list(filter(lambda x: pairs[x] == max_freq, pairs.keys()))
        to_merge = max(all_max_freq_pairs)

        if to_merge is None:
            break
        merged_bytes = to_merge[0] + to_merge[1]
        add_vocab(merged_bytes)
        merges.append(to_merge)

        # Update freqs by merging old
        freqs2 = defaultdict(int)
        for tup, cnt in freqs.items():
            tmp_list = []
            skip_next = False
            for i, val in enumerate(tup):
                if skip_next:
                    skip_next = False
                elif i != len(tup) - 1 and val == to_merge[0] and tup[i + 1] == to_merge[1]:
                    skip_next = True
                    tmp_list.append(merged_bytes)
                else:
                    tmp_list.append(val)
            freqs2[tuple(tmp_list)] += cnt

        freqs = freqs2

    return vocab, merges


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.to_id: dict[bytes, int] = {y: x for x, y in vocab.items()}
        self.merges = merges
        self.special_tokens = (
            list(reversed(sorted(special_tokens, key=lambda x: len(x)))) if special_tokens is not None else None
        )
        self._compiled = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output) and (optionally) a list of special tokens"""
        # with open(vocab_filepath, "r") as f:
        pass

    def _split_on_special(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]

        pattern = "|".join(map(re.escape, self.special_tokens))
        pat = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"

        # pattern = "|".join(self.special_tokens)
        splitter = self._compiled or re.compile(pattern)
        self._compiled = splitter

        return re.split(pat, text)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        # split across special tokens -> pretokenize -> apply merges
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        ret = []

        for split in self._split_on_special(text):
            if self.special_tokens is not None and split in self.special_tokens:
                ret.append(self.to_id[split.encode()])
                continue

            for m in re.finditer(PAT, split):
                tmp = m.group().encode()
                tmp2: list[bytes] = [bytes([x]) for x in tmp]
                for merge in self.merges:
                    tmp3 = []
                    skip_next = False
                    for idx, val in enumerate(tmp2):
                        if skip_next:
                            skip_next = False
                        elif idx != len(tmp2) - 1 and val == merge[0] and tmp2[idx + 1] == merge[1]:
                            skip_next = True
                            tmp3.append(merge[0] + merge[1])
                        else:
                            tmp3.append(val)
                    tmp2 = tmp3
                ret.extend([self.to_id[x] for x in tmp2])

        return ret

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory."""
        for text in iterable:
            data = self.encode(text)  # should reimpl the extend part or this ok?
            yield from data

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        encoded_bytes = bytes([])
        for id in ids:
            encoded_bytes += self.vocab.get(id)
        # replace errors with U+FFFD
        return encoded_bytes.decode(errors="replace")


from pathlib import Path
import json


def save_bpe_state(vocab, merges, vocab_json_path, merges_txt_path):
    vocab_json_path = Path(vocab_json_path)
    vocab_json_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2))
    pass


if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 100
    end_of_text_token = "<|endoftext|>"
    special_tokens = [end_of_text_token]
    print("HIHI")
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    print("vocab", vocab)
    print("merges", merges)
    save_bpe_state(vocab, merges, "data/tiny_stories_vocab.json", "data/tiny_stories_merges.txt")

    # with open(data_path, "r") as f:

    # tokenizer = Tokenizer()
