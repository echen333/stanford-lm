from typing import Iterable, Iterator
import regex as re
import os
from collections import defaultdict
import cProfile


import regex as re
from collections import defaultdict
from multiprocessing import Pool
import multiprocessing as mp
from itertools import repeat


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


def find_tups(chunk, search_bytes):
    ret = []
    for tup in chunk:
        for i, val in enumerate(tup):
            if val == search_bytes[0] and i != len(tup) - 1 and tup[i + 1] == search_bytes[1]:
                ret.append(tup)
                break
    return ret


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
        boundaries = find_chunk_boundaries(f, os.cpu_count(), end_of_text_token.encode())

        tasks = [(input_path, s, e, split_pattern) for s, e in zip(boundaries[:-1], boundaries[1:])]
        with Pool(os.cpu_count()) as pool:
            partials: list[dict] = pool.starmap(bpe_get_freqs_per_chunk, tasks)

        for chunk_freqs in partials:
            for x, y in chunk_freqs.items():
                freqs[x] += y

    print("Finished pretokenizing")
    from collections import Counter

    pairs: dict[tuple[bytes, bytes], int] = Counter()
    for tup, cnt in freqs.items():
        for i in range(len(tup) - 1):
            pairs[(tup[i], tup[i + 1])] += cnt
    print("Finished initializing pairs")

    ctx = mp.get_context("fork")
    n_procs = os.cpu_count()
    # n_procs = 4
    # all_keys = list(freqs.keys())

    all_keys = list(freqs.keys())
    num_deleted = 0
    num_all_keys = len(all_keys)

    # Create pool once and reuse it - this eliminates the overhead of creating pools every iteration
    with ctx.Pool(n_procs) as pool:
        while vocab_cnt < vocab_size:
            to_merge = max(pairs.items(), key=lambda x: (x[1], x[0]))[0]

            if to_merge is None:
                break
            merged_bytes = to_merge[0] + to_merge[1]
            add_vocab(merged_bytes)
            merges.append(to_merge)

            # Update freqs by merging old
            tups_with_merged = []
            chunks = [all_keys[i::n_procs] for i in range(n_procs)]

            partials = pool.starmap(find_tups, zip(chunks, repeat(to_merge)))
            # partials = [find_tups(chunks[0], to_merge)] # if n_procs is 1

            for partial in partials:
                tups_with_merged.extend(partial)

            to_delete: list[tup] = []
            to_upd: dict[tuple[bytes], int] = defaultdict(int)
            for tup in tups_with_merged:
                cnt = freqs[tup]
                new_list = []
                skip_next = False
                for i, val in enumerate(tup):
                    if skip_next:
                        skip_next = False
                    elif i != len(tup) - 1 and val == to_merge[0] and tup[i + 1] == to_merge[1]:
                        skip_next = True
                        new_list.append(merged_bytes)
                    else:
                        new_list.append(val)

                for i in range(len(tup) - 1):
                    pairs[(tup[i], tup[i + 1])] -= cnt
                for i in range(len(new_list) - 1):
                    pairs[(new_list[i], new_list[i + 1])] += cnt

                to_upd[tuple(new_list)] = cnt
                to_delete.append(tup)

            for tup in to_delete:
                del freqs[tup]
            num_deleted += len(to_delete)

            for tup, cnt in to_upd.items():
                freqs[tup] += cnt
                all_keys.append(tup)

            del pairs[(to_merge[0], to_merge[1])]

            if int(3 * num_deleted) > num_all_keys:
                # print("RESET all_keys")
                num_deleted = 0
                all_keys = list(freqs.keys())
            if vocab_cnt % (vocab_size // 25) == 0:
                print(f"Have {vocab_cnt} tokens in vocab now")

    return vocab, merges


class Tokenizer:
    """Given vocab and merges after training BPE, encode and decode bytes."""

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
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None, encoding: str = "latin-1"):
        """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output) and (optionally) a list of special tokens"""

        # shouldve just used pkl here...
        vocab_str = Path(vocab_filepath).read_text()
        raw_tok2id = json.loads(vocab_str)
        vocab: dict[int, bytes] = {
            int(idx): base64.b64decode(token.encode("ascii")) for idx, token in raw_tok2id.items()
        }

        assert os.path.splitext(merges_filepath)[1] == ".pkl"
        with open(merges_filepath, "rb") as f:
            merges = pkl.load(f)

        return cls(vocab, merges, special_tokens)

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
import json, base64
import pickle as pkl


def save_bpe_state(vocab: dict[int, bytes], merges, vocab_json_path, merges_path, encoding: str = "latin-1"):
    safe_vocab = {idx: base64.b64encode(b).decode("ascii") for idx, b in vocab.items()}
    Path(vocab_json_path).write_text(json.dumps(safe_vocab, ensure_ascii=False))

    with open(merges_path, "wb") as f:
        pkl.dump(merges, f)


def encode_text_to_npy(data_path, path_prefix: str, special_tokens=None):
    vocab_path = path_prefix + "_vocab.json"
    merges_path = path_prefix + "_merges.pkl"
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

    # am i pretokenizing or what idk
    pass


def main():
    # input_path = "data/owt_train.txt"
    # prefix_path = "data/owt_train"
    # input_path = "data/owt_valid.txt"
    # prefix_path = "data/owt_valid"
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    prefix_path = "data/tiny_stories_train_testing"
    # input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    # prefix_path = "data/tiny_stories_train_testing"

    vocab_size = 32000 if "owt_train" in input_path else 10000
    end_of_text_token = "<|endoftext|>"
    special_tokens = [end_of_text_token]
    print(f"Training bpe on path {input_path}")
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    print("found vocab and merges")

    vocab_json_path = f"{prefix_path}_{vocab_size}_vocab.json"
    merges_txt_path = f"{prefix_path}_{vocab_size}_merges.pkl"
    save_bpe_state(vocab, merges, vocab_json_path, merges_txt_path)

    tokenizer: Tokenizer = Tokenizer.from_files(vocab_json_path, merges_txt_path, special_tokens)
    print(tokenizer.vocab)
    print(tokenizer.merges)


if __name__ == "__main__":
    # cProfile.run("main()")
    main()
