from typing import Iterable, Iterator
import time
import regex as re
import os
from collections import defaultdict
import cProfile
import sys
import pickle
import regex as re
from collections import defaultdict, Counter
from multiprocessing import Pool
import multiprocessing as mp
from itertools import repeat
import numpy as np
from pathlib import Path
import json, base64
import pickle as pkl


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


def find_tups2(ind, pth, search_bytes):
    ret = []
    upd_pairs = Counter()
    data_path = os.path.join(pth, f"data_{ind}.pkl")
    with open(data_path, "rb") as f:
        obj = pkl.load(f)

    for tup in obj:
        for i, val in enumerate(tup[:-1]):
            if val == search_bytes[0] and tup[i + 1] == search_bytes[1]:
                ret.append(tup)
                break

    merged_bytes = search_bytes[0] + search_bytes[1]
    for tup in ret:
        cnt = obj[tup]
        new_list = []
        skip_next = False
        for i, val in enumerate(tup):
            if skip_next:
                skip_next = False
            elif val == search_bytes[0] and i != len(tup) - 1 and tup[i + 1] == search_bytes[1]:
                skip_next = True
                new_list.append(merged_bytes)
            else:
                new_list.append(val)

        for i in range(len(tup) - 1):
            upd_pairs[(tup[i], tup[i + 1])] -= cnt
        for i in range(len(new_list) - 1):
            upd_pairs[(new_list[i], new_list[i + 1])] += cnt

        del obj[tup]
        obj[tuple(new_list)] = cnt

    with open(data_path, "wb") as f:
        pkl.dump(obj, f)
        del obj

    return upd_pairs


def deep_dict_sizeof(dic):
    tot_size = 0
    for k, v in dic.items():
        tot_size += sys.getsizeof(k)
        tot_size += sys.getsizeof(v)
    return tot_size


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], prefix_path: str = None, debug=False
):
    start_time = time.time()
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
    ctx = mp.get_context("fork")
    n_procs = os.cpu_count()
    n_chunks = os.cpu_count() * 2
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, os.cpu_count(), end_of_text_token.encode())

        tasks = [(input_path, s, e, split_pattern) for s, e in zip(boundaries[:-1], boundaries[1:])]
        with Pool(os.cpu_count()) as pool:
            partials: list[dict] = pool.starmap(bpe_get_freqs_per_chunk, tasks)

        for chunk_freqs in partials:
            for x, y in chunk_freqs.items():
                freqs[x] += y

    print("Finished pretokenizing", time.time() - start_time)

    pairs: dict[tuple[bytes, bytes], int] = Counter()
    for tup, cnt in freqs.items():
        for i in range(len(tup) - 1):
            pairs[(tup[i], tup[i + 1])] += cnt
    print("Finished initializing pairs", time.time() - start_time)

    # initialize pkl split
    pkl_dir = os.path.join(os.path.dirname(Path(input_path)), "pkl")
    os.makedirs(pkl_dir, exist_ok=True)
    freqs_keys = list(freqs.keys())
    freqs_chunks = [freqs_keys[i::n_chunks] for i in range(n_chunks)]
    for i in range(n_chunks):
        with open(os.path.join(pkl_dir, f"data_{i}.pkl"), "wb") as f:
            obj = {x: freqs[x] for x in freqs_chunks[i]}
            pickle.dump(obj, f)
    print("Finished dumping freqs")

    print("len of freqs", len(freqs.items()), deep_dict_sizeof(freqs))
    del freqs

    with ctx.Pool(n_procs) as pool:
        while vocab_cnt < vocab_size:
            to_merge = max(pairs.items(), key=lambda x: (x[1], x[0]))[0]

            if to_merge is None:
                break
            merged_bytes = to_merge[0] + to_merge[1]
            add_vocab(merged_bytes)
            merges.append(to_merge)

            chunks = [i for i in range(n_chunks)]
            # Load pickled chunks and search for merged_bytes
            pairs_updates = pool.starmap(find_tups2, zip(chunks, repeat(pkl_dir), repeat(to_merge)))

            for update in pairs_updates:
                for k, v in update.items():
                    pairs[k] += v

            del pairs[(to_merge[0], to_merge[1])]

            if debug and vocab_cnt % (vocab_size // 100) == 0:
                print(f"Have {vocab_cnt} tokens in vocab now", time.time() - start_time)
                print(f"Pairs size of {deep_dict_sizeof(pairs)}")
                if prefix_path is not None:
                    vocab_json_path = f"{prefix_path}_{vocab_cnt}_vocab.json"
                    merges_txt_path = f"{prefix_path}_{vocab_cnt}_merges.pkl"
                    save_bpe_state(vocab, merges, vocab_json_path, merges_txt_path)

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
        self.offset = len(self.vocab) - len(self.merges)
        self.merge_to_idx = {x: i for i, x in enumerate(self.merges)}
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

    def _encode_bytes(self, arr: list[bytes]):
        import heapq

        pq = []
        tot_set = set()
        for idx, val in enumerate(arr[:-1]):
            if (val, arr[idx + 1]) in self.merge_to_idx:
                merge_idx = self.merge_to_idx[(val, arr[idx + 1])]
                if merge_idx not in tot_set:
                    tot_set.add(merge_idx)
                    heapq.heappush(pq, merge_idx)

        prev = arr
        while pq:
            idx = heapq.heappop(pq)
            merge = self.merges[idx]

            nex = []
            skip_next = False
            for idx, val in enumerate(prev):
                if skip_next:
                    skip_next = False
                elif idx != len(prev) - 1 and val == merge[0] and prev[idx + 1] == merge[1]:
                    skip_next = True
                    nex.append(merge[0] + merge[1])
                else:
                    nex.append(val)
            prev = nex

            for idx, val in enumerate(prev[:-1]):
                if (val, prev[idx + 1]) in self.merge_to_idx:
                    merge_idx = self.merge_to_idx[(val, prev[idx + 1])]
                    if merge_idx not in tot_set:
                        tot_set.add(merge_idx)
                        heapq.heappush(pq, merge_idx)
        return [self.to_id[x] for x in prev]

    def encode(self, text: str, parallelize=False, chunk_idx=None) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        # split across special tokens -> pretokenize -> apply merges
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        ret = []

        from tqdm import tqdm

        splits = self._split_on_special(text)
        for ind, split in enumerate(splits):
            if self.special_tokens is not None and split in self.special_tokens:
                ret.append(self.to_id[split.encode()])
                continue

            subtasks = []
            for m in re.finditer(PAT, split):
                tmp = m.group().encode()
                tmp2: list[bytes] = [bytes([x]) for x in tmp]
                subtasks.append(tmp2)

            if parallelize:
                with Pool(os.cpu_count()) as pool:
                    for arr in tqdm(pool.imap(self._encode_bytes, subtasks), desc="Encoding", total=len(subtasks)):
                        ret.extend(arr)
            else:
                for val in subtasks:
                    arr = self._encode_bytes(val)
                    ret.extend(arr)

            if chunk_idx == 0 and ind % (len(splits) // 25) == 0:
                print("chunk idx", chunk_idx, " done with ", ind, f"out of {len(splits)}")

        return ret

    def _encode_chunk(self, save_base: str, ind: int, input_path: str, start: int, end: int):
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_encoded = self.encode(chunk, chunk_idx=ind)
            print("chunk_encoded", len(chunk_encoded), chunk_encoded[:10])
            arr = np.array(chunk_encoded, np.uint16)
            np.save(f"{save_base}_chunk_{ind}.npy", arr)
            print(f"done with chunk {ind}")

    def encode_file(self, input_path: str, save_path: str | None = None):
        start_time = time.time()
        from cs336_basics.pretokenization_example import find_chunk_boundaries

        end_of_text_token = "<|endoftext|>"

        if save_path is None:
            save_path = input_path
        save_base, _ = os.path.splitext(save_path)
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, os.cpu_count(), end_of_text_token.encode())

            print("len boundaries", len(boundaries), boundaries)
            print("got boundaries", time.time() - start_time)

            tasks = [
                (save_base, ind, input_path, s, e) for ind, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:]))
            ]
            print("tasks 0", tasks[0])
            with Pool(os.cpu_count()) as pool:
                pool.starmap(self._encode_chunk, tasks)
            print("finished encoding chunks", time.time() - start_time)

        tmp = np.load(f"{save_base}_chunk_0.npy")
        print("tmp", tmp[:10])
        arr = np.concat([np.load(f"{save_base}_chunk_{ind}.npy") for ind in range(len(tasks))])
        print("saving arr", arr[:100])
        np.save(save_base + ".npy", arr)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory."""
        for text in iterable:
            data = self.encode(text)
            yield from data

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        encoded_bytes = bytes([])
        for id in ids:
            encoded_bytes += self.vocab.get(id)
        # replace errors with U+FFFD
        return encoded_bytes.decode(errors="replace")


def save_bpe_state(vocab: dict[int, bytes], merges, vocab_json_path, merges_path, encoding: str = "latin-1"):
    safe_vocab = {idx: base64.b64encode(b).decode("ascii") for idx, b in vocab.items()}
    Path(vocab_json_path).write_text(json.dumps(safe_vocab, ensure_ascii=False))

    with open(merges_path, "wb") as f:
        pkl.dump(merges, f)


def train_main():
    input_path = "data/owt_train.txt"
    prefix_path = "data/owt_train"
    # input_path = "data/owt_valid.txt"
    # prefix_path = "data/owt_valid_test"
    # input_path = "data/TinyStoriesV2-GPT4-train.txt"
    # prefix_path = "data/tiny_stories_train_testing"
    # input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    # prefix_path = "data/tiny_stories_train_testing"

    vocab_size = 32000 if "owt_train" in input_path else 10000
    end_of_text_token = "<|endoftext|>"
    special_tokens = [end_of_text_token]
    print(f"Training bpe on path {input_path} with vocab_size {vocab_size}")
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens, prefix_path=prefix_path)
    print("found vocab and merges")

    vocab_json_path = f"{prefix_path}_{vocab_size}_vocab.json"
    merges_txt_path = f"{prefix_path}_{vocab_size}_merges.pkl"
    save_bpe_state(vocab, merges, vocab_json_path, merges_txt_path)


def encode_main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    # input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    prefix_path = "data/tiny_stories"
    vocab_size = 10000

    end_of_text_token = "<|endoftext|>"
    special_tokens = [end_of_text_token]

    vocab_json_path = f"{prefix_path}_{vocab_size}_vocab.json"
    merges_txt_path = f"{prefix_path}_{vocab_size}_merges.pkl"

    tokenizer: Tokenizer = Tokenizer.from_files(vocab_json_path, merges_txt_path, special_tokens)
    print("so common", tokenizer.decode([0, 11, 383, 327, 45, 259, 390, 477, 402, 824]))
    tokenizer.encode_file(input_path)


if __name__ == "__main__":
    # cProfile.run("main()")
    # train_main()
    encode_main()
