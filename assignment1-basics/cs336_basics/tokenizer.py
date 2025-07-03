from typing import Iterable, Iterator
import regex as re


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.to_id: dict[bytes, int] = {y: x for x, y in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        self._compiled = None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output) and (optionally) a list of special tokens"""
        pass

    def _split_on_special(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]

        # pattern = "|".join(map(re.escape, self.special_tokens))
        pattern = "|".join(self.special_tokens)
        splitter = self._compiled or re.compile(pattern)
        self._compiled = splitter

        return re.split(pattern, text)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        # split across special tokens -> pretokenize -> apply merges
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        ret = []

        for split in self._split_on_special(text):
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
