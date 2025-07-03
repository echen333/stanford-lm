from tests.adapters import run_train_bpe

a, b = run_train_bpe("data/TinyStoriesV2-GPT4-valid.txt", vocab_size=10000, special_tokens=["<|endoftext|>"])

print("a", a)
print("b", b)
