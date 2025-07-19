# 68-71??

# uv run cs336_basics/train.py optimizer.cosine_lr_params='[1e-2, 1e-5, 8000, 20_000]' max_steps=8_000
# uv run cs336_basics/train.py optimizer.cosine_lr_params='[1e-1, 1e-5, 8000, 20_000]' max_steps=8_000
# uv run cs336_basics/train.py optimizer.cosine_lr_params='[1e-2, 1e-5, 4000, 20_000]' max_steps=8_000
# uv run cs336_basics/train.py optimizer.cosine_lr_params='[3e-1, 1e-5, 8000, 20_000]' max_steps=8_000

# 2 and 4 diverge, need more time to converge

# 72-75
# uv run cs336_basics/train.py optimizer.cosine_lr_params='[3e-2, 1e-5, 4000, 20_000]' max_steps=8_000
# uv run cs336_basics/train.py optimizer.cosine_lr_params='[6e-2, 1e-5, 2000, 20_000]' max_steps=8_000
# uv run cs336_basics/train.py optimizer.cosine_lr_params='[3e-2, 1e-5, 4000, 20_000]' max_steps=8_000
# uv run cs336_basics/train.py optimizer.cosine_lr_params='[3e-2, 1e-5, 8000, 20_000]' max_steps=8_000

# all never converge, need more steps?
# think 75 says to me that i like 3e-2, but need to stop earlier at like 400

uv run cs336_basics/train.py optimizer.cosine_lr_params='[3e-2, 1e-5, 400, 40_000]' max_steps=20_000
uv run cs336_basics/train.py optimizer.cosine_lr_params='[3e-2, 1e-4, 400, 40_000]' max_steps=20_000
uv run cs336_basics/train.py optimizer.cosine_lr_params='[5e-2, 1e-5, 400, 40_000]' max_steps=20_000
uv run cs336_basics/train.py optimizer.cosine_lr_params='[7e-2, 1e-5, 400, 40_000]' max_steps=20_000