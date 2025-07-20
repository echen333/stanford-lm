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

# 77, so bad for some reason
# uv run cs336_basics/train.py optimizer.cosine_lr_params='[3e-2, 1e-5, 400, 40_000]' max_steps=20_000

# yea i just have no idea what im doing. decreasing T_init messes things up...
# should prob just run more to more steps

# 80, 81, 82 these are aight, 80 diverge, 81 seems best
# uv run cs336_basics/train.py optimizer.cosine_lr_params='[5e-2, 1e-5, 400, 40_000]' max_steps=20_000
# uv run cs336_basics/train.py optimizer.cosine_lr_params='[1e-3, 1e-5, 400, 40_000]' max_steps=20_000
# uv run cs336_basics/train.py optimizer.cosine_lr_params='[3e-3, 1e-5, 400, 40_000]' max_steps=20_000

# 83, 84, 85 => 84 best by a little, very little diff as expected
# uv run cs336_basics/train.py optimizer.cosine_lr_params='[1e-3, 3e-5, 400, 40_000]' max_steps=20_000
# uv run cs336_basics/train.py optimizer.cosine_lr_params='[1e-3, 1e-5, 400, 40_000]' max_steps=20_000
# uv run cs336_basics/train.py optimizer.cosine_lr_params='[1e-3, 1e-6, 400, 40_000]' max_steps=20_000

# Slight tunes to run 84 in LR and T_warmup, 87-90
# uv run cs336_basics/train.py optimizer.cosine_lr_params='[0.5e-3, 1e-5, 400, 40_000]' max_steps=20_000
# uv run cs336_basics/train.py optimizer.cosine_lr_params='[1e-3, 1e-5, 800, 40_000]' max_steps=20_000
# # 1.43
# uv run cs336_basics/train.py optimizer.cosine_lr_params='[3e-3, 1e-5, 400, 40_000]' max_steps=40_000
# # Full run
# uv run cs336_basics/train.py optimizer.cosine_lr_params='[5e-3, 1e-5, 400, 40_000]' max_steps=40_000


# BATCH_SIZE_EXPERIMENTS 94-
uv run cs336_basics/train.py batch_size=64 max_steps=10_000
uv run cs336_basics/train.py batch_size=128 max_steps=5_000
uv run cs336_basics/train.py batch_size=256 max_steps=2_500
uv run cs336_basics/train.py batch_size=512 max_steps=1_250
uv run cs336_basics/train.py batch_size=1024 max_steps=625
uv run cs336_basics/train.py batch_size=16 max_steps=20_000
uv run cs336_basics/train.py batch_size=1 max_steps=20_000