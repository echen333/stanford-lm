uv run cs336_systems/benchmarking_script.py
uv run cs336_systems/benchmarking_script.py --d_model=1024 --d_ff=4096 --num_layers=24 --num_heads=16
uv run cs336_systems/benchmarking_script.py --d_model=1280 --d_ff=5120 --num_layers=36 --num_heads=20
uv run cs336_systems/benchmarking_script.py --d_model=1600 --d_ff=6400 --num_layers=48 --num_heads=25
uv run cs336_systems/benchmarking_script.py --d_model=2560 --d_ff=10240 --num_layers=32 --num_heads=32

uv run cs336_systems/benchmarking_script.py --mixed_precision=True
uv run cs336_systems/benchmarking_script.py --d_model=1024 --d_ff=4096 --num_layers=24 --num_heads=16 --mixed_precision=True
uv run cs336_systems/benchmarking_script.py --d_model=1280 --d_ff=5120 --num_layers=36 --num_heads=20 --mixed_precision=True
uv run cs336_systems/benchmarking_script.py --d_model=1600 --d_ff=6400 --num_layers=48 --num_heads=25 --mixed_precision=True
uv run cs336_systems/benchmarking_script.py --d_model=2560 --d_ff=10240 --num_layers=32 --num_heads=32 --mixed_precision=True