"""Fine-tune a small local student model with MLX LoRA on fake-game play.

``data`` builds (exact agent prompt, target action JSON) pairs and scores play with a process
reward. ``mlx`` wraps mlx-lm (an in-process client, and the LoRA trainer as a subprocess).
``dagger`` runs the roll out, relabel, soft SFT loop. Everything except the mlx-lm calls
runs without mlx-lm installed (``pip install -e '.[mlx]'``).
"""
