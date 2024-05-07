# battlesnake-train

This is a [Rye](https://rye-up.com/) project with [Rust](https://www.rust-lang.org/tools/install); install them and follow their instructions to set up the environment:

```sh
rye sync
```

`./battlesnake_gym/` is the simulation environment wrapping `../snork/`, `./snork_snakes_binding/` is the tree search agent binding for `../snork/`, `./src/battlesnake_train/ppo` is the customized Proximal Policy Optimization implementation.

Scripts used to train and evaluate the models are in `./src/battlesnake_train/scripts/` and are supposed to be copied into the REPL and run here (at `./`). For example, `./src/battlesnake_train/scripts/train_vit.py` contains the script to train the Vision Transformer model, and run simulation with steps rendered on the terminal with ANSI color.
