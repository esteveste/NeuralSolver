# Training the models
## Simple test
Use the random agent on the doorkey task to quickly test if everything is working (you need to create the doorkey dataset first).

From the root folder:
```bash
# Random
python ./deep-thinking/train_model.py problem.hyp.epochs=1 problem.name=curriculumv6_doorkey problem.model.width=64  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=20  problem.model.model=dt_net_random_out4  problem.test_data=33 problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_curriculumv6_doorkey_a0_dt_net_random_out4_w64_c2_t20_it30_ep1 problem.model.test_iterations.high=100

```

## Symetrical tasks

Prefix Sum
```bash
# NeuralSolver
python ./deep-thinking/train_model.py problem.model.width=100  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.hyp.train_mode=all_outputs problem.model.model=neuralsolver_1d problem.test_data=48 problem.hyp.alpha=0 problem/model=dt_net_1d problem=prefix_sums name=new_arches wandb_group=replicate_prefix_sums_allout_alpha_0_neuralsolver_1d_200_iters_w100 problem.model.test_iterations.high=200


# DT PL=0
python ./deep-thinking/train_model.py problem.test_data=48 problem.hyp.alpha=0.0 problem/model=dt_net_1d problem=prefix_sums name=mazes_ablation wandb_group=replicate_prefix_sums_alpha_0.0_dt_net_1d_200_iters problem.model.test_iterations.high=200
# DT PL=1
python ./deep-thinking/train_model.py problem.test_data=48 problem.hyp.alpha=1.0 problem/model=dt_net_1d problem=prefix_sums name=mazes_ablation wandb_group=replicate_prefix_sums_alpha_1.0_dt_net_1d_200_iters problem.model.test_iterations.high=200

# Feedforward
python ./deep-thinking/train_model.py problem.model.width=400  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.hyp.train_mode=all_outputs problem.model.model=feedforward_net_recall_1dproblem.test_data=48 problem.hyp.alpha=0 problem/model=ff_net_recall_1d problem=prefix_sums name=new_arches wandb_group=replicate_prefix_sums_allout_alpha_0_feedforward_net_recall_1d_200_iters_w400 problem.model.test_iterations.high=200

```

Maze
```bash

# NeuralSolver 
python ./deep-thinking/train_model.py problem.hyp.epochs=150 problem.model.width=32  problem.hyp.warmup_period=0 problem.hyp.clip=2  problem.model.model=neuralsolver_2d  problem.test_data=13 problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_mazes_a0_neuralsolver_2d_100_it_c2_w32_ep150 problem.model.test_iterations.high=100

# DT PL=0.01
python ./deep-thinking/train_model.py problem.hyp.clip=10 problem.hyp.epochs=150 problem.hyp.lr_schedule=[] problem.model.model=dt_net_recall_2d  problem.test_data=13 problem.hyp.alpha=0.01 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_more_ep150_clip10_mazes_alpha_0.01_dt_net_recall_2d_100_iters problem.model.test_iterations.high=100
# DT PL=0
python ./deep-thinking/train_model.py problem.hyp.clip=10 problem.hyp.epochs=150 problem.hyp.lr_schedule=[] problem.model.model=dt_net_recall_2d  problem.test_data=13 problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_more_ep150_clip10_mazes_alpha_0_dt_net_recall_2d_100_iters problem.model.test_iterations.high=100

#Feedforward
python ./deep-thinking/train_model.py problem.hyp.epochs=150 problem.model.width=128  problem.hyp.warmup_period=0 problem.hyp.clip=2  problem.model.model=feedforward_net_recall_2d  problem.test_data=13 problem.hyp.alpha=0 problem/model=ff_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_mazes_a0_feedforward_net_recall_2d_30_it_c2_w128_ep150 problem.model.test_iterations.high=30


```

Thin Maze
```bash
# NeuralSolver
python ./deep-thinking/train_model.py problem.hyp.epochs=150 problem.name=thin_maze problem.model.width=32  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=9  problem.model.model=neuralsolver_2d  problem.test_data=13 problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_thin_maze_a0_neuralsolver_2d_w32_c2_t9_it30_ep150 problem.model.test_iterations.high=100


# DT PL=0.01, change the hyp.alpha to test other values of PL loss
python ./deep-thinking/train_model.py problem.hyp.epochs=150 problem.name=thin_maze problem.model.max_iters=30 problem.train_data=9  problem.model.model=dt_net_recall_2d  problem.test_data=13 problem.hyp.alpha=0.01 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_thin_maze_a0.01_dt_net_recall_2d_t9_it30_ep150 problem.model.test_iterations.high=100


# Feedforward
python ./deep-thinking/train_model.py problem.hyp.epochs=150 problem.name=thin_maze problem.model.width=128  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=9  problem.model.model=feedforward_net_recall_2d  problem.test_data=13 problem.hyp.alpha=0 problem/model=ff_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_thin_maze_a0_feedforward_net_recall_2d_w128_c2_t9_it30_ep150 problem.model.test_iterations.high=30


```

chess

```bash

# NeuralSolver chess
python ./deep-thinking/train_model.py problem.model.width=128  problem.hyp.warmup_period=0 problem.hyp.clip=2  problem.model.model=neuralsolver_2d_nodrop  problem.test_data=700000 problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=chess name=chess_ablation wandb_group=replicate_chess_a0_neuralsolver_2d_nodrop_100_it_c2_w128 problem.model.test_iterations.high=100

# FeedForward chess
python ./deep-thinking/train_model.py problem.model.width=128  problem.hyp.warmup_period=0 problem.hyp.clip=2  problem.model.model=feedforward_net_recall_2d  problem.test_data=700000 problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=chess name=chess_ablation wandb_group=replicate_chess_a0_feedforward_net_recall_2d_it_c2_w128 problem.model.test_iterations.high=100

# DeepThink PL=0.5 chess
python ./deep-thinking/train_model.py problem.model.model=dt_net_recall_2d  problem.test_data=700000 problem.hyp.alpha=0.5 problem/model=dt_net_recall_2d problem=chess name=chess_ablation wandb_group=replicate_chess_alpha_0.5_dt_net_recall_2d_100_iters problem.model.test_iterations.high=100
# DeepThink PL=0 chess
python ./deep-thinking/train_model.py problem.model.model=dt_net_recall_2d  problem.test_data=700000 problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=chess name=chess_ablation wandb_group=replicate_chess_alpha_0_dt_net_recall_2d_100_iters problem.model.test_iterations.high=100

```


## Asymetrical tasks

1S-Maze
```bash
# NeuralSolver 
python ./deep-thinking/train_model.py problem.hyp.epochs=150 problem.name=curriculumv6_1s_maze problem.model.width=64  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=13  problem.model.model=dt_convlstm_noln_1l_sgal04_py03_2d_out4_maxpool  problem.test_data=23 problem.hyp.alpha=0 problem/model=dt_net_recall_2dproblem=mazes name=mazes_ablation wandb_group=replicate_curriculumv6_1s_maze_a0_dt_convlstm_noln_1l_sgal04_py03_2d_out4_maxpool_w64_c2_t13_it30_ep150 problem.model.test_iterations.high=100


# DT PL=0.01, change the hyp.alpha to test other values of PL loss
python ./deep-thinking/train_model.py problem.hyp.epochs=150 problem.name=curriculumv6_1s_maze problem.model.width=256  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=13  problem.model.model=dt_net_recall_2d_out4_maxpool_fixhead  problem.test_data=23 problem.hyp.alpha=0.01 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_curriculumv6_1s_maze_a0.01_dt_net_recall_2d_out4_maxpool_fixhead_w256_c2_t13_it30_ep150 problem.model.test_iterations.high=100

# Feedforward, 64 width works better than 256
python ./deep-thinking/train_model.py problem.hyp.epochs=150 problem.name=curriculumv6_1s_maze problem.model.width=64  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=13  problem.model.model=feedforward_net_recall_2d_out4_maxpool_fixhead  problem.test_data=23 problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_curriculumv6_1s_maze_a0_feedforward_net_recall_2d_out4_maxpool_fixhead_w64_c2_t13_it30_ep150 problem.model.test_iterations.high=100

# Random
python ./deep-thinking/train_model.py problem.hyp.epochs=1 problem.name=curriculumv6_1s_maze problem.model.width=64  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=13  problem.model.model=dt_net_random_out4  problem.test_data=23 problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_curriculumv6_1s_maze_a0_dt_net_random_out4_w64_c2_t13_it30_ep1 problem.model.test_iterations.high=100



```

GoTo
```bash
# NeuralSolver
python ./deep-thinking/train_model.py problem.name=curriculumv5_goto problem.model.width=64  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=20  problem.model.model=neuralsolver_1l_sgal04_py03_2d_out4_maxpool  problem.test_data=33 problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_curriculumv5_goto_a0_neuralsolver_1l_sgal04_py03_2d_out4_maxpool_w64_c2_t20_it30 problem.model.test_iterations.high=100

# DT
python ./deep-thinking/train_model.py problem.name=curriculumv5_goto problem.model.width=256  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=20  problem.model.model=dt_net_recall_2d_out4_maxpool_fixhead  problem.test_data=33 problem.hyp.alpha=0.1 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_curriculumv5_goto_a0.1_dt_net_recall_2d_out4_maxpool_fixhead_w256_c2_t20_it30 problem.model.test_iterations.high=100

# Feedforward
python ./deep-thinking/train_model.py problem.name=curriculumv5_goto problem.model.width=64  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=20  problem.model.model=feedforward_net_recall_2d_out4_maxpool_fixhead  problem.test_data=33 problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_curriculumv5_goto_a0_feedforward_net_recall_2d_out4_maxpool_fixhead_w64_c2_t20_it30 problem.model.test_iterations.high=100

# Random
python ./deep-thinking/train_model.py problem.name=curriculumv5_goto problem.model.width=64  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=20  problem.model.model=dt_net_random_out4  problem.test_data=33 problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_curriculumv5_goto_a0_dt_net_random_out4_w64_c2_t20_it30 problem.model.test_iterations.high=100




```

Pong
```bash
# NeuralSolver
python ./deep-thinking/train_model.py problem.hyp.lr=0.00025 problem.name=curriculumv5_simple_pong_square problem.model.width=64  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=20  problem.model.model=neuralsolver_1l_sgal04_py03_2d_out3_maxpool  problem.test_data=64 problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_curriculumv5_simple_pong_square_a0_neuralsolver_1l_sgal04_py03_2d_out3_maxpool_w64_c2_t20_it30_lr0.00025 problem.model.test_iterations.high=200


# DT
python ./deep-thinking/train_model.py problem.hyp.lr=0.00025 problem.name=curriculumv5_simple_pong_square problem.model.width=256  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=20  problem.model.model=dt_net_recall_2d_out3_maxpool_fixhead  problem.test_data=64 problem.hyp.alpha=0.01 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_curriculumv5_simple_pong_square_a0.01_dt_net_recall_2d_out3_maxpool_fixhead_w256_c2_t20_it30_lr0.00025 problem.model.test_iterations.high=200

# Feedforward
python ./deep-thinking/train_model.py problem.hyp.lr=0.00025 problem.name=curriculumv5_simple_pong_square problem.model.width=64  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=20  problem.model.model=feedforward_net_recall_2d_out3_maxpool_fixhead  problem.test_data=64 problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_curriculumv5_simple_pong_square_a0_feedforward_net_recall_2d_out3_maxpool_fixhead_w64_c2_t20_it30_lr0.00025 problem.model.test_iterations.high=200

# Random
python ./deep-thinking/train_model.py problem.hyp.epochs=1 problem.name=curriculumv5_simple_pong_square problem.model.width=64  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=20  problem.model.model=dt_net_random_out3  problem.test_data=64 problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_curriculumv5_simple_pong_square_a0_dt_net_random_out3_w64_c2_t20_it30 problem.model.test_iterations.high=200



```

Doorkey
```bash
# NeuralSolver
python ./deep-thinking/train_model.py problem.name=curriculumv6_doorkey problem.model.width=64  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=20  problem.model.model=neuralsolver_1l_sgal04_py03_2d_out4_maxpool  problem.test_data=33 problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_curriculumv6_doorkey_a0_neuralsolver_1l_sgal04_py03_2d_out4_maxpool_w64_c2_t20_it30 problem.model.test_iterations.high=100

# DT with PL=0.01, change the hyp.alpha to test other values of PL loss
python ./deep-thinking/train_model.py problem.name=curriculumv6_doorkey problem.model.width=256  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=20  problem.model.model=dt_net_recall_2d_out4_maxpool_fixhead  problem.test_data=33 problem.hyp.alpha=0.01 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablationwandb_group=replicate_curriculumv6_doorkey_a0.01_dt_net_recall_2d_out4_maxpool_fixhead_w256_c2_t20_it30 problem.model.test_iterations.high=100


# Feedforward
python ./deep-thinking/train_model.py problem.name=curriculumv6_doorkey problem.model.width=64  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=20  problem.model.model=feedforward_net_recall_2d_out4_maxpool_fixhead  problem.test_data=33 problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_curriculumv6_doorkey_a0_feedforward_net_recall_2d_out4_maxpool_fixhead_w64_c2_t20_it30 problem.model.test_iterations.high=100


# Random
python ./deep-thinking/train_model.py problem.hyp.epochs=1 problem.name=curriculumv6_doorkey problem.model.width=64  problem.hyp.warmup_period=0 problem.hyp.clip=2 problem.model.max_iters=30 problem.train_data=20  problem.model.model=dt_net_random_out4  problem.test_data=33 problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation wandb_group=replicate_curriculumv6_doorkey_a0_dt_net_random_out4_w64_c2_t20_it30_ep1 problem.model.test_iterations.high=100




```

## Ablation on Assymetrical Tasks
To perform the ablation tests in the article, you must substitute the problem.model.model value in the NeuralSolver python command for the desired ablation model. 

```bash
#example for average pool
problem.model.model=neuralsolver_1l_sgal04_py03_2d_out4_avgpool
```

Note that you should indicate the proper
Use AvgPool -> `neuralsolver_1l_sgal04_py03_2d_out4_avgpool`
Use 5L -> `neuralsolver_5l_sgal04_py03_2d_out4_maxpool`
No LSTM

For the curriculum learning ablation, remove the curriculum from the problem.name.

```bash
# change this
problem.name=curriculumv6_doorkey
# to this
problem.name=doorkey
```

## Sequential Decision Making Task - Doorkey


To evaluate the oracle run the following command:

```bash
python deep-thinking/eval_doorkey_algo_oracle.py
```

The evaluation of the other baselines are present in the train_model.py file.



## How to perform more evaluations
For simplicity I set the evaluation sizes for all problems in Table 1 to 3.
To set up more evaluation sizes, you can uncomment or change the eval function in the train_model.py file.

Note that in the evaluation we use the best_so_far_or_equal checkpoint to evaluate the all models, with the exception of DeepThink models that use the best_so_far as originally proposed in the paper.

