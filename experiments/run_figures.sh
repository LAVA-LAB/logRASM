#!/bin/bash
# Expected runtime (with GPU acceleration): 10-15 minutes

models=("LinearSystem" "LinearSystem --layout 1" "MyPendulum" "CollisionAvoidance --noise_partition_cells 24 --verify_batch_size 10000")
all_flags="--logger_prefix figures --eps_decrease 0.01 --ppo_max_policy_lipschitz 10 --hidden_layers 3 --expDecr_multiplier 10 --pretrain_method PPO_JAX --pretrain_total_steps 100000 --refine_threshold 250000000"
flags_mesh1="--mesh_loss 0.001 --mesh_loss_decrease_per_iter 0.9"
flags_mesh2="--mesh_loss 0.01 --mesh_loss_decrease_per_iter 0.8"

# Generate figures of selected RASMs
timeout 2000 python run.py --seed 1 --model ${models[0]} $all_flags $flags_mesh1 --probability_bound 0.999999 --exp_certificate;
timeout 2000 python run.py --seed 1 --model ${models[1]} $all_flags $flags_mesh1 --probability_bound 0.999999 --exp_certificate;
timeout 2000 python run.py --seed 1 --model ${models[2]} $all_flags $flags_mesh1 --probability_bound 0.999999 --exp_certificate;
timeout 2000 python run.py --seed 1 --model ${models[3]} $all_flags $flags_mesh2 --probability_bound 0.999999 --exp_certificate;