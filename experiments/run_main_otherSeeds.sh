#!/bin/bash

models=("LinearSystem" "LinearSystem --layout 1" "MyPendulum" "CollisionAvoidance --noise_partition_cells 24 --verify_batch_size 10000")
all_flags="--logger_prefix main --eps_decrease 0.01 --ppo_max_policy_lipschitz 10 --hidden_layers 3 --expDecr_multiplier 10 --pretrain_method PPO_JAX --pretrain_total_steps 100000 --refine_threshold 250000000"
flags_mesh1="--mesh_loss 0.001 --mesh_loss_decrease_per_iter 0.8"
flags_mesh2="--mesh_loss 0.01 --mesh_loss_decrease_per_iter 0.8"

for seed in {3..10};
do
  # Linsys layout=0
  # ours
  for p in 0.9 0.99
  do
    timeout 2000 python run.py --seed $seed --model ${models[0]} $all_flags $flags_mesh1 --probability_bound $p --exp_certificate;
  done
  # no-lip
  for p in 0.9 0.99
  do
    timeout 2000 python run.py --seed $seed --model ${models[0]} $all_flags $flags_mesh1 --probability_bound $p --exp_certificate --no-weighted --no-cplip;
  done
  # no-exp
  for p in 0.9 0.99
  do
    timeout 2000 python run.py --seed $seed --model ${models[0]} $all_flags $flags_mesh1 --probability_bound $p --no-exp_certificate;
  done
  # base
  for p in 0.9 0.99
  do
    timeout 2000 python run.py --seed $seed --model ${models[0]} $all_flags $flags_mesh1 --probability_bound $p --no-exp_certificate --no-weighted --no-cplip;
  done
  
  # Linsys layout=1
  # ours
  for p in 0.9 0.99
  do
    timeout 2000 python run.py --seed $seed --model ${models[1]} $all_flags $flags_mesh1 --probability_bound $p --exp_certificate;
  done
  # no-lip
  for p in 0.9
  do
    timeout 2000 python run.py --seed $seed --model ${models[1]} $all_flags $flags_mesh1 --probability_bound $p --exp_certificate --no-weighted --no-cplip;
  done
  # no-exp
  # base
  
  # Pendulum
  # ours
  for p in 0.9 0.99
  do
    timeout 2000 python run.py --seed $seed --model ${models[2]} $all_flags $flags_mesh1 --probability_bound $p --exp_certificate;
  done
  # no-lip
  for p in 0.9 0.99
  do
    timeout 2000 python run.py --seed $seed --model ${models[2]} $all_flags $flags_mesh1 --probability_bound $p --exp_certificate --no-weighted --no-cplip;
  done
  # no-exp
  for p in 0.9
  do
    timeout 2000 python run.py --seed $seed --model ${models[2]} $all_flags $flags_mesh1 --probability_bound $p --no-exp_certificate;
  done
  # base
  
  # CollisionAvoidance
  # ours
  for p in 0.9 0.99
  do
    timeout 2000 python run.py --seed $seed --model ${models[3]} $all_flags $flags_mesh2 --probability_bound $p --exp_certificate;
  done
  # no-lip
  for p in 0.9 0.99
  do
    timeout 2000 python run.py --seed $seed --model ${models[3]} $all_flags $flags_mesh2 --probability_bound $p --exp_certificate --no-weighted --no-cplip;
  done
  # no-exp
  for p in 0.9 0.99
  do
    timeout 2000 python run.py --seed $seed --model ${models[3]} $all_flags $flags_mesh2 --probability_bound $p --no-exp_certificate;
  done
  # base
  
done
