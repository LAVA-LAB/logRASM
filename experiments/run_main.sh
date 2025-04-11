#!/bin/bash
# bash run_main.sh | tee output/log_main.txt

# This bash script runs the main experiments presented in the paper. First, the script runs all benchmarks for 3 seeds. Then, it only runs the remaining 7 seeds for benchmarks
# that did not lead to too many timeouts on the first 3 seeds.

models=("LinearSystem" "LinearSystem --layout 1" "MyPendulum" "CollisionAvoidance --noise_partition_cells 24 --verify_batch_size 10000")
all_flags="--logger_prefix main --eps_decrease 0.01 --ppo_max_policy_lipschitz 10 --hidden_layers 3 --expDecr_multiplier 10 --pretrain_method PPO_JAX --pretrain_total_steps 100000 --refine_threshold 250000000"
flags_mesh1="--mesh_loss 0.001 --mesh_loss_decrease_per_iter 0.8"
flags_mesh2="--mesh_loss 0.01 --mesh_loss_decrease_per_iter 0.8"

prob_bounds=(0.95 0.99 0.999 0.9999 0.999999)

# Run linear system, pendulum, and linear system (hard)
for x in 0 1 2
do
  for seed in 1 2 3
  do
    for p in "${prob_bounds[@]}"
    do
      timeout 2000 python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh1 --probability_bound $p --exp_certificate;
      timeout 2000 python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh1 --probability_bound $p --exp_certificate --no-weighted --no-cplip;
    done
    for p in "${prob_bounds[@]}"
    do
      timeout 2000 python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh1 --probability_bound $p --no-exp_certificate;
      timeout 2000 python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh1 --probability_bound $p --no-exp_certificate --no-weighted --no-cplip;
    done
  done
done
# Run collision avoidance
for x in 3
do
  for seed in 1 2 3
  do
    for p in "${prob_bounds[@]}"
    do
      timeout 2000 python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh2 --probability_bound $p --exp_certificate;
      timeout 2000 python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh2 --probability_bound $p --exp_certificate --no-weighted --no-cplip;
    done
    for p in "${prob_bounds[@]}"
    do
      timeout 2000 python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh2 --probability_bound $p --no-exp_certificate;
      timeout 2000 python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh2 --probability_bound $p --no-exp_certificate --no-weighted --no-cplip;
    done
  done
done

python main_genOtherSeeds.py;
bash experiments/run_main_otherSeeds.sh;

# Generate table
python table_generator.py --folders main