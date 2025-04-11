#!/bin/bash

models=("LinearSystem" "LinearSystem --layout 1" "MyPendulum" "CollisionAvoidance --noise_partition_cells 24 --verify_batch_size 10000")
all_flags="--logger_prefix main --eps_decrease 0.01 --ppo_max_policy_lipschitz 10 --hidden_layers 3 --expDecr_multiplier 10 --pretrain_method PPO_JAX --pretrain_total_steps 100000 --refine_threshold 250000000"
flags_mesh1="--mesh_loss 0.001 --mesh_loss_decrease_per_iter 0.8"
flags_mesh2="--mesh_loss 0.01 --mesh_loss_decrease_per_iter 0.8"

prob_bounds=(0.9 0.99)
TO=600

############################################################
### GENERATE FIGURES
############################################################

# The following script runs four individual benchmark instances, to generate the plots presented in the paper.
bash experiments/run_figures.sh

############################################################
### MAIN BENCHMARKS
############################################################

# Run linear system, pendulum, and linear system (hard)
for x in 0 1 2
do
  for seed in 1
  do
    for p in "${prob_bounds[@]}"
    do
      timeout $TO python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh1 --probability_bound $p --exp_certificate;
      timeout $TO python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh1 --probability_bound $p --exp_certificate --no-weighted --no-cplip;
    done
    for p in "${prob_bounds[@]}"
    do
      timeout $TO python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh1 --probability_bound $p --no-exp_certificate;
      timeout $TO python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh1 --probability_bound $p --no-exp_certificate --no-weighted --no-cplip;
    done
  done
done
# Run collision avoidance
for x in 3
do
  for seed in 1
  do
    for p in "${prob_bounds[@]}"
    do
      timeout $TO python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh2 --probability_bound $p --exp_certificate;
      timeout $TO python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh2 --probability_bound $p --exp_certificate --no-weighted --no-cplip;
    done
    for p in "${prob_bounds[@]}"
    do
      timeout $TO python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh2 --probability_bound $p --no-exp_certificate;
      timeout $TO python run.py --seed $seed --model ${models[x]} $all_flags $flags_mesh2 --probability_bound $p --no-exp_certificate --no-weighted --no-cplip;
    done
  done
done

# Generate table
python table_generator.py --folders main

############################################################
### STABLE BASELINES
############################################################

steps=(10000 100000 1000000)
algos=("TRPO" "SAC" "TQC" "A2C")
models=("LinearSystem" "LinearSystem --layout 1" "MyPendulum" "CollisionAvoidance --noise_partition_cells 24 --verify_batch_size 10000")
all_flags="--epochs 100 --eps_decrease 0.01 --hidden_layers 3 --refine_threshold 100000000"

flags_linsys=("--expDecr_multiplier 10 --mesh_loss 0.0005" "--expDecr_multiplier 0.1 --mesh_loss 0.001")
flags_linsys1=("--expDecr_multiplier 10 --mesh_loss 0.0005" "--expDecr_multiplier 0.1 --mesh_loss 0.001")
flags_pendulum=("--expDecr_multiplier 10 --mesh_loss 0.0005" "--expDecr_multiplier 0.1 --mesh_loss 0.001")
flags_collision=("--expDecr_multiplier 10 --mesh_loss 0.0005" "--expDecr_multiplier 0.1 --mesh_loss 0.001")

TO=600
for flags in 0;
do
  for i in 2 1 0;
  do
    for j in {0..3};
    do
      for seed in 1;
      do
          checkpoint="ckpt_pretrain_sb3/LinearSystem_layout=0_alg=${algos[j]}_layers=3_neurons=128_outfn=None_seed=${seed}_steps=${steps[i]}"
          timeout $TO python run.py --load_ckpt $checkpoint --logger_prefix linsys_sb --seed $seed --model ${models[0]} ${flags_linsys[flags]} $all_flags --probability_bound 0.999999 --exp_certificate;
      done
    done
  done

  for i in 2 1 0;
  do
    for j in {0..3};
    do
      for seed in 1;
      do
          checkpoint="ckpt_pretrain_sb3/LinearSystem_layout=1_alg=${algos[j]}_layers=3_neurons=128_outfn=None_seed=${seed}_steps=${steps[i]}"
          timeout $TO python run.py --load_ckpt $checkpoint --logger_prefix linsys1_sb --seed $seed --model ${models[1]} ${flags_linsys1[flags]} $all_flags --probability_bound 0.999999 --exp_certificate;
      done
    done
  done

  for i in 2 1 0;
  do
    for j in {0..3};
    do
      for seed in 1;
      do
          checkpoint="ckpt_pretrain_sb3/MyPendulum_layout=0_alg=${algos[j]}_layers=3_neurons=128_outfn=None_seed=${seed}_steps=${steps[i]}"
          timeout $TO python run.py --load_ckpt $checkpoint --logger_prefix pendulum_sb --seed $seed --model ${models[2]} ${flags_pendulum[flags]} $all_flags --probability_bound 0.999999 --exp_certificate;
      done
    done
  done

  for i in 2 1 0;
  do
    for j in {0..3};
    do
      for seed in 1;
      do
          checkpoint="ckpt_pretrain_sb3/CollisionAvoidance_layout=0_alg=${algos[j]}_layers=3_neurons=128_outfn=None_seed=${seed}_steps=${steps[i]}"
          timeout $TO python run.py --load_ckpt $checkpoint --logger_prefix collision_sb --seed $seed --model ${models[3]} ${flags_collision[flags]} $all_flags --probability_bound 0.999999 --exp_certificate;
      done
    done
  done
done

# Generate table
python table_generator.py --folders sb3

############################################################
### HARD EXPERIMENTS (ONLY TRIPLE INTEGRATOR)
############################################################

all_flags="--eps_decrease 0.01 --ppo_max_policy_lipschitz 10 --expDecr_multiplier 10 --pretrain_method PPO_JAX --refine_threshold 250000000 --epochs 100"

flags_triple="--model TripleIntegrator --logger_prefix TripleIntegrator --pretrain_total_steps 100000 --hidden_layers 3 --mesh_loss 0.005 --mesh_loss_decrease_per_iter 0.9 --mesh_verify_grid_init 0.04 --mesh_verify_grid_min 0.04 --noise_partition_cells 6 --max_refine_factor 4 --verify_batch_size 20000"

# Triple integrator
for seed in 1;
do
  for p in 0.8 0.9 0.99 0.9999 0.999999
  do
    # Our method
    timeout 2000 python run.py --seed $seed $flags_triple $all_flags --probability_bound $p --exp_certificate;
  done
done

# Generate table
python table_generator.py --folders hard