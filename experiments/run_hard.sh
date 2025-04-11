#!/bin/bash
# bash run_hard.sh | tee output/log_hard.txt

all_flags="--eps_decrease 0.01 --ppo_max_policy_lipschitz 10 --expDecr_multiplier 10 --pretrain_method PPO_JAX --refine_threshold 250000000 --epochs 100"

flags_triple="--model TripleIntegrator --logger_prefix TripleIntegrator --pretrain_total_steps 100000 --hidden_layers 3 --mesh_loss 0.005 --mesh_loss_decrease_per_iter 0.9 --mesh_verify_grid_init 0.04 --mesh_verify_grid_min 0.04 --noise_partition_cells 6 --max_refine_factor 4 --verify_batch_size 20000"
flags_planar="--model PlanarRobot --logger_prefix PlanarRobot --pretrain_total_steps 10000000 --hidden_layers 3 --mesh_loss 0.005 --mesh_loss_decrease_per_iter 0.9 --mesh_verify_grid_init 0.04 --mesh_verify_grid_min 0.04 --noise_partition_cells 12 --max_refine_factor 4 --verify_batch_size 20000"
flags_drone4D="--model Drone4D --layout 2 --logger_prefix Drone4D --pretrain_total_steps 1000000 --hidden_layers 2 --mesh_loss 0.01 --mesh_verify_grid_init 0.06 --mesh_verify_grid_min 0.06 --refine_threshold 50000000 --verify_threshold 600000000 --noise_partition_cells 6 --max_refine_factor 2 --verify_batch_size 30000"

# Triple integrator
for seed in {1..10};
do
  for p in 0.8 0.9 0.99 0.9999 0.999999
  do
    # Our method
    timeout 2000 python run.py --seed $seed $flags_triple $all_flags --probability_bound $p --exp_certificate;
    # logRASM only
    timeout 2000 python run.py --seed $seed $flags_triple $all_flags --probability_bound $p --exp_certificate --no-weighted --no-cplip;
  done
  for p in 0.8 0.9 0.99
  do
    # Lipschitz only
    timeout 2000 python run.py --seed $seed $flags_triple $all_flags --probability_bound $p --no-exp_certificate;
  done
done
#
for seed in {1..3};
do
  for p in 0.9999 0.999999
  do
    # Lipschitz only
    timeout 2000 python run.py --seed $seed $flags_triple $all_flags --probability_bound $p --no-exp_certificate;
  done
  for p in 0.8 0.9 0.99 0.9999 0.999999
  do
    # Baseline
    timeout 2000 python run.py --seed $seed $flags_triple $all_flags --probability_bound $p --no-exp_certificate --no-weighted --no-cplip;
done

# Planar robot
for seed in {1..10};
do
  for p in 0.8 0.9 0.99 0.9999 0.999999
  do
    # Our method
    timeout 2000 python run.py --seed $seed $flags_planar $all_flags --probability_bound $p --exp_certificate;
  done
done
#
for seed in {1..3};
do
  for p in 0.8 0.9 0.99 0.9999 0.999999
  do
    # logRASM only
    timeout 2000 python run.py --seed $seed $flags_planar $all_flags --probability_bound $p --exp_certificate --no-weighted --no-cplip;
    # Lipschitz only
    timeout 2000 python run.py --seed $seed $flags_planar $all_flags --probability_bound $p --no-exp_certificate;
    # Baseline
    timeout 2000 python run.py --seed $seed $flags_planar $all_flags --probability_bound $p --no-exp_certificate --no-weighted --no-cplip;
  done
done

# Drone4D
for seed in {1..10};
do
  for p in 0.8 0.9 0.99 0.9999 0.999999
  do
    # Our method
    timeout 2000 python run.py --seed $seed $flags_drone4D $all_flags --probability_bound $p --exp_certificate;
  done
done
#
for seed in {1..3};
do
  for p in 0.8 0.9 0.99 0.9999 0.999999
  do
    # logRASM only
    timeout 2000 python run.py --seed $seed $flags_drone4D $all_flags --probability_bound $p --exp_certificate --no-weighted --no-cplip;
    # Lipschitz only
    timeout 2000 python run.py --seed $seed $flags_drone4D $all_flags --probability_bound $p --no-exp_certificate;
    # Baseline
    timeout 2000 python run.py --seed $seed $flags_drone4D $all_flags --probability_bound $p --no-exp_certificate --no-weighted --no-cplip;
  done
done

# Generate table
python table_generator.py --folders hard

