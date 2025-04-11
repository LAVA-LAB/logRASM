#!/bin/bash
# bash run_stablebaselines.sh | tee output/log_stablebaselines.txt

steps=(10000 100000 1000000)
algos=("TRPO" "SAC" "TQC" "A2C")
models=("LinearSystem" "LinearSystem --layout 1" "MyPendulum" "CollisionAvoidance --noise_partition_cells 24 --verify_batch_size 10000")
all_flags="--epochs 100 --eps_decrease 0.01 --hidden_layers 3 --refine_threshold 100000000"

flags_linsys=("--expDecr_multiplier 10 --mesh_loss 0.0005" "--expDecr_multiplier 0.1 --mesh_loss 0.001")
flags_linsys1=("--expDecr_multiplier 10 --mesh_loss 0.0005" "--expDecr_multiplier 0.1 --mesh_loss 0.001")
flags_pendulum=("--expDecr_multiplier 10 --mesh_loss 0.0005" "--expDecr_multiplier 0.1 --mesh_loss 0.001")
flags_collision=("--expDecr_multiplier 10 --mesh_loss 0.0005" "--expDecr_multiplier 0.1 --mesh_loss 0.001")

TO=1850

for flags in 0 1;
do
  for i in 2 1 0;
  do
    for j in {0..3};
    do
      for seed in {1..10};
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
      for seed in {1..10};
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
      for seed in {1..10};
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
      for seed in {1..10};
      do
          checkpoint="ckpt_pretrain_sb3/CollisionAvoidance_layout=0_alg=${algos[j]}_layers=3_neurons=128_outfn=None_seed=${seed}_steps=${steps[i]}"
          timeout $TO python run.py --load_ckpt $checkpoint --logger_prefix collision_sb --seed $seed --model ${models[3]} ${flags_collision[flags]} $all_flags --probability_bound 0.999999 --exp_certificate;
      done
    done
  done
done

# Generate table
python table_generator.py --folders sb3