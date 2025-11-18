# CUPID-Inspired Importance Weighting for Residual PPO

This implementation adds CUPID-inspired sample importance weighting to the residual PPO training loop.

## Overview

CUPID (Curating Performance-Influencing Demonstrations) uses influence functions to identify which training samples most affect policy performance. While the original CUPID is designed for offline imitation learning with diffusion policies, this implementation adapts the core principles to online PPO training.

## How It Works

### Core Concept

Instead of filtering/selecting demonstrations (offline), we use importance scores to weight minibatch sampling (online):

1. **Collect rollouts** for the current iteration
2. **Compute importance scores** periodically based on:
   - **Advantage magnitude**: High advantage = important for learning
   - **Value error**: High error = model uncertainty, needs more learning
   - **Success signal**: Samples from successful trajectories
3. **Weighted sampling**: Use importance scores to prioritize samples in minibatches
4. **Reuse scores**: Cache scores across all 50 epochs of training
5. **Periodic recomputation**: Recompute every N iterations as policy changes

### Mathematical Formula

For each sample, the importance score is computed as:

```
importance = α·|advantage| + β·|value_error| + γ·success_mask

where:
- α = advantage_weight (default: 0.4)
- β = value_error_weight (default: 0.3)
- γ = success_weight (default: 0.3)
- advantage = GAE advantage from PPO
- value_error = |returns - value_predictions|
- success_mask = 1 if trajectory succeeded, 0 otherwise
```

## Usage

### Option 1: Use the CUPID Config

```bash
# Train with CUPID enabled
python src/train/residual_ppo_cupid.py --config-name=base_residual_rl_cupid
```

### Option 2: Add CUPID Settings to Existing Config

Add these parameters to your config file:

```yaml
# CUPID settings
use_cupid: true
cupid_compute_interval: 5  # Recompute every 5 iterations
cupid_advantage_weight: 0.4  # Weight for advantage component
cupid_value_error_weight: 0.3  # Weight for value error component
cupid_success_weight: 0.3  # Weight for success component
```

### Option 3: Override from Command Line

```bash
python src/train/residual_ppo_cupid.py \
  --config-name=base_residual_rl \
  use_cupid=true \
  cupid_compute_interval=5 \
  cupid_advantage_weight=0.4 \
  cupid_value_error_weight=0.3 \
  cupid_success_weight=0.3
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_cupid` | `false` | Enable CUPID importance weighting |
| `cupid_compute_interval` | `5` | How often to recompute importance (iterations) |
| `cupid_advantage_weight` | `0.4` | Weight for advantage magnitude component |
| `cupid_value_error_weight` | `0.3` | Weight for value prediction error component |
| `cupid_success_weight` | `0.3` | Weight for success signal component |

## Monitoring

When CUPID is enabled, the following metrics are logged to Weights & Biases:

### CUPID Metrics

- `cupid/importance_min`: Minimum importance score
- `cupid/importance_max`: Maximum importance score
- `cupid/importance_mean`: Average importance score
- `cupid/importance_std`: Standard deviation of importance scores
- `cupid/top10_mean`: Average importance of top 10% samples
- `cupid/bottom10_mean`: Average importance of bottom 10% samples
- `histograms/importance`: Distribution of importance scores

### What to Look For

1. **Importance spread**: Check `cupid/top10_mean` vs `cupid/bottom10_mean`
   - Large difference = CUPID is identifying important samples
   - Small difference = All samples have similar importance

2. **Importance distribution**: Look at `histograms/importance`
   - Should see variation across samples
   - Peaks indicate clusters of similar-importance samples

3. **Correlation with success rate**: 
   - If `charts/success_rate` improves with CUPID, it's working
   - Compare runs with/without CUPID

## Implementation Details

### Data Flow

```
Iteration N:
├─ Collect rollouts (716,800 samples = 700 steps × 1024 envs)
├─ Compute advantages & returns
├─ [Every cupid_compute_interval iterations]
│  └─ Compute importance scores for all samples
└─ Train for 50 epochs:
   └─ For each minibatch:
      ├─ Sample based on importance (if CUPID enabled)
      └─ Perform PPO update
```

### Why This Works

1. **On-policy compatibility**: Importance is computed from the current iteration's data, which is already on-policy
2. **Efficiency**: Compute once per iteration, reuse for all 50 epochs
3. **Adaptive**: Recomputes periodically as policy changes
4. **Principle alignment**: Uses similar concepts to CUPID (advantage, error, success)

### Differences from Original CUPID

| Aspect | Original CUPID | This Implementation |
|--------|----------------|---------------------|
| **Setting** | Offline imitation learning | Online reinforcement learning |
| **Policy** | Diffusion policy | PPO (residual policy) |
| **Method** | TRAK influence functions | Heuristic importance scores |
| **Action** | Filter/select demonstrations | Weight minibatch sampling |
| **Data** | Fixed demonstration dataset | On-policy rollouts (discarded after use) |
| **Computation** | Gradient-based (expensive) | Direct computation (cheap) |

## Example Output

When CUPID is enabled, you'll see:

```
CUPID settings: use_cupid=True, compute_interval=5
Iteration: 5/1395
Computing CUPID importance scores at iteration 5...
CUPID computation took 0.15s
Importance stats: min=0.000001, max=0.000523, mean=0.000001
SR: 15.23%, SPS: 7543.21, STS: 8.45%, MSEL: 387.25
Policy update: 100%|██████████| 50/50 [00:09<00:00,  5.52it/s]
```

## Troubleshooting

### Issue: Importance scores all similar

**Symptoms**: `cupid/importance_std` is very small

**Solutions**:
- Increase `cupid_compute_interval` (wait longer between recomputations)
- Adjust component weights to emphasize differences
- Check if `success_rate` is too low or too high (need mix of success/failure)

### Issue: CUPID slows down training

**Symptoms**: Training time increases significantly

**Solutions**:
- Increase `cupid_compute_interval` (compute less frequently)
- The computation is ~0.1-0.2s, negligible compared to 50 epochs of training

### Issue: No performance improvement

**Symptoms**: Success rate same with/without CUPID

**Possible reasons**:
- Data quality already good (CUPID helps when data has varying quality)
- Need to tune component weights for your task
- Try longer training to see cumulative effects

## Comparison with Baseline

To compare CUPID with baseline PPO:

```bash
# Baseline (no CUPID)
python src/train/residual_ppo.py --config-name=base_residual_rl

# With CUPID
python src/train/residual_ppo_cupid.py --config-name=base_residual_rl_cupid
```

Compare:
- `charts/success_rate`: Final success rate
- `charts/mean_success_episode_length`: Efficiency
- Training curves in W&B

## Future Improvements

1. **Full TRAK Integration**: Implement proper gradient-based influence computation
2. **Adaptive Weights**: Learn α, β, γ weights automatically
3. **Trajectory-Level**: Compute importance at trajectory level instead of sample level
4. **Experience Replay**: Maintain a buffer of high-importance samples

## References

- [CUPID Paper](https://arxiv.org/abs/2506.19121): Original CUPID for imitation learning
- [TRAK](https://github.com/MadryLab/trak): Training Data Attribution framework
- [PPO](https://arxiv.org/abs/1707.06347): Proximal Policy Optimization

## Citation

If you use this implementation, please cite:

```bibtex
@article{agia2025cupid,
    title   = {CUPID: Curating Data your Robot Loves with Influence Functions},
    author  = {Agia, Christopher and Sinha, Rohan and Yang, Jingyun and Antonova, Rika and Pavone, Marco and Nishimura, Haruki and Itkina, Masha and Bohg, Jeannette},
    year    = 2025,
    journal = {arXiv preprint arXiv:2506.19121}
}
```


