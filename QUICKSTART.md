# Quick Start Guide

**Git Commit Hash:** `d145e07` (include this in your report!)

**✅ ALL CRITICAL FIXES COMPLETE + IMPROVEMENTS ADDED**
- Phase 1: Critical bugs fixed (agent-swap, team-advantage, seeding)
- Phase 2: Improved reward shaping + pot handoffs collaboration metric
- Validation suite ready to test everything

## What You Have

A complete implementation of **PPO with Centralized Critic (CTDE)** for multi-agent Overcooked:

✅ **Algorithm:** Decentralized actors + centralized critic
✅ **Reward Shaping:** Fixed shaped rewards with agent-index swap correction
✅ **Hyperparameters:** Fixed across all layouts (as required)
✅ **Metrics:** Training curves, collaboration metrics, evaluation
✅ **Visualization:** Automated graph generation for report

## Installation (5 minutes)

### 1. Install PyTorch
```bash
# CPU version (recommended - faster for RL)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Install other dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify installation
```bash
python -c "import overcooked_ai_py; import torch; print('✓ All dependencies installed')"
```

### 4. **IMPORTANT: Run Validation First!**
```bash
python src/validate.py
```

This tests critical bug fixes (agent-swap, team-advantage, seeding).
**All tests must pass before training!** If any fail, something is broken.

Expected output:
```
✅ PASS: swap_caching
✅ PASS: team_advantages
✅ PASS: shaped_rewards
✅ PASS: determinism
🎉 ALL TESTS PASSED - Safe to start training!
```

## Training (Main Task)

### Start with Cramped Room (Easiest)
```bash
python src/train.py --layout cramped_room --episodes 50000
```

**What to expect:**
- Training time: ~4-6 hours on CPU
- Should reach ≥7 soups after ~20,000-30,000 episodes
- Progress logged every 100 episodes
- Checkpoints saved every 5,000 episodes

### Then Coordination Ring
```bash
python src/train.py --layout coordination_ring --episodes 100000
```

**What to expect:**
- Training time: ~8-12 hours
- More challenging - requires coordination
- Should reach ≥7 soups after ~50,000-70,000 episodes

### Finally Counter Circuit (Hardest)
```bash
python src/train.py --layout counter_circuit_o_1order --episodes 150000
```

**What to expect:**
- Training time: ~12-15 hours
- Most challenging layout
- May need full 150k episodes
- Target is still ≥7 soups

### Tips for Training
- **Run overnight:** Training takes hours, set it up before bed
- **Monitor progress:** Check `results/logs/<layout>_metrics.json`
- **Resume if crashed:** Use `--resume` flag to continue from checkpoint
- **CPU is fine:** You don't need a GPU (actually faster for PPO)

## Evaluation (Required for Report)

After training each layout:

```bash
# Evaluate single layout
python src/evaluate.py --layout cramped_room --num_episodes 100

# Or evaluate all at once
python src/evaluate.py --num_episodes 100
```

**Output:**
- Mean ± std soups delivered (need ≥7 for all layouts)
- Saved to `results/evaluation/<layout>_eval.json`
- Console shows if target is met

## Create Report Graphs (Required)

```bash
python src/visualize.py
```

**Creates:**
1. `figure1_training_curves.png` - Training progress all layouts (required)
2. `figure2_evaluation_performance.png` - Bar chart of eval results (required)
3. `figure_collab_<layout>.png` - Collaboration metrics (custom metrics)

## What Goes in Your Report

### Required Sections

**1. Algorithm Description**
- Explain PPO with centralized critic (CTDE)
- Why centralized critic helps coordination
- How agent-index swap ensures correct credit assignment

**2. Hyperparameters** (from `configs/hyperparameters.py`)
```
Learning rate: 3e-4
Gamma: 0.99
GAE lambda: 0.95
PPO clip: 0.2
Batch size: 2048
Mini-batch size: 256
PPO epochs: 10
```

**3. Reward Shaping**
```
+0.5: Onion placed in pot
+1.0: Cooking started with 3 onions
+1.5: Soup picked up
+2.0: Correct delivery
-0.5: Dropped soup
+20.0: Final delivery (sparse)
```
Include annealing schedule (70%-90% of training)

**4. Required Graphs**
- Training curves (soups vs episodes) for all 3 layouts
- Evaluation performance (100 episodes each)
- At least 2 custom metrics (we provide idle time and pot handoffs)

**5. Results Table**
| Layout | Mean Soups | Std | Target Met? |
|--------|-----------|-----|-------------|
| cramped_room | X.XX ± Y.YY | ✓/✗ |
| coordination_ring | X.XX ± Y.YY | ✓/✗ |
| counter_circuit_o_1order | X.XX ± Y.YY | ✓/✗ |

**6. Analysis**
- Why CTDE works for coordination
- How shaping speeds convergence
- What collaboration metrics show
- Why some layouts are harder

### Custom Metrics Explanation

**Idle Time:** Fraction of time agents spend on "stay" action
- Lower is better (shows efficiency)
- Should decrease as agents learn coordination

**Pot Handoffs:** How often agents work on same pot
- Higher is better (shows collaboration)
- Important for coordination_ring

## Troubleshooting

### Training is slow
- Normal! Each layout takes hours
- Run overnight or on multiple days
- Can reduce episodes but may not reach ≥7 soups

### Not reaching 7 soups
- Train longer (increase `--episodes`)
- Check logs to see if still improving
- cramped_room should definitely work (~25k episodes)
- coordination_ring may need 100k+
- counter_circuit is hardest, may need tweaking

### Import errors
```bash
# Make sure you're in the repo root
cd /Users/dheerkt/Desktop/repos/cs-rl

# And Python can find modules
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Agent-index swap warning
The code handles this automatically in `reward_shaping.py`.
Don't modify the swap logic or you'll break credit assignment!

## Timeline Estimate

**Week 1:** Setup + Train cramped_room (6-8 hours work)
- Day 1: Install dependencies, test notebook (1-2 hours)
- Day 2-3: Train cramped_room overnight (training time, minimal work)
- Day 4: Evaluate, check results (1 hour)

**Week 2:** Train other layouts (8-10 hours work)
- Day 1-3: Train coordination_ring (overnight training)
- Day 4-6: Train counter_circuit (overnight training)
- Day 7: Evaluate both, generate graphs (2 hours)

**Week 3:** Polish and handle issues (5-10 hours)
- If any layout didn't hit ≥7 soups, retrain with more episodes
- Fine-tune if needed (can adjust training duration only)

**Week 4:** Report writing (10-15 hours)
- Write 6-8 pages explaining algorithm, experiments, results
- Include all required graphs
- Analysis and discussion

**Total effort:** ~40-50 hours (realistic for an A)

## Key Implementation Details

### Centralized Critic
```python
# Each actor sees only its own observation
actor_0: obs[0] → action[0]
actor_1: obs[1] → action[1]

# Critic sees both (concatenated)
critic: [obs[0], obs[1]] → value
```

### Agent-Index Swap
Agents swap starting positions randomly each episode.
We detect and correct this in `reward_shaping.py`:
```python
if current_pos_0 != initial_pos_0:
    shaped_rewards = shaped_rewards[::-1]  # Swap!
```

### Why This Works
- **CTDE** solves non-stationarity from independent learning
- **Centralized critic** learns true value of joint actions
- **Reward shaping** speeds up learning sparse +20 reward
- **Agent-swap fix** prevents assigning credit to wrong agent

## Report Checklist

- [ ] Git commit hash included: `9a15c05`
- [ ] Algorithm explained (PPO + centralized critic)
- [ ] Hyperparameters listed (same across layouts)
- [ ] Reward shaping function described
- [ ] Training curves for all 3 layouts
- [ ] Evaluation results (100 episodes each)
- [ ] At least 2 custom metrics graphs
- [ ] Analysis of why approach works
- [ ] Discussion of collaboration
- [ ] All graphs have labels, axes, captions
- [ ] Code pushed to GitHub repo

## Questions?

Check:
1. README.md - Detailed project documentation
2. notebooks/getting_started.ipynb - Interactive testing
3. Comments in source code - Implementation details

## Good Luck!

This implementation follows best practices from recent MARL papers on Overcooked.
You have a solid foundation for an A grade - just train the agents and write a good report!
