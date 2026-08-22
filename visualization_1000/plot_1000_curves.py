"""
Plot 1000-item GRPO training curves + results comparison
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VIZ_DIR = os.path.dirname(os.path.abspath(__file__))

# ============ 1. Training curves from trainer_state ============
with open(os.path.join(VIZ_DIR, 'trainer_state_1000steps.json'), 'r', encoding='utf-8') as f:
    state = json.load(f)
logs = [e for e in state.get('log_history', []) if 'step' in e and 'reward' in e]

steps = [e['step'] for e in logs]
rewards = [e['reward'] for e in logs]
reward_stds = [e.get('reward_std', 0) for e in logs]
grad_norms = [e.get('grad_norm', 0) for e in logs]
kls = [e.get('kl', 0) for e in logs]
completion_lengths = [e.get('completion_length', 0) for e in logs]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('GRPO Training Curves - 3B + three_level + 1000 items (1000 steps)', fontsize=14)

ax = axes[0, 0]
ax.plot(steps, rewards, 'b-o', markersize=4, label='Mean Reward')
ax.fill_between(steps,
                [r - s for r, s in zip(rewards, reward_stds)],
                [r + s for r, s in zip(rewards, reward_stds)],
                alpha=0.2, color='blue', label='±1 std')
ax.axhline(0.1, color='orange', linestyle='--', alpha=0.7, label='0.1 partial level')
ax.set_xlabel('Training Step'); ax.set_ylabel('Reward')
ax.set_title('Reward Mean (with ±1 std)'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(steps, reward_stds, 'g-o', markersize=4)
ax.set_xlabel('Training Step'); ax.set_ylabel('Reward Std')
ax.set_title('Reward Std'); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(steps, grad_norms, 'r-o', markersize=4)
ax.set_xlabel('Training Step'); ax.set_ylabel('Gradient Norm')
ax.set_title('Gradient Norm'); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(steps, kls, 'm-o', markersize=4)
ax.set_xlabel('Training Step'); ax.set_ylabel('KL Divergence')
ax.set_title('KL Divergence'); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'training_curves_1000.png'), dpi=150, bbox_inches='tight')
plt.close()
print('[OK] training_curves_1000.png')

# ============ 2. Results comparison 100 vs 1000 ============
fig, ax = plt.subplots(figsize=(10, 6))

labels_100 = ['Baseline', 'ckpt-25', 'ckpt-50', 'ckpt-75', 'final']
vals_100 = [45.0, 50.0, 34.0, 36.0, 36.0]  # 100-item experiment (fixed extractor: baseline 45, ckpt25 50)

labels_1000 = ['Baseline', 'ckpt-25', 'ckpt-50', 'ckpt-75', 'final']
vals_1000 = [52.4, 53.5, 53.9, 53.0, 53.0]  # 1000-item experiment

x = range(len(labels_100))
ax.plot(x, vals_100, 'b-o', markersize=8, linewidth=2, label='100-item experiment')
ax.plot(x, vals_1000, 'r-s', markersize=8, linewidth=2, label='1000-item experiment')

for i, (v1, v2) in enumerate(zip(vals_100, vals_1000)):
    ax.annotate(f'{v1}%', (i, v1), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color='blue')
    ax.annotate(f'{v2}%', (i, v2), textcoords="offset points", xytext=(0, -18), ha='center', fontsize=10, color='red')

ax.set_xticks(list(x))
ax.set_xticklabels(labels_100)
ax.set_ylabel('Match Rate (%)')
ax.set_title('100 vs 1000 Item Experiment Comparison (three_level reward)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'compare_100_vs_1000.png'), dpi=150, bbox_inches='tight')
plt.close()
print('[OK] compare_100_vs_1000.png')

# ============ 3. Print metrics table ============
print('\n=== 1000-item Training Metrics ===')
print(f'{"Step":>5} {"Reward":>8} {"Std":>8} {"GradNorm":>10} {"KL":>10} {"Length":>8}')
for i, step in enumerate(steps):
    print(f'{step:>5} {rewards[i]:>8.3f} {reward_stds[i]:>8.3f} '
          f'{grad_norms[i]:>10.4f} {kls[i]:>10.6f} {completion_lengths[i]:>8.1f}')

# ============ 4. Results summary table ============
print('\n=== 1000-item Results ===')
results = {
    'Baseline': json.load(open(os.path.join(VIZ_DIR, '3b_1000_baseline_summary.json'))),
    'ckpt-25': json.load(open(os.path.join(VIZ_DIR, '3b_1000_ckpt25_summary.json'))),
    'ckpt-50': json.load(open(os.path.join(VIZ_DIR, '3b_1000_ckpt50_summary.json'))),
    'ckpt-75': json.load(open(os.path.join(VIZ_DIR, '3b_1000_ckpt75_summary.json'))),
    'final': json.load(open(os.path.join(VIZ_DIR, '3b_1000_final_summary.json'))),
}
print(f'{"Stage":<10} {"Parse":>8} {"Exec":>8} {"Match":>8}')
for name, s in results.items():
    print(f'{name:<10} {s["parse_success_rate"]:>7.1%} {s["prediction_execution_success_rate"]:>7.1%} {s["custom_execution_match_rate"]:>7.1%}')
