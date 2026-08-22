"""
Plot training curves WITH accumulated reward.
Usage: python3 plot_with_accumulated.py <trainer_state.json> <output_prefix>
- reward mean (per-step, with std)
- ACCUMULATED reward (sum of rewards so far) <- convergence indicator
- grad norm / KL / completion length
"""
import json, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 'trainer_state_checkpoint-75.json'
    prefix = sys.argv[2] if len(sys.argv) > 2 else 'training_accumulated'
    out_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(src):
        src = os.path.join(out_dir, src)

    with open(src, 'r', encoding='utf-8') as f:
        state = json.load(f)
    logs = [e for e in state.get('log_history', []) if 'step' in e and 'reward' in e]
    logs.sort(key=lambda e: e['step'])

    steps = [e['step'] for e in logs]
    rewards = [e['reward'] for e in logs]
    stds = [e.get('reward_std', 0) for e in logs]
    grads = [e.get('grad_norm', 0) for e in logs]
    kls = [e.get('kl', 0) for e in logs]
    lengths = [e.get('completion_length', 0) for e in logs]

    # accumulated reward
    cum = []
    acc = 0.0
    for r in rewards:
        acc += r
        cum.append(acc)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Curves with Accumulated Reward ({os.path.basename(src)})', fontsize=14)

    # 1. reward mean
    ax = axes[0, 0]
    ax.plot(steps, rewards, 'b-o', markersize=4, label='Reward Mean')
    ax.fill_between(steps, [r-s for r,s in zip(rewards,stds)],
                    [r+s for r,s in zip(rewards,stds)], alpha=0.2, color='blue', label='±1 std')
    ax.axhline(0.1, color='orange', ls='--', alpha=0.6, label='0.1 partial')
    ax.set_xlabel('Step'); ax.set_ylabel('Reward')
    ax.set_title('Reward Mean (per-step, oscillating)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 2. ACCUMULATED reward (key!)
    ax = axes[0, 1]
    ax.plot(steps, cum, 'r-o', markersize=4, linewidth=2, color='darkred', label='Accumulated Reward')
    ax.set_xlabel('Step'); ax.set_ylabel('Cumulative Reward')
    ax.set_title('ACCUMULATED REWARD (convergence indicator)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 3. reward std
    ax = axes[0, 2]
    ax.plot(steps, stds, 'g-o', markersize=4)
    ax.set_xlabel('Step'); ax.set_ylabel('Reward Std')
    ax.set_title('Reward Std (group discrimination)')
    ax.grid(alpha=0.3)

    # 4. grad norm
    ax = axes[1, 0]
    ax.plot(steps, grads, 'm-o', markersize=4)
    ax.set_xlabel('Step'); ax.set_ylabel('Grad Norm')
    ax.set_title('Gradient Norm')
    ax.grid(alpha=0.3)

    # 5. KL
    ax = axes[1, 1]
    ax.plot(steps, kls, 'c-o', markersize=4)
    ax.set_xlabel('Step'); ax.set_ylabel('KL')
    ax.set_title('KL Divergence')
    ax.grid(alpha=0.3)

    # 6. completion length
    ax = axes[1, 2]
    ax.plot(steps, lengths, 'y-o', markersize=4)
    ax.axhline(512, color='red', ls='--', label='Max 512')
    ax.set_xlabel('Step'); ax.set_ylabel('Length')
    ax.set_title('Completion Length')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{prefix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[OK] Saved: {out_path}')
    print(f'  Steps: {len(logs)}, Total accumulated reward: {cum[-1]:.3f}')
    print(f'  Final reward: {rewards[-1]:.3f}')

if __name__ == '__main__':
    main()
