"""
Visualization utilities for creating report graphs
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def load_training_logs(log_dir, layout_name):
    """Load training logs from JSON file"""
    log_path = os.path.join(log_dir, f'{layout_name}_metrics.json')

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    with open(log_path, 'r') as f:
        data = json.load(f)

    return data


def moving_average(data, window=100):
    """Compute moving average"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_combined_training_curves(log_dir, layouts, save_path):
    """
    Plot training curves for all layouts on one graph

    Args:
        log_dir: Directory containing log files
        layouts: List of layout names
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    window = 100

    for i, layout in enumerate(layouts):
        try:
            data = load_training_logs(log_dir, layout)
            soups = data['episode_soups']

            # Plot raw data (transparent)
            episodes = np.arange(1, len(soups) + 1)
            plt.plot(episodes, soups, alpha=0.1, color=colors[i])

            # Plot moving average
            if len(soups) > window:
                smoothed = moving_average(soups, window)
                smoothed_episodes = np.arange(window, len(soups) + 1)
                plt.plot(smoothed_episodes, smoothed, label=layout.replace('_', ' ').title(),
                        color=colors[i], linewidth=2)

        except FileNotFoundError:
            print(f"Warning: Could not find logs for {layout}")

    # Target line
    plt.axhline(y=7, color='red', linestyle='--', linewidth=2, label='Target (7 soups)', alpha=0.7)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Soups Delivered', fontsize=12)
    plt.title('Training Progress Across All Layouts', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined training curves to {save_path}")
    plt.close()


def plot_collaboration_metrics(log_dir, layout_name, save_path):
    """
    Plot collaboration metrics (pot handoffs, idle time)

    Args:
        log_dir: Directory containing log files
        layout_name: Layout name
        save_path: Path to save figure
    """
    data = load_training_logs(log_dir, layout_name)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Idle time
    if 'idle_times_agent0' in data and len(data['idle_times_agent0']) > 0:
        window = 100
        idle0 = np.array(data['idle_times_agent0'])
        idle1 = np.array(data['idle_times_agent1'])

        episodes = np.arange(1, len(idle0) + 1)

        # Raw data
        axes[0].plot(episodes, idle0, alpha=0.2, color='#1f77b4')
        axes[0].plot(episodes, idle1, alpha=0.2, color='#ff7f0e')

        # Smoothed
        if len(idle0) > window:
            smooth0 = moving_average(idle0, window)
            smooth1 = moving_average(idle1, window)
            smooth_episodes = np.arange(window, len(idle0) + 1)

            axes[0].plot(smooth_episodes, smooth0, label='Agent 0', color='#1f77b4', linewidth=2)
            axes[0].plot(smooth_episodes, smooth1, label='Agent 1', color='#ff7f0e', linewidth=2)

        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Idle Time (fraction)')
        axes[0].set_title('Agent Idle Time Over Training')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Pot handoffs (if available)
    if 'pot_handoffs' in data and len(data['pot_handoffs']) > 0:
        window = 100
        handoffs = np.array(data['pot_handoffs'])
        episodes = np.arange(1, len(handoffs) + 1)

        axes[1].plot(episodes, handoffs, alpha=0.2, color='#2ca02c')

        if len(handoffs) > window:
            smoothed = moving_average(handoffs, window)
            smooth_episodes = np.arange(window, len(handoffs) + 1)
            axes[1].plot(smooth_episodes, smoothed, color='#2ca02c', linewidth=2)

        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Pot Handoffs per Episode')
        axes[1].set_title('Coordination: Pot Handoffs Over Training')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No pot handoff data available',
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title('Coordination Metric')

    plt.suptitle(f'{layout_name.replace("_", " ").title()} - Collaboration Metrics',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved collaboration metrics to {save_path}")
    plt.close()


def create_all_report_graphs(log_dir, eval_dir, graph_dir):
    """
    Create all graphs required for the report

    Args:
        log_dir: Directory containing training logs
        eval_dir: Directory containing evaluation results
        graph_dir: Directory to save graphs
    """
    layouts = ['cramped_room', 'coordination_ring', 'counter_circuit_o_1order']

    print("Creating report graphs...")

    # 1. Combined training curves (required)
    print("1. Creating combined training curves...")
    plot_combined_training_curves(
        log_dir, layouts,
        os.path.join(graph_dir, 'figure1_training_curves.png')
    )

    # 2. Individual collaboration metrics (custom metrics)
    print("2. Creating collaboration metric graphs...")
    for layout in layouts:
        try:
            plot_collaboration_metrics(
                log_dir, layout,
                os.path.join(graph_dir, f'figure_collab_{layout}.png')
            )
        except FileNotFoundError:
            print(f"   Warning: No logs found for {layout}")

    # 3. Evaluation results bar chart
    print("3. Creating evaluation results chart...")
    try:
        plot_evaluation_bar_chart(eval_dir, layouts, graph_dir)
    except Exception as e:
        print(f"   Warning: Could not create evaluation chart: {e}")

    print(f"\nAll graphs saved to {graph_dir}")


def plot_evaluation_bar_chart(eval_dir, layouts, graph_dir):
    """
    Create bar chart of evaluation results

    Args:
        eval_dir: Directory containing evaluation JSON files
        layouts: List of layout names
        graph_dir: Directory to save graph
    """
    means = []
    stds = []
    layout_names = []

    for layout in layouts:
        eval_path = os.path.join(eval_dir, f'{layout}_eval.json')

        if not os.path.exists(eval_path):
            print(f"Warning: Evaluation file not found: {eval_path}")
            continue

        with open(eval_path, 'r') as f:
            data = json.load(f)

        means.append(data['mean_soups'])
        stds.append(data['std_soups'])
        layout_names.append(layout.replace('_', ' ').title())

    if not means:
        print("No evaluation data found")
        return

    # Create bar chart
    plt.figure(figsize=(10, 6))

    x = np.arange(len(layout_names))
    bars = plt.bar(x, means, yerr=stds, alpha=0.7, capsize=5,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(layout_names)])

    # Target line
    plt.axhline(y=7, color='red', linestyle='--', linewidth=2, label='Target (7 soups)', alpha=0.7)

    # Labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + std + 0.3, f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xlabel('Layout', fontsize=12)
    plt.ylabel('Soups Delivered (mean ± std)', fontsize=12)
    plt.title('Evaluation Performance Across Layouts (100 episodes each)',
             fontsize=14, fontweight='bold')
    plt.xticks(x, layout_names)
    plt.legend(fontsize=10)
    plt.grid(True, axis='y', alpha=0.3)
    plt.ylim(0, max(means) + max(stds) + 2)
    plt.tight_layout()

    save_path = os.path.join(graph_dir, 'figure2_evaluation_performance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved evaluation bar chart to {save_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Create visualization graphs for report')
    parser.add_argument('--log_dir', type=str, default='results/logs',
                       help='Directory containing training logs')
    parser.add_argument('--eval_dir', type=str, default='results/evaluation',
                       help='Directory containing evaluation results')
    parser.add_argument('--graph_dir', type=str, default='results/graphs',
                       help='Directory to save graphs')

    args = parser.parse_args()

    create_all_report_graphs(args.log_dir, args.eval_dir, args.graph_dir)


if __name__ == '__main__':
    main()
