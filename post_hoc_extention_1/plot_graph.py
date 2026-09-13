import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Data from the table
models = [
    'Mamba',
    'Mamba-2',
    'DenseMamba',
    'Hyena-150M',
    'MiniPLM-Mamba-130M',
    'Mamba-1.4B',
    'Mamba-2.8B',
]
properties = [
    'Simple Recall',
    'Instruction Following',
    'Long Context Recall',
    'Answering Query',
    'Chain Reasoning',
    'Basic Reasoning'
]

# Data structure: [property][model][without, with]
# Result set - 4
data = {
    'Simple Recall': {
        'Mamba': [94.0, 100.0],
        'Mamba-2': [91.0, 91.0],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [100.0, 100.0],
        'MiniPLM-Mamba-130M': [100.0, 100.0],
        'Mamba-1.4B': [99.0, 99.0],
        'Mamba-2.8B': [98.0, 100.0],
    },
    'Instruction Following': { #stress test
        'Mamba': [72.5, 90.0],
        'Mamba-2': [61.0, 68.0],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [100.0, 100.0],
        'MiniPLM-Mamba-130M': [50.0, 50.0],
        'Mamba-1.4B': [77.0, 80.5],
        'Mamba-2.8B': [58.0, 65.5],
    },
    'Long Context Recall': {
        'Mamba': [38.0, 50.5],
        'Mamba-2': [79.0, 86.0],
        'DenseMamba': [33.3, 33.3],
        'Hyena-150M': [100.0, 100.0],
        'MiniPLM-Mamba-130M': [41.5, 60.0],
        'Mamba-1.4B': [40.5, 45.0],
        'Mamba-2.8B': [68.0, 76.0],
    },
    'Answering Query': { # query dataset
        'Mamba': [40.0, 45.0],
        'Mamba-2': [100.0, 100.0],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [100.0, 100.0],
        'MiniPLM-Mamba-130M': [66.7, 66.7],
        'Mamba-1.4B': [35.0, 45.0],
        'Mamba-2.8B': [35.0, 45.0],
    },
    'Chain Reasoning': { #Two-Hop Reasoning
        'Mamba': [40.0, 55.0],
        'Mamba-2': [53.0, 67.0],
        'DenseMamba': [75.0, 100.0],
        'Hyena-150M': [64.0, 75.0],
        'MiniPLM-Mamba-130M': [26.5, 47.0],
        'Mamba-1.4B': [22.5, 26.0],
        'Mamba-2.8B': [45.0, 52.0],
    },
    'Basic Reasoning': { #combined reasoning
        'Mamba': [24.5, 30.7],
        'Mamba-2': [42.0, 41.0],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [47.0, 66.7],
        'MiniPLM-Mamba-130M': [48.0, 60.0],
        'Mamba-1.4B': [28.5, 32.0],
        'Mamba-2.8B': [31.0, 43.0],
    }
}

"""
# Result set - 3
data = {
    'Simple Recall': {
        'Mamba': [21.7, 32.0],
        'Mamba-2': [70.5, 80.0],
        'DenseMamba': [69.4, 100.0],
        'Hyena-150M': [100.0, 100.0],
        'MiniPLM-Mamba-130M': [66.1, 70.5]
    },
    'Instruction Following': { #stress test
        'Mamba': [50.0, 61.0],
        'Mamba-2': [53.1, 64.0],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [59.5, 59.5],
        'MiniPLM-Mamba-130M': [43.1, 43.1]
    },
    'Long Context Recall': {
        'Mamba': [29.0, 43.0],
        'Mamba-2': [52.0, 72.0],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [100.0, 100.0],
        'MiniPLM-Mamba-130M': [41.5, 60.0]
    },
    'Answering Query': { # query dataset
        'Mamba': [40.0, 66.7],
        'Mamba-2': [100.0, 100.0],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [100.0, 100.0],
        'MiniPLM-Mamba-130M': [66.7, 66.7]
    },
    'Chain Reasoning': { #Two-Hop Reasoning
        'Mamba': [46.7, 48.7],
        'Mamba-2': [32.5, 41.0],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [59.5, 75.0],
        'MiniPLM-Mamba-130M': [26.5, 47.0]
    },
    'Basic Reasoning': { #combined reasoning
        'Mamba': [66.7, 66.7],
        'Mamba-2': [36.0, 36.0],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [55.0, 66.7],
        'MiniPLM-Mamba-130M': [48.0, 60.0]
    }
}
"""

# Dataset → task mapping (used for AVG computation)
dataset_tasks = {
    'SQuAD': [
        'Simple Recall',
        'Instruction Following',
        'Long Context Recall'
    ],
    'MuSiQue': [
        'Chain Reasoning'
    ],
    'DROP': [
        'Basic Reasoning'
    ],
    'TriviaQA': [
        'Answering Query'
    ]
}

def compute_dataset_avg(data, models, tasks, with_steering=True):
    """
    Compute dataset-level AVG (W/O) or AVG (W)
    Averaged over all models and all tasks associated with the dataset.
    """
    idx = 1 if with_steering else 0
    values = [
        data[task][model][idx]
        for task in tasks
        for model in models
    ]
    return sum(values) / len(values)

def compute_task_avg(data, models, task, with_steering=True):
    """
    Compute task-level AVG (W/O) or AVG (W)
    Averaged over all models for a specific task/property.
    """
    idx = 1 if with_steering else 0
    values = [data[task][model][idx] for model in models]
    return sum(values) / len(values)

# Compute AVG (W/O) and AVG (W) for each dataset
dataset_avgs = {}

for dataset, tasks in dataset_tasks.items():
    avg_wo = compute_dataset_avg(data, models, tasks, with_steering=False)
    avg_w  = compute_dataset_avg(data, models, tasks, with_steering=True)
    dataset_avgs[dataset] = {
        'AVG (W/O)': avg_wo,
        'AVG (W)': avg_w
    }

# Compute task-level averages for Simple Recall and Instruction Following
task_avgs = {}
for task in ['Simple Recall', 'Instruction Following']:
    avg_wo = compute_task_avg(data, models, task, with_steering=False)
    avg_w = compute_task_avg(data, models, task, with_steering=True)
    task_avgs[task] = {
        'AVG (W/O)': avg_wo,
        'AVG (W)': avg_w
    }

# Print for verification / paper table
print("\nDataset-level averages:")
for dataset, avgs in dataset_avgs.items():
    print(f"{dataset}: AVG (W/O) = {avgs['AVG (W/O)']:.2f}, AVG (W) = {avgs['AVG (W)']:.2f}")

print("\nTask-level averages:")
for task, avgs in task_avgs.items():
    print(f"{task}: AVG (W/O) = {avgs['AVG (W/O)']:.2f}, AVG (W) = {avgs['AVG (W)']:.2f}")

'''
# Result set - 2
data = {
    'Simple Recall': {
        'Mamba': [97.0, 100.0],
        'Mamba-2': [93.0, 93.0],
        'DenseMamba': [44.0, 100.0],
        'Hyena-150M': [100.0, 100.0],
        'MiniPLM-Mamba-130M': [78.0, 100.0]
    },
    'Instruction Following': { #stress test
        'Mamba': [50.0, 67.0],
        'Mamba-2': [64.0, 64.0],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [100.0, 100.0],
        'MiniPLM-Mamba-130M': [53.0, 53.0]
    },
    'Long Context Recall': {
        'Mamba': [69.0, 100.0],
        'Mamba-2': [77.0, 100.0],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [100.0, 100.0],
        'MiniPLM-Mamba-130M': [80.0, 100.0]
    },
    'Answering Query': { # query dataset
        'Mamba': [40.0, 46.7],
        'Mamba-2': [100.0, 100.0],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [100.0, 100.0],
        'MiniPLM-Mamba-130M': [66.7, 66.7]
    },
    'Chain Reasoning': { #Two-Hop Reasoning
        'Mamba': [46.7, 46.7],
        'Mamba-2': [51.0, 76.0],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [64.0, 75.0],
        'MiniPLM-Mamba-130M': [47.0, 75.0]
    },
    'Basic Reasoning': { #combined reasoning
        'Mamba': [66.7, 66.7],
        'Mamba-2': [42.0, 48.0],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [44.0, 66.7],
        'MiniPLM-Mamba-130M': [60.0, 66.7]
    }
}
'''

"""
# Result set - 1
data = {
    'Needle in Haystack': {
        'Mamba': [80.0, 100.0],
        'Mamba-2': [100.0, 100.0],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [100.0, 100.0],
        'MiniPLM-Mamba-130M': [80.0, 100.0]
    },
    'Instruction Following': {
        'Mamba': [33.0, 67.0],
        'Mamba-2': [100.0, 100.0],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [100.0, 100.0],
        'MiniPLM-Mamba-130M': [66.7, 66.7]
    },
    'Long Context Recall': {
        'Mamba': [67.0, 100.0],
        'Mamba-2': [100.0, 100.0],
        'DenseMamba': [66.7, 100.0],
        'Hyena-150M': [100.0, 100.0],
        'MiniPLM-Mamba-130M': [100.0, 100.0]
    },
    'Answering Query': {
        'Mamba': [30.0, 36.7],
        'Mamba-2': [100.0, 100.0],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [100.0, 100.0],
        'MiniPLM-Mamba-130M': [66.7, 66.7]
    },
    'Chain Reasoning': {
        'Mamba': [66.7, 66.7],
        'Mamba-2': [75.0, 100.0],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [75.0, 100.0],
        'MiniPLM-Mamba-130M': [50.0, 75.0]
    },
    'Basic Reasoning': {
        'Mamba': [66.7, 66.7],
        'Mamba-2': [33.3, 66.7],
        'DenseMamba': [100.0, 100.0],
        'Hyena-150M': [33.3, 66.7],
        'MiniPLM-Mamba-130M': [66.7, 66.7]
    }
}
"""

import matplotlib.pyplot as plt
import numpy as np

def create_grouped_bar_chart():
    """Create a single grouped bar chart comparing W/O and W steering for each model and property."""
    fig, ax = plt.subplots(figsize=(40, 6.3))  # Wider for seven model pairs per task
    
    n_properties = len(properties)
    n_models = len(models)
    
    # Define colors for each model (same light color for both with/without)
    model_colors = {
        'Mamba': '#DDA0DD',           # Light purple
        'Mamba-2': '#B55239',         # Rust (swapped with Mamba-2.8B)
        'DenseMamba': '#87CEEB',      # Light blue
        'Hyena-150M': '#FF9999',      # Light red
        'MiniPLM-Mamba-130M': '#90EE90',  # Light green
        'Mamba-1.4B': '#F4A460',      # Sandy orange (distinct from Mamba-2.8B yellow)
        'Mamba-2.8B': '#FFFF00',      # Gold/Yellow (swapped with Mamba-2)
    }
    
    # Increased bar width for better spacing
    bar_width = 4  # Increased from 0.18

    # Reduced gaps for tighter grouping
    gap_within_model = 0.1  # Gap between without/with bars
    gap_between_models = 1  # Gap between different models
    gap_between_groups = 6  # Gap between property groups (increased for better separation)
    
    # Map properties to their corresponding datasets for AVG placement
    # For task-level averages (Simple Recall, Instruction Following), use the property name itself
    property_to_dataset = {
        'Simple Recall': 'Simple Recall',  # Task-level AVG
        'Instruction Following': 'Instruction Following',  # Task-level AVG
        'Long Context Recall': 'SQuAD',
        'Answering Query': 'TriviaQA',
        'Chain Reasoning': 'MuSiQue',
        'Basic Reasoning': 'DROP'
    }
    
    # Filter datasets to only include those where W/O and W values differ
    valid_datasets = {d: dataset_avgs[d] for d in dataset_tasks.keys() 
                      if abs(dataset_avgs[d]['AVG (W/O)'] - dataset_avgs[d]['AVG (W)']) > 0.01}
    
    # Add task-level averages to valid datasets
    for task, avgs in task_avgs.items():
        if abs(avgs['AVG (W/O)'] - avgs['AVG (W)']) > 0.01 or True:  # Always show task-level AVGs
            valid_datasets[task] = avgs
    
    # Calculate positions for each property group and insert AVG bars after corresponding properties
    x_positions = []
    avg_x_positions = []
    current_x = 0
    
    for prop_idx, prop in enumerate(properties):
        group_start = current_x
        model_positions = []
        
        for model_idx in range(n_models):
            # Position for "without" bar (light color)
            x_without = group_start + model_idx * (bar_width * 2 + gap_within_model + gap_between_models)
            # Position for "with" bar (dark color)
            x_with = x_without + bar_width + gap_within_model
            model_positions.append((x_without, x_with))
        
        x_positions.append((group_start, model_positions))
        # Calculate group width
        group_width = n_models * (bar_width * 2 + gap_within_model) + (n_models - 1) * gap_between_models
        
        # Check if this property should have an AVG bar after it
        if prop in property_to_dataset:
            dataset_or_task = property_to_dataset[prop]
            if dataset_or_task in valid_datasets:
                # Place AVG bar right after this property group with smaller gap
                avg_gap = 0.5  # Smaller gap between result bars and AVG bars
                avg_group_start = current_x + group_width + avg_gap
                x_without = avg_group_start
                x_with = x_without + bar_width + gap_within_model
                avg_x_positions.append((dataset_or_task, avg_group_start, (x_without, x_with)))
                # Move to next group (AVG bar width + full gap)
                avg_group_width = bar_width * 2 + gap_within_model
                current_x = avg_group_start + avg_group_width + gap_between_groups
            else:
                # No AVG bar, use normal gap
                current_x += group_width + gap_between_groups
        else:
            # No AVG bar, use normal gap
            current_x += group_width + gap_between_groups
    
    # Plot all bars and add labels immediately above each bar
    for prop_idx, prop in enumerate(properties):
        group_start, model_positions = x_positions[prop_idx]
        for model_idx, model in enumerate(models):
            x_without, x_with = model_positions[model_idx]
            without_val = data[prop][model][0]
            with_val = data[prop][model][1]
            
            # Get color for this model (same for both bars)
            bar_color = model_colors[model]
            
            # Plot bars with explicit alignment
            # Without steering: solid color
            bar1 = ax.bar(x_without, without_val, bar_width,
                         color=bar_color, alpha=0.9, edgecolor='black', 
                         linewidth=1, align='edge', hatch='')
            # With steering: same color with stripes
            bar2 = ax.bar(x_with, with_val, bar_width,
                         color=bar_color, alpha=0.9, edgecolor='black', 
                         linewidth=1, align='edge', hatch='///')
            
            # Calculate EXACT center of each bar
            # With align='edge', the bar starts at x position
            # So center is at x + bar_width/2
            bar_center_without = x_without + bar_width / 2
            bar_center_with = x_with + bar_width / 2
            
            # Add value labels - VERTICALLY inside each bar
            # For "without" bars - positioned at middle of bar height
            if without_val > 0:
                ax.text(bar_center_without, without_val / 2, f'{without_val:.1f}',
                       rotation=90, ha='center', va='center', 
                       fontsize=20, fontweight='bold', color='black')
            
            # For "with" bars - positioned at middle of bar height
            if with_val > 0:
                ax.text(bar_center_with, with_val / 2, f'{with_val:.1f}',
                       rotation=90, ha='center', va='center', 
                       fontsize=20, fontweight='bold', color='black')
    
    # Plot AVG bars for each dataset or task
    avg_color = 'white'  # White color for AVG bars
    for dataset_or_task, group_start, (x_without, x_with) in avg_x_positions:
        # Check if it's a task-level average or dataset-level average
        if dataset_or_task in task_avgs:
            avg_wo = task_avgs[dataset_or_task]['AVG (W/O)']
            avg_w = task_avgs[dataset_or_task]['AVG (W)']
        else:
            avg_wo = dataset_avgs[dataset_or_task]['AVG (W/O)']
            avg_w = dataset_avgs[dataset_or_task]['AVG (W)']
        
        # Plot AVG bars
        bar1 = ax.bar(x_without, avg_wo, bar_width,
                     color=avg_color, alpha=0.9, edgecolor='black', 
                     linewidth=1, align='edge', hatch='')
        bar2 = ax.bar(x_with, avg_w, bar_width,
                     color=avg_color, alpha=0.9, edgecolor='black', 
                     linewidth=1, align='edge', hatch='///')
        
        # Add value labels
        bar_center_without = x_without + bar_width / 2
        bar_center_with = x_with + bar_width / 2
        
        if avg_wo > 0:
            ax.text(bar_center_without, avg_wo / 2, f'{avg_wo:.1f}',
                   rotation=90, ha='center', va='center', 
                   fontsize=20, fontweight='bold', color='black')
        
        if avg_w > 0:
            ax.text(bar_center_with, avg_w / 2, f'{avg_w:.1f}',
                   rotation=90, ha='center', va='center', 
                   fontsize=20, fontweight='bold', color='black')
    
    # Mapping from property names to dataset names
    property_to_dataset_name = {
        'Simple Recall': 'SQuAD',
        'Instruction Following': 'IFEval',
        'Long Context Recall': 'RULER',
        'Answering Query': 'TriviaQA',
        'Chain Reasoning': 'MuSiQue',
        'Basic Reasoning': 'DROP'
    }
    
    # Format property names: single horizontal line (task — dataset)
    def wrap_property_name(name):
        dataset_name = property_to_dataset_name.get(name, '')
        if dataset_name:
            return f'{name} — {dataset_name}'
        return name
    
    # Build x-axis labels and centers in the correct order (interleaving properties and AVG bars)
    all_centers = []
    all_labels = []
    
    # Create a mapping from dataset to AVG position info
    avg_positions_dict = {dataset: (group_start, (x_without, x_with)) 
                          for dataset, group_start, (x_without, x_with) in avg_x_positions}
    
    for prop_idx, prop in enumerate(properties):
        # Calculate center for property group
        group_start, model_positions = x_positions[prop_idx]
        first_model_start = model_positions[0][0]
        last_model_end = model_positions[-1][1] + bar_width
        
        # Check if this property has an AVG bar after it
        if prop in property_to_dataset:
            dataset_or_task = property_to_dataset[prop]
            if dataset_or_task in avg_positions_dict:
                # Calculate center of combined group (property bars + AVG bars)
                avg_group_start, (avg_x_without, avg_x_with) = avg_positions_dict[dataset_or_task]
                avg_end = avg_x_with + bar_width
                # Center of entire group (from first property bar to end of AVG bar)
                combined_center = (first_model_start + avg_end) / 2
                all_centers.append(combined_center)
                all_labels.append(wrap_property_name(prop))
            else:
                # No AVG bar, use property group center
                group_center = (first_model_start + last_model_end) / 2
                all_centers.append(group_center)
                all_labels.append(wrap_property_name(prop))
        else:
            # No AVG bar, use property group center
            group_center = (first_model_start + last_model_end) / 2
            all_centers.append(group_center)
            all_labels.append(wrap_property_name(prop))
    
    ax.set_xticks(all_centers)
    ax.set_xticklabels(all_labels, rotation=0, ha='center', fontsize=24, fontweight='bold', color='black')
    
    # Add vertical lines to separate property groups
    for prop_idx, prop in enumerate(properties):
        if prop_idx > 0:  # Skip first line
            group_start, model_positions = x_positions[prop_idx]
            first_model_start = model_positions[0][0]
            ax.axvline(x=first_model_start - gap_between_groups/2, color='gray', 
                      linestyle='--', alpha=0.3, linewidth=1)
    
    # Add model labels as a separate legend
    from matplotlib.patches import Patch
    legend_elements = []
    for model in models:
        bar_color = model_colors[model]
        # With steering: same color with stripes
        legend_elements.append(Patch(facecolor=bar_color, edgecolor='black', 
                                   alpha=0.9, hatch='///', label=f'{model} (W)'))
    
    # Add AVG to legend
    legend_elements.append(Patch(facecolor=avg_color, edgecolor='black', 
                               alpha=0.9, hatch='///', label='AVG (W)'))
    
    n_leg = len(legend_elements)
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        fontsize=18,
        ncol=n_leg,
        bbox_to_anchor=(0, 1),
        framealpha=0.9,
        labelcolor='black',
    )
    
    # Calculate x-axis limits to remove gaps on left and right
    # First bar start position
    first_group_start, first_model_positions = x_positions[0]
    x_min = first_model_positions[0][0] - 1  # Small padding before first bar
    
    # Last bar end position (check if last property has AVG bar)
    last_prop = properties[-1]
    last_group_start, last_model_positions = x_positions[-1]
    last_model_end = last_model_positions[-1][1] + bar_width
    
    if last_prop in property_to_dataset:
        dataset_or_task = property_to_dataset[last_prop]
        if dataset_or_task in avg_positions_dict:
            # Last property has AVG bar, use AVG bar end
            avg_group_start, (avg_x_without, avg_x_with) = avg_positions_dict[dataset_or_task]
            x_max = avg_x_with + bar_width + 1  # Small padding after last bar
        else:
            x_max = last_model_end + 1
    else:
        x_max = last_model_end + 1
    
    ax.set_xlim(x_min, x_max)
    
    ax.set_ylabel('Performance (%)', fontsize=24, fontweight='bold', color='black')
    #ax.set_title('Performance Comparison Across Models With and Without Steering', 
    #             fontsize=24, fontweight='bold', pad=30, color='black')
    ax.set_ylim(0, 120)
    ax.set_yticks([40, 80, 120])
    ax.tick_params(axis='both', colors='black', labelsize=24)
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])  # Single-line x labels need less bottom margin
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Figure saved as 'performance_comparison.png'")
    plt.show()

# Also update the other functions to use the same color scheme
def create_improvement_chart():
    """Create a chart showing the improvement percentage from steering."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(properties))
    width = 0.15
    
    # Use the same color pairs from the bar chart
    color_pairs = {
        'Mamba': ('#90EE90', '#228B22'),           # Light green / Dark green
        'Mamba-2': ('#B55239', '#5C2414'),
        'DenseMamba': ('#87CEEB', '#4682B4'),      # Light blue / Dark blue
        'Hyena-150M': ('#FFB6C1', '#DC143C'),      # Light red / Dark red
        'MiniPLM-Mamba-130M': ('#DDA0DD', '#8B008B'),  # Light purple / Dark purple
        'Mamba-1.4B': ('#F4A460', '#CD853F'),
        'Mamba-2.8B': ('#FFFF00', '#FF8C00'),         # Yellow / Dark orange
    }
    
    improvement_data = {}
    for model in models:
        improvements = []
        for prop in properties:
            without = data[prop][model][0]
            with_val = data[prop][model][1]
            improvement = with_val - without
            improvements.append(improvement)
        improvement_data[model] = improvements
    
    for idx, model in enumerate(models):
        offset = (idx - len(models)/2) * width + width/2
        improvements = improvement_data[model]
        # Use the DARK color for improvement bars
        color_dark = color_pairs[model][1]
        bars = ax.bar(x + offset, improvements, width, 
                     label=model, color=color_dark, alpha=0.8, 
                     edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height != 0:
                va_pos = 'bottom' if height > 0 else 'top'
                y_pos = height + 1 if height > 0 else height - 1
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'+{height:.1f}%' if height > 0 else f'{height:.1f}%',
                       ha='center', va=va_pos, fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Improvement from Steering', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(properties, rotation=45, ha='right', fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Create legend using color patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_pairs[model][1], alpha=0.8, 
                           edgecolor='black', label=model) for model in models]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, ncol=2)
    
    plt.tight_layout()
    plt.savefig('improvement_chart.png', dpi=300, bbox_inches='tight')
    print("Improvement chart saved as 'improvement_chart.png'")
    plt.show()

def create_heatmap():
    """Create a heatmap showing the improvement from steering."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Prepare data for heatmaps
    without_matrix = np.array([[data[prop][model][0] for model in models] for prop in properties])
    with_matrix = np.array([[data[prop][model][1] for model in models] for prop in properties])
    improvement_matrix = with_matrix - without_matrix
    
    # Heatmap for Without Steering
    im1 = ax1.imshow(without_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=100)
    ax1.set_title('Performance Without Steering (%)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(np.arange(len(models)))
    ax1.set_yticks(np.arange(len(properties)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_yticklabels(properties)
    
    # Add text annotations
    for i in range(len(properties)):
        for j in range(len(models)):
            text = ax1.text(j, i, f'{without_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="white", fontweight='bold', fontsize=9)
    
    # Heatmap for With Steering
    im2 = ax2.imshow(with_matrix, cmap='Greens', aspect='auto', vmin=0, vmax=100)
    ax2.set_title('Performance With Steering (%)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(np.arange(len(models)))
    ax2.set_yticks(np.arange(len(properties)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_yticklabels(properties)
    
    # Add text annotations
    for i in range(len(properties)):
        for j in range(len(models)):
            text = ax2.text(j, i, f'{with_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="white", fontweight='bold', fontsize=9)
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1, label='Performance (%)')
    plt.colorbar(im2, ax=ax2, label='Performance (%)')
    
    #plt.suptitle('Performance Comparison: With vs Without Steering', 
    #             fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('performance_heatmap.png', dpi=300, bbox_inches='tight')
    print("Heatmap saved as 'performance_heatmap.png'")
    plt.show()

if __name__ == "__main__":
    print("Creating visualizations...")
    print("\n1. Grouped Bar Chart")
    create_grouped_bar_chart()
    
    print("\n2. Heatmap Comparison")
    create_heatmap()
    
    print("\n3. Improvement Chart")
    create_improvement_chart()
    
    print("\nAll visualizations created successfully!")

