import matplotlib.pyplot as plt
import numpy as np

# Set premium styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig_color = '#111827' # sleek dark background
text_color = '#F3F4F6'
grid_color = '#374151'
accent_colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']

def plot_16class():
    stages = ['Stage 1: RandomForest Baseline', 'Stage 2: Fine-tuned BERT (Epoch 2)', 'Stage 5: SVC Optimized']
    accuracies = [21.0, 30.18, 21.0]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=fig_color)
    ax.set_facecolor(fig_color)

    bars = ax.bar(stages, accuracies, color=['#EF4444', '#3B82F6', '#F59E0B'], width=0.5, edgecolor=text_color, linewidth=0.5)

    # Styling text
    ax.set_title('MBTI 16-Class Classification Accuracy Comparison', fontsize=16, fontweight='bold', color=text_color, pad=20)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color=text_color)
    ax.set_ylim(0, 45)
    
    # Tick styling
    ax.tick_params(colors=text_color, labelsize=11)
    ax.grid(color=grid_color, linestyle='--', linewidth=0.5)

    # Adding value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color=text_color)

    plt.tight_layout()
    plt.savefig('/workspace/mbti_16class_comparison.png', dpi=300, facecolor=fig_color)
    plt.close()
    print("16-class comparison plot saved successfully!")

def plot_traits():
    traits = ['I/E (Introversion/Extraversion)', 'N/S (Intuition/Sensing)', 'F/T (Feeling/Thinking)', 'P/J (Perceiving/Judging)']
    accuracies = [77.0, 86.0, 61.0, 61.0]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=fig_color)
    ax.set_facecolor(fig_color)

    bars = ax.bar(traits, accuracies, color='#8B5CF6', width=0.5, edgecolor=text_color, linewidth=0.5)

    # Styling text
    ax.set_title('MBTI Binary Trait Classification Accuracy', fontsize=16, fontweight='bold', color=text_color, pad=20)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color=text_color)
    ax.set_ylim(0, 100)
    
    # Tick styling
    ax.tick_params(colors=text_color, labelsize=11)
    ax.grid(color=grid_color, linestyle='--', linewidth=0.5)

    # Adding value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color=text_color)

    plt.tight_layout()
    plt.savefig('/workspace/mbti_traits_comparison.png', dpi=300, facecolor=fig_color)
    plt.close()
    print("Traits comparison plot saved successfully!")

if __name__ == '__main__':
    plot_16class()
    plot_traits()
