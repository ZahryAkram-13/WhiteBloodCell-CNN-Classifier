import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger

from paths import PLOTS_DIR, MODELS_DIR


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', save_path=None):
    """Plot confusion matrix with percentages"""
    plt.figure(figsize=(10, 8))
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage (%)'},
        square=True
    )
    
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    
    plt.show()





def plot_per_class_metrics(report, class_names, save_path=None):

    metrics = ['precision', 'recall', 'f1-score']
    data = {metric: [report[cls][metric] for cls in class_names] for metric in metrics}
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(13, 7))
    
    bars1 = ax.bar(x - width, data['precision'], width, label='Precision', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, data['recall'], width, label='Recall', color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + width, data['f1-score'], width, label='F1-Score', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.1])
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    
    plt.show()

sns.set_style("whitegrid")
sns.set_palette("husl")

def load_history(filename):
    data = np.load(filename)
    history = {
        'train_loss': data['train_loss'],
        'val_loss': data['val_loss'],
        'train_acc': data['train_acc'],
        'val_acc': data['val_acc']
    }
    return history

def plot_training_history(history, save_path=None, title="Training History"):
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    ax1 = axes[0]
    sns.lineplot(x=epochs, y=history['train_loss'], label='Training Loss', 
                 linewidth=2.5, marker='o', markersize=6, ax=ax1)
    sns.lineplot(x=epochs, y=history['val_loss'], label='Validation Loss', 
                 linewidth=2.5, marker='s', markersize=6, ax=ax1)
    
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    min_val_loss_idx = np.argmin(history['val_loss'])
    min_val_loss = history['val_loss'][min_val_loss_idx]
    ax1.annotate(f'Min Val Loss: {min_val_loss:.4f}',
                xy=(epochs[min_val_loss_idx], min_val_loss),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                fontsize=9, fontweight='bold')
    
    ax2 = axes[1]
    train_acc_pct = history['train_acc'] * 100
    val_acc_pct = history['val_acc'] * 100
    
    sns.lineplot(x=epochs, y=train_acc_pct, label='Training Accuracy', 
                 linewidth=2.5, marker='o', markersize=6, ax=ax2)
    sns.lineplot(x=epochs, y=val_acc_pct, label='Validation Accuracy', 
                 linewidth=2.5, marker='s', markersize=6, ax=ax2)
    
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    max_val_acc_idx = np.argmax(history['val_acc'])
    max_val_acc = val_acc_pct[max_val_acc_idx]
    ax2.annotate(f'Max Val Acc: {max_val_acc:.2f}%',
                xy=(epochs[max_val_acc_idx], max_val_acc),
                xytext=(10, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                fontsize=9, fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93]) 
    
    if save_path:
        path = PLOTS_DIR / save_path
        plt.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {path}")
    
    plt.show()
def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', save_path=None):
    plt.figure(figsize=(12, 12))
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage (%)'},
        square=True
    )
    
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(PLOTS_DIR / save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def print_summary(history):
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total epochs trained: {len(history['train_loss'])}")
    logger.info(f"\nLoss Metrics:")
    logger.info(f"  Final train loss:      {history['train_loss'][-1]:.4f}")
    logger.info(f"  Final val loss:        {history['val_loss'][-1]:.4f}")
    logger.info(f"  Best val loss:         {np.min(history['val_loss']):.4f} (epoch {np.argmin(history['val_loss'])+1})")
    logger.info(f"\nAccuracy Metrics:")
    logger.info(f"  Final train accuracy:  {history['train_acc'][-1]*100:.2f}%")
    logger.info(f"  Final val accuracy:    {history['val_acc'][-1]*100:.2f}%")
    logger.info(f"  Best val accuracy:     {np.max(history['val_acc'])*100:.2f}% (epoch {np.argmax(history['val_acc'])+1})")
    logger.info("="*60)

def compare_models_with_summary(history_files, save_prefix="models_comparison"):
   
    colors = sns.color_palette("husl", len(history_files))
    summary_data = []
    
    # ========== CREATE COMPARISON PLOTS ==========
    fig_plots, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax_loss = axes[0]
    ax_acc = axes[1]
    
    for idx, history_file in enumerate(history_files):
        data = np.load(MODELS_DIR / history_file)
        val_loss = data['val_loss']
        val_acc = data['val_acc'] * 100
        
        try:
            model_name = history_file.split('_')[2]
        except:
            model_name = history_file.split('.')[0]
        
        epochs = np.arange(1, len(val_loss) + 1)
        
        ax_loss.plot(epochs, val_loss, 
                    label=model_name, 
                    linewidth=2.5, 
                    marker='o', 
                    markersize=5,
                    color=colors[idx],
                    alpha=0.85)
        
        ax_acc.plot(epochs, val_acc, 
                   label=model_name, 
                   linewidth=2.5, 
                   marker='s', 
                   markersize=5,
                   color=colors[idx],
                   alpha=0.85)
        
        min_loss_idx = np.argmin(val_loss)
        min_loss = val_loss[min_loss_idx]
        ax_loss.scatter(epochs[min_loss_idx], min_loss, 
                       s=150, color=colors[idx], 
                       edgecolors='black', linewidth=2.5, zorder=5)
        
        max_acc_idx = np.argmax(val_acc)
        max_acc = val_acc[max_acc_idx]
        ax_acc.scatter(epochs[max_acc_idx], max_acc, 
                      s=150, color=colors[idx], 
                      edgecolors='black', linewidth=2.5, zorder=5)
        
        summary_data.append([
            model_name,
            f"{min_loss:.4f}",
            f"{epochs[min_loss_idx]}",
            f"{max_acc:.2f}%",
            f"{epochs[max_acc_idx]}",
            f"{val_loss[-1]:.4f}",
            f"{val_acc[-1]:.2f}%",
            f"{len(epochs)}"
        ])
    
    ax_loss.set_title('Validation Loss Comparison Across Models', 
                     fontsize=16, fontweight='bold', pad=15)
    ax_loss.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax_loss.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
    ax_loss.legend(fontsize=11, frameon=True, shadow=True, loc='best')
    ax_loss.grid(True, alpha=0.3, linestyle='--')
    
    ax_acc.set_title('Validation Accuracy Comparison Across Models', 
                    fontsize=16, fontweight='bold', pad=15)
    ax_acc.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax_acc.set_ylabel('Validation Accuracy (%)', fontsize=13, fontweight='bold')
    ax_acc.legend(fontsize=11, frameon=True, shadow=True, loc='best')
    ax_acc.grid(True, alpha=0.3, linestyle='--')
    
    fig_plots.suptitle('Model Performance Comparison - Validation Metrics', 
                      fontsize=18, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    plots_path = PLOTS_DIR / f"{save_prefix}_plots.png"
    fig_plots.savefig(plots_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Comparison plots saved to: {plots_path}")
    plt.close(fig_plots)
    
    # ========== CREATE SUMMARY TABLE ==========
    fig_table = plt.figure(figsize=(14, 6))
    ax_table = fig_table.add_subplot(111)
    ax_table.axis('off')
    
    headers = ['Model', 'Best Val Loss', 'Epoch', 'Best Val Acc', 'Epoch', 
               'Final Val Loss', 'Final Val Acc', 'Total Epochs']
    
    table = ax_table.table(cellText=summary_data, 
                          colLabels=headers,
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.13, 0.13, 0.08, 0.13, 0.08, 0.13, 0.13, 0.11])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white', size=12)
    
    for i in range(1, len(summary_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('#ffffff')
            table[(i, j)].set_edgecolor('#95a5a6')
            table[(i, j)].set_linewidth(1.5)
    
    losses = [float(row[1]) for row in summary_data]
    best_loss_idx = losses.index(min(losses))
    table[(best_loss_idx + 1, 1)].set_facecolor('#d5f4e6')
    table[(best_loss_idx + 1, 1)].set_text_props(weight='bold', color='#27ae60')
    
    accs = [float(row[3].strip('%')) for row in summary_data]
    best_acc_idx = accs.index(max(accs))
    table[(best_acc_idx + 1, 3)].set_facecolor('#d5f4e6')
    table[(best_acc_idx + 1, 3)].set_text_props(weight='bold', color='#27ae60')
    
    fig_table.suptitle('Model Performance Summary - Best and Final Metrics', 
                      fontsize=16, fontweight='bold', y=0.95)
    
    legend_text = "Green highlight = Best performance across all models"
    fig_table.text(0.5, 0.05, legend_text, 
                  ha='center', fontsize=11, style='italic', color='#27ae60')
    
    plt.tight_layout()
    
    table_path = PLOTS_DIR / f"{save_prefix}_summary_table.png"
    fig_table.savefig(table_path, dpi=300, bbox_inches='tight')
    logger.info(f"Summary table saved to: {table_path}")
    plt.close(fig_table)
    
    logger.info(f"\n Comparison complete! Generated 2 files:")
    logger.info(f"   1. {save_prefix}_plots.png (loss & accuracy comparison)")
    logger.info(f"   2. {save_prefix}_summary_table.png (performance metrics)")
    
    plt.show()

if __name__ == "__main__":
   


    HISTORY_FILES_CLASSIFIER = [
    
        "history__efficientnet_b0_64BATCH_2025-12-13-17:59:57.npz",
        "history__mobilenet_v3_small_64BATCH_2025-12-14-00:59:17.npz",
        "history__shufflenet_v2_64BATCH_2025-12-14-00:37:28.npz",
        "history__squeezenet_64BATCH_2025-12-13-21:19:26.npz",

        # "history_mobilenet_v2_64BATCH_2025-12-13-18:58:11.npz",

    ]
    HISTORY_FILES_NO_CLASSIFIER = [

        "training_history_efficientnet_b0_64BATCH_2025-12-13-13:05:46.npz",
        "training_history_mobilenet_v2_128BATCH_2025-12-13-12:33:43.npz",
        "training_history_resnet18_20251213_102606.npz",


        # "training_history_resnet18_only_20251213_103948.npz"
    ]

    # history_file = HISTORY_FILES_NO_CLASSIFIER[2]


    
    # model_name = history_file.split('_')[2]
    # history = load_history(MODELS_DIR / history_file)
    # save_path = f"plot_{model_name.upper()}_no_classifier.png"

    # print_summary(history)

    # plot_training_history(history, save_path=save_path, title=plot_title)

    
    
    # plot_title = f"Training History - {model_name.upper()}"
    


    logger.info("Creating model comparison plots...")
    
    compare_models_with_summary(
        HISTORY_FILES_NO_CLASSIFIER, 
        save_prefix="ALL_MODELS_NO_CLASSIFIER"
    )

    
