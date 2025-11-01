#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample YOLO grafikleri oluşturur
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Output klasörü
output_dir = Path(__file__).parent
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Confusion Matrix
def create_confusion_matrix():
    classes = ['person', 'car', 'dog', 'cat']
    matrix = np.array([
        [850, 30, 10, 10],   # person
        [20, 780, 15, 5],    # car
        [25, 10, 720, 45],   # dog
        [15, 5, 55, 725]     # cat
    ])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='Blues')

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, matrix[i, j],
                          ha="center", va="center",
                          color="white" if matrix[i, j] > 500 else "black",
                          fontsize=14, fontweight='bold')

    ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('True', fontsize=12, fontweight='bold')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ confusion_matrix.png oluşturuldu")

# 2. F1 Curve
def create_f1_curve():
    confidence = np.linspace(0, 1, 100)

    # Sample F1 curves for different classes
    f1_all = 0.85 * np.exp(-((confidence - 0.5) ** 2) / 0.08)
    f1_person = 0.92 * np.exp(-((confidence - 0.45) ** 2) / 0.07)
    f1_car = 0.88 * np.exp(-((confidence - 0.5) ** 2) / 0.09)
    f1_dog = 0.78 * np.exp(-((confidence - 0.55) ** 2) / 0.08)
    f1_cat = 0.82 * np.exp(-((confidence - 0.52) ** 2) / 0.085)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(confidence, f1_all, 'b-', linewidth=3, label='all classes (0.85 at 0.447)')
    ax.plot(confidence, f1_person, 'g--', linewidth=2, label='person (0.92)')
    ax.plot(confidence, f1_car, 'r--', linewidth=2, label='car (0.88)')
    ax.plot(confidence, f1_dog, 'orange', linestyle='--', linewidth=2, label='dog (0.78)')
    ax.plot(confidence, f1_cat, 'purple', linestyle='--', linewidth=2, label='cat (0.82)')

    ax.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1', fontsize=12, fontweight='bold')
    ax.set_title('F1-Confidence Curve', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / 'F1_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ F1_curve.png oluşturuldu")

# 3. PR Curve (Precision-Recall)
def create_pr_curve():
    recall = np.linspace(0, 1, 100)

    # Sample PR curves
    pr_all = 0.9 * np.exp(-((recall - 0.5) ** 2) / 0.5)
    pr_person = 0.95 * np.exp(-((recall - 0.5) ** 2) / 0.5)
    pr_car = 0.91 * np.exp(-((recall - 0.5) ** 2) / 0.48)
    pr_dog = 0.83 * np.exp(-((recall - 0.5) ** 2) / 0.52)
    pr_cat = 0.87 * np.exp(-((recall - 0.5) ** 2) / 0.51)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(recall, pr_all, 'b-', linewidth=3, label='all classes (mAP@0.5: 0.847)')
    ax.plot(recall, pr_person, 'g--', linewidth=2, label='person (0.925)')
    ax.plot(recall, pr_car, 'r--', linewidth=2, label='car (0.889)')
    ax.plot(recall, pr_dog, 'orange', linestyle='--', linewidth=2, label='dog (0.781)')
    ax.plot(recall, pr_cat, 'purple', linestyle='--', linewidth=2, label='cat (0.832)')

    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / 'PR_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ PR_curve.png oluşturuldu")

# 4. Results (Training curves)
def create_results():
    epochs = np.arange(1, 101)

    # Training metrics
    train_loss = 0.08 + 0.4 * np.exp(-epochs / 15) + 0.02 * np.random.randn(100)
    val_loss = 0.10 + 0.45 * np.exp(-epochs / 18) + 0.025 * np.random.randn(100)

    precision = 0.85 - 0.3 * np.exp(-epochs / 12) + 0.01 * np.random.randn(100)
    recall = 0.82 - 0.28 * np.exp(-epochs / 13) + 0.01 * np.random.randn(100)

    map50 = 0.847 - 0.35 * np.exp(-epochs / 14) + 0.01 * np.random.randn(100)
    map50_95 = 0.621 - 0.25 * np.exp(-epochs / 14) + 0.01 * np.random.randn(100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs, train_loss, 'b-', label='train', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, 'r-', label='val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Box Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Precision & Recall
    axes[0, 1].plot(epochs, precision, 'g-', label='Precision', linewidth=2)
    axes[0, 1].plot(epochs, recall, 'orange', label='Recall', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision & Recall')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # mAP
    axes[1, 0].plot(epochs, map50, 'b-', label='mAP@0.5', linewidth=2)
    axes[1, 0].plot(epochs, map50_95, 'r-', label='mAP@0.5:0.95', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP')
    axes[1, 0].set_title('Mean Average Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Combined view
    axes[1, 1].plot(epochs, precision, 'g-', alpha=0.7, label='Precision', linewidth=2)
    axes[1, 1].plot(epochs, recall, 'orange', alpha=0.7, label='Recall', linewidth=2)
    axes[1, 1].plot(epochs, map50, 'b-', alpha=0.7, label='mAP@0.5', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Overall Metrics')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Training Results', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ results.png oluşturuldu")

if __name__ == '__main__':
    print("Sample YOLO grafikleri oluşturuluyor...")
    create_confusion_matrix()
    create_f1_curve()
    create_pr_curve()
    create_results()
    print("\n✅ Tüm grafikler oluşturuldu!")
    print(f"📁 Konum: {output_dir.absolute()}")
