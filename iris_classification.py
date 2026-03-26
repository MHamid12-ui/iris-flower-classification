# ============================================================
#   IRIS FLOWER CLASSIFICATION
#   Complete ML Training Code
#   Libraries: NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)

# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

print("=" * 60)
print("   IRIS FLOWER CLASSIFICATION")
print("=" * 60)

iris = load_iris()
df   = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({
    0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'
})

print(f"\n[STEP 1] Data Loaded")
print(f"  Shape   : {df.shape}")
print(f"  Classes : Setosa, Versicolor, Virginica")
print(f"  Samples : {len(df)} (50 per class)")
print(f"\n{df.describe().round(2)}")

# ============================================================
# STEP 2 — DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("   STEP 2 — DATA PREPROCESSING")
print("=" * 60)

# Missing values check
print(f"\n  Missing Values : {df.isnull().sum().sum()}")

# Features aur Target
feature_names = iris.feature_names
X = df[feature_names]
y = df['species']

print(f"  Features : {list(feature_names)}")
print(f"  Target   : species (0=Setosa, 1=Versicolor, 2=Virginica)")

# Class distribution
print(f"\n  Class Distribution:")
for name, count in df['species_name'].value_counts().items():
    print(f"    {name:<12} : {count} samples")

# ============================================================
# STEP 3 — TRAIN / TEST SPLIT & SCALING
# ============================================================

print("\n" + "=" * 60)
print("   STEP 3 — TRAIN/TEST SPLIT & SCALING")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\n  Train Set : {X_train.shape[0]} rows (80%)")
print(f"  Test Set  : {X_test.shape[0]} rows (20%)")
print(f"  Scaling   : StandardScaler applied")

# ============================================================
# STEP 4 — MODEL TRAINING (6 Algorithms)
# ============================================================

print("\n" + "=" * 60)
print("   STEP 4 — MODEL TRAINING")
print("=" * 60)

models = {
    'Logistic Regression':   LogisticRegression(max_iter=200, random_state=42),
    'K-Nearest Neighbors':   KNeighborsClassifier(n_neighbors=5),
    'Decision Tree':         DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':         RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':     GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
}

results = {}
print(f"\n  {'Model':<25} {'Accuracy':>10} {'CV Score':>10}")
print("  " + "-" * 48)

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred   = model.predict(X_test_s)
    acc      = accuracy_score(y_test, y_pred)
    cv_score = cross_val_score(model, X_train_s, y_train, cv=5).mean()
    results[name] = {
        'model': model, 'accuracy': acc,
        'cv_score': cv_score, 'pred': y_pred
    }
    print(f"  {name:<25} {acc*100:>9.2f}% {cv_score*100:>9.2f}%")

# Best model
best_name  = max(results, key=lambda k: results[k]['accuracy'])
best_acc   = results[best_name]['accuracy']
best_pred  = results[best_name]['pred']
best_model = results[best_name]['model']

print(f"\n  ★ Best Model : {best_name}")
print(f"  ★ Accuracy   : {best_acc*100:.2f}%")

# ============================================================
# STEP 5 — DETAILED EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("   STEP 5 — DETAILED EVALUATION")
print("=" * 60)

print(f"\n  Classification Report — {best_name}:\n")
print(classification_report(y_test, best_pred,
      target_names=['Setosa', 'Versicolor', 'Virginica']))

# ============================================================
# STEP 6 — VISUALIZATION DASHBOARD
# ============================================================

print("\n" + "=" * 60)
print("   STEP 6 — GENERATING DASHBOARD")
print("=" * 60)

plt.style.use('dark_background')

ACCENT  = '#00C8FF'
ACCENT2 = '#FF6B6B'
GREEN   = '#4ECDC4'
YELLOW  = '#FFD93D'
PURPLE  = '#C77DFF'
GRID_C  = '#2a2a3e'
BG      = '#0d0d1a'

SPECIES_COLORS = ['#00C8FF', '#4ECDC4', '#FFD93D']

fig = plt.figure(figsize=(20, 22), facecolor=BG)
fig.suptitle('Iris Flower Classification — ML Dashboard',
             fontsize=22, fontweight='bold', color='white', y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig,
                       hspace=0.45, wspace=0.35,
                       left=0.06, right=0.97,
                       top=0.94, bottom=0.04)

# ── Panel 1: Model Accuracy Comparison ──
ax1 = fig.add_subplot(gs[0, :2])
names = list(results.keys())
accs  = [results[n]['accuracy'] * 100 for n in names]
clrs  = [GREEN if n == best_name else ACCENT for n in names]
bars  = ax1.barh(names, accs, color=clrs, height=0.5, edgecolor='none')
for bar, val in zip(bars, accs):
    ax1.text(bar.get_width() - 1, bar.get_y() + bar.get_height() / 2,
             f'{val:.1f}%', va='center', ha='right',
             fontsize=11, fontweight='bold', color='black')
ax1.set_xlim(0, 110)
ax1.set_xlabel('Accuracy (%)', color='white')
ax1.set_title('Model Accuracy Comparison', color=ACCENT, fontsize=13, pad=10)
ax1.set_facecolor(GRID_C)
ax1.tick_params(colors='white')
ax1.spines[:].set_visible(False)

# ── Panel 2: CV Score vs Test Accuracy ──
ax2 = fig.add_subplot(gs[0, 2])
cv_scores = [results[n]['cv_score'] * 100 for n in names]
x_pos = np.arange(len(names))
short_names = ['LR', 'KNN', 'DT', 'RF', 'GB', 'SVM']
ax2.bar(x_pos - 0.2, accs, 0.35, label='Test Acc', color=ACCENT, alpha=0.85)
ax2.bar(x_pos + 0.2, cv_scores, 0.35, label='CV Score', color=ACCENT2, alpha=0.85)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(short_names, color='white', fontsize=9)
ax2.set_ylabel('Score (%)', color='white')
ax2.set_title('Test vs CV Score', color=ACCENT, fontsize=13)
ax2.set_facecolor(GRID_C)
ax2.tick_params(colors='white')
ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
ax2.spines[:].set_visible(False)

# ── Panel 3: Confusion Matrix ──
ax3 = fig.add_subplot(gs[1, :2])
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Setosa', 'Versicolor', 'Virginica'],
            yticklabels=['Setosa', 'Versicolor', 'Virginica'],
            ax=ax3, linewidths=1, linecolor='#1a1a2e',
            annot_kws={'size': 14, 'weight': 'bold'})
ax3.set_title(f'Confusion Matrix — {best_name}', color=ACCENT, fontsize=13)
ax3.set_xlabel('Predicted', color='white')
ax3.set_ylabel('Actual', color='white')
ax3.tick_params(colors='white')

# ── Panel 4: Class Distribution ──
ax4 = fig.add_subplot(gs[1, 2])
species_counts = df['species_name'].value_counts()
wedges, texts, autotexts = ax4.pie(
    species_counts.values,
    labels=species_counts.index,
    colors=SPECIES_COLORS,
    autopct='%1.0f%%',
    startangle=90,
    wedgeprops={'edgecolor': BG, 'linewidth': 2}
)
for text in texts:
    text.set_color('white')
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
ax4.set_title('Class Distribution', color=ACCENT, fontsize=13)

# ── Panel 5: Petal Length vs Petal Width ──
ax5 = fig.add_subplot(gs[2, :2])
species_list = ['Setosa', 'Versicolor', 'Virginica']
for i, (sp, col) in enumerate(zip(species_list, SPECIES_COLORS)):
    mask = df['species_name'] == sp
    ax5.scatter(df[mask]['petal length (cm)'],
                df[mask]['petal width (cm)'],
                c=col, label=sp, alpha=0.8, s=60, edgecolors='none')
ax5.set_xlabel('Petal Length (cm)', color='white')
ax5.set_ylabel('Petal Width (cm)', color='white')
ax5.set_title('Petal Length vs Petal Width', color=ACCENT, fontsize=13)
ax5.set_facecolor(GRID_C)
ax5.legend(facecolor='#1a1a2e', labelcolor='white')
ax5.tick_params(colors='white')
ax5.spines[:].set_visible(False)

# ── Panel 6: Sepal Length vs Sepal Width ──
ax6 = fig.add_subplot(gs[2, 2])
for i, (sp, col) in enumerate(zip(species_list, SPECIES_COLORS)):
    mask = df['species_name'] == sp
    ax6.scatter(df[mask]['sepal length (cm)'],
                df[mask]['sepal width (cm)'],
                c=col, label=sp, alpha=0.8, s=60, edgecolors='none')
ax6.set_xlabel('Sepal Length (cm)', color='white')
ax6.set_ylabel('Sepal Width (cm)', color='white')
ax6.set_title('Sepal Length vs Sepal Width', color=ACCENT, fontsize=13)
ax6.set_facecolor(GRID_C)
ax6.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
ax6.tick_params(colors='white')
ax6.spines[:].set_visible(False)

# ── Panel 7: Feature Importance ──
ax7 = fig.add_subplot(gs[3, :2])
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
else:
    importances = np.abs(best_model.coef_).mean(axis=0)

short_features = ['Sepal Len', 'Sepal Wid', 'Petal Len', 'Petal Wid']
fi_df = pd.DataFrame({'Feature': short_features, 'Importance': importances})
fi_df = fi_df.sort_values('Importance', ascending=True)
bar_colors = [YELLOW if v == fi_df['Importance'].max()
              else ACCENT for v in fi_df['Importance']]
ax7.barh(fi_df['Feature'], fi_df['Importance'],
         color=bar_colors, edgecolor='none')
ax7.set_title(f'Feature Importance — {best_name}', color=ACCENT, fontsize=13)
ax7.set_facecolor(GRID_C)
ax7.tick_params(colors='white')
ax7.spines[:].set_visible(False)

# ── Panel 8: Correlation Heatmap ──
ax8 = fig.add_subplot(gs[3, 2])
corr = df[list(feature_names)].corr()
sns.heatmap(corr, ax=ax8, cmap='coolwarm', annot=True, fmt='.2f',
            linewidths=0.5, linecolor='#1a1a2e',
            xticklabels=['SL', 'SW', 'PL', 'PW'],
            yticklabels=['SL', 'SW', 'PL', 'PW'],
            cbar_kws={'shrink': 0.7})
ax8.set_title('Feature Correlation\nHeatmap', color=ACCENT, fontsize=13)
ax8.tick_params(colors='white', labelsize=9)

plt.savefig('iris_classification.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
print("\n  Dashboard saved → iris_classification.png ✔")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("   FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"\n  Dataset  : {len(df)} samples, {len(feature_names)} features, 3 classes")
print(f"  Train    : {len(X_train)} | Test: {len(X_test)}")
print(f"\n  {'Model':<25} {'Accuracy':>10} {'CV Score':>10}")
print("  " + "-" * 48)
for name in sorted(results, key=lambda k: results[k]['accuracy'], reverse=True):
    star = " ★" if name == best_name else ""
    acc  = results[name]['accuracy'] * 100
    cv   = results[name]['cv_score'] * 100
    print(f"  {name:<25} {acc:>9.2f}% {cv:>9.2f}%{star}")

print(f"\n  Best Model : {best_name}")
print(f"  Accuracy   : {best_acc * 100:.2f}%")
print("\n" + "=" * 60)
print("   CLASSIFICATION COMPLETE ✔")
print("=" * 60)
