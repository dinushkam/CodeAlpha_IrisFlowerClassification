import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. LOAD THE CSV FILE

df = pd.read_csv('Iris.csv')

print("=" * 50)
print("DATASET — FIRST LOOK")
print("=" * 50)
print(df.head())          
print(f"\nShape: {df.shape}") 
print(f"\nColumn names:\n{list(df.columns)}")
print(f"\nSpecies counts:\n{df['Species'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum()}")  
print(f"\nStatistical summary:")
print(df.describe())

# 2. PREPARE THE DATA
df = df.drop('Id', axis=1)
X = df.drop('Species', axis=1).values   
y_text = df['Species'].values           

le = LabelEncoder()
y = le.fit_transform(y_text)
class_names = le.classes_             

print("\n" + "=" * 50)
print("LABEL ENCODING")
print("=" * 50)
for i, name in enumerate(class_names):
    print(f"  {name}  →  {i}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y       
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   
X_test_s  = scaler.transform(X_test)        
X_all_s   = scaler.transform(X)             

print(f"\nTrain samples: {len(X_train)}")
print(f"Test  samples: {len(X_test)}")
# 3. TRAIN THREE MODELS

models = {
    "KNN (k=5)"      : KNeighborsClassifier(n_neighbors=5),
    "Decision Tree"  : DecisionTreeClassifier(random_state=42, max_depth=4),
    "Random Forest"  : RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
print("\n" + "=" * 50)
print("MODEL TRAINING & EVALUATION")
print("=" * 50)

for name, model in models.items():
   
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    
    acc = accuracy_score(y_test, y_pred)
    cv  = cross_val_score(model, X_all_s, y, cv=5)

    results[name] = {
        "model"   : model,
        "accuracy": acc,
        "cv_mean" : cv.mean(),
        "cv_std"  : cv.std(),
        "y_pred"  : y_pred,
    }

    print(f"\n── {name} ──")
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(f"  CV  Accuracy  : {cv.mean() * 100:.2f}% ± {cv.std() * 100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

# 4. VISUALIZATIONS

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Iris Flower Classification — Model Comparison", fontsize=14, fontweight='bold')

ax = axes[0, 0]
names    = list(results.keys())
accs     = [r["accuracy"] * 100 for r in results.values()]
cv_means = [r["cv_mean"]  * 100 for r in results.values()]
cv_stds  = [r["cv_std"]   * 100 for r in results.values()]
x = np.arange(len(names))
w = 0.35
ax.bar(x - w/2, accs,     w, label="Test Accuracy",       color="#378ADD")
ax.bar(x + w/2, cv_means, w, yerr=cv_stds, capsize=4,
       label="CV Accuracy ±1 std", color="#1D9E75")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Accuracy Comparison")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=10, ha="right")
ax.set_ylim(80, 102)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

ax = axes[0, 2]
rf_model    = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
feat_names  = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
colors = ["#B5D4F4", "#9FE1CB", "#378ADD", "#185FA5"]
bars = ax.barh(feat_names, importances * 100, color=colors)
ax.set_xlabel("Importance (%)")
ax.set_title("Feature Importance (Random Forest)")
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, importances):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%", va="center", fontsize=9)

for idx, (name, res) in enumerate(results.items()):
    ax = axes[1, idx]
    cm = confusion_matrix(y_test, res["y_pred"])
    short_names = ['setosa', 'versicolor', 'virginica']
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=short_names,
                yticklabels=short_names,
                cbar=False)
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(axis='x', rotation=15)

axes[0, 1].axis('off')

plt.tight_layout()
plt.savefig("iris_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved as iris_results.png")

# 5. SCATTER PLOT - PETAL FEATURES

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Iris Species — Feature Scatter Plots", fontsize=13, fontweight='bold')

color_map = {0: "#378ADD", 1: "#1D9E75", 2: "#D85A30"}
label_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

for label in range(3):
    mask = y == label
    
    axes2[0].scatter(X[mask, 2], X[mask, 3],
                     c=color_map[label], label=label_map[label],
                     alpha=0.7, edgecolors='white', linewidths=0.5, s=60)
    
    axes2[1].scatter(X[mask, 0], X[mask, 1],
                     c=color_map[label], label=label_map[label],
                     alpha=0.7, edgecolors='white', linewidths=0.5, s=60)

axes2[0].set_xlabel("Petal Length (cm)")
axes2[0].set_ylabel("Petal Width (cm)")
axes2[0].set_title("Petal: most separable features")
axes2[0].legend()
axes2[0].grid(alpha=0.3)

axes2[1].set_xlabel("Sepal Length (cm)")
axes2[1].set_ylabel("Sepal Width (cm)")
axes2[1].set_title("Sepal: more overlap between classes")
axes2[1].legend()
axes2[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("iris_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("Scatter plot saved as iris_scatter.png")

# 6. FINAL SUMMARY
print("\n" + "=" * 50)
print("FINAL SUMMARY")
print("=" * 50)
best = max(results.items(), key=lambda x: x[1]["cv_mean"])
print(f"Best model by CV accuracy : {best[0]}")
print(f"CV Accuracy               : {best[1]['cv_mean']*100:.2f}% ± {best[1]['cv_std']*100:.2f}%")
print("\nKey Insights:")
print("  • Id column was dropped — not a useful feature.")
print("  • Species text labels were encoded to numbers (0, 1, 2).")
print("  • Setosa is perfectly separable from the other two species.")
print("  • Petal features carry ~87% of the predictive information.")
print("  • All three models achieve >90% accuracy on this clean dataset.")