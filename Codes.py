#!/usr/bin/env python
# coding: utf-8

# # "Multiple Performance Evaluation with Clustering and Classification Models with Machine Learning"

# In[2]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Read data
df = pd.read_csv("Data.csv", encoding="ISO-8859-9", sep=';', header=0)
df.columns = df.columns.str.strip()
df.rename(columns={
    "Country": "country",
    "Human Development Index (HDI)": "hdi",
    "Life expectancy at birth": "life_expectancy",
    "Expected years of schooling": "expected_school",
    "Mean years of schooling": "mean_school",
    "Gross national income (GNI) per capita (2021 PPP $)": "gni_per_capita"
}, inplace=True)

# GNI numerical conversion
df["gni_per_capita"] = df["gni_per_capita"].astype(str).str.replace(",", ".").astype(float)

# Columns to be normalized
features_to_normalize = ["life_expectancy", "expected_school", "mean_school", "gni_per_capita"]

# Min-Max normalization
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

# Regresyon modeli
X = df_normalized[["life_expectancy", "expected_school", "mean_school", "gni_per_capita"]]
y = df_normalized["hdi"]
model = LinearRegression()
model.fit(X, y)
coefficients = pd.Series(model.coef_, index=X.columns)

# Highest GNI and lowest HDI countries
top_gni = df_normalized.sort_values(by="gni_per_capita", ascending=False).head(5)
bottom_hdi = df_normalized.sort_values(by="hdi").head(5)

df_desc = df_normalized.describe()

df_desc, coefficients, top_gni[["country", "gni_per_capita"]], bottom_hdi[["country", "hdi"]]


# In[3]:


import pandas as pd

df = pd.read_csv("Data.csv", encoding="ISO-8859-9", sep=';', header=0)  # veya utf-8 ile de deneyebilirsin
print(df.head())


# In[4]:


df.rename(columns={
    "Country": "country",
    "Human Development Index (HDI)": "hdi",
    "Life expectancy at birth": "life_expectancy",
    "Expected years of schooling": "expected_school",
    "Mean years of schooling": "mean_school",
    "Gross national income (GNI) per capita (2021 PPP $)": "gni_per_capita"
}, inplace=True)

# Let's check
print(df.head())


# In[ ]:





# In[5]:


df.columns = df.columns.str.strip()  # Removes spaces from all column names


# In[6]:


df.columns


# In[7]:


import pandas as pd

# Specify the path to the file
df = pd.read_csv("Data.csv", encoding="ISO-8859-9", sep=';', header=0)

# Clear and rename column names
df.columns = df.columns.str.strip()
df.rename(columns={
    "Country": "country",
    "Human Development Index (HDI)": "hdi",
    "Life expectancy at birth": "life_expectancy",
    "Expected years of schooling": "expected_school",
    "Mean years of schooling": "mean_school",
    "Gross national income (GNI) per capita (2021 PPP $)": "gni_per_capita"
}, inplace=True)

# Convert numeric columns to appropriate type (some GNI data may have commas instead of periods)
df["gni_per_capita"] = df["gni_per_capita"].astype(str).str.replace(",", ".").astype(float)

# Statistical summary
print("\nÖzet İstatistikler:")
print(df.describe())

# The 5 countries with the highest GNI
print("\nGNI en yüksek 5 ülke:")
print(df.sort_values(by="gni_per_capita", ascending=False).head(5))

# The 5 countries with the lowest HDI
print("\nHDI en düşük 5 ülke:")
print(df.sort_values(by="hdi").head(5))


# # Advanced Data Analysis: 2023 Human Development Index (HDI)

# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
df = pd.read_csv("Data.csv", encoding="ISO-8859-9", sep=';', header=0)

# Clean and rename columns
df.columns = df.columns.str.strip()
df.rename(columns={
    "Country": "country",
    "Human Development Index (HDI)": "hdi",
    "Life expectancy at birth": "life_expectancy",
    "Expected years of schooling": "expected_school",
    "Mean years of schooling": "mean_school",
    "Gross national income (GNI) per capita (2021 PPP $)": "gni_per_capita"
}, inplace=True)

# Convert GNI to float (handling commas)
df["gni_per_capita"] = df["gni_per_capita"].astype(str).str.replace(",", ".").astype(float)

# Calculate correlation matrix
correlation = df[['hdi', 'life_expectancy', 'expected_school', 'mean_school', 'gni_per_capita']].corr()

# Visualize the correlation matrix
plt.figure(figsize=(10, 7))
sns.heatmap(correlation, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix', fontsize=16)
plt.show()

correlation


# In[ ]:





# # K-Means

# In[12]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

# Load the data
df = pd.read_csv("Data.csv", encoding="ISO-8859-9", sep=';', header=0)
df.columns = df.columns.str.strip()
df.rename(columns={
    "Country": "country",
    "Human Development Index (HDI)": "hdi",
    "Life expectancy at birth": "life_expectancy",
    "Expected years of schooling": "expected_school",
    "Mean years of schooling": "mean_school",
    "Gross national income (GNI) per capita (2021 PPP $)": "gni_per_capita"
}, inplace=True)

# Convert commas to periods and then to numerical values for the "gni_per_capita" column
df["gni_per_capita"] = df["gni_per_capita"].astype(str).str.replace(",", ".").astype(float)

# Normalize the features
features = ["life_expectancy", "expected_school", "mean_school", "gni_per_capita"]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features])

# Elbow Method and Silhouette Score
inertias = []
silhouette_scores = []
k_range = range(2, 11)

# Perform clustering and calculate inertia and silhouette score for each k
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

    # Print numerical values for each k
    print(f"Number of Clusters: {k}")
    print(f"Inertia: {kmeans.inertia_}")
    print(f"Silhouette Score: {silhouette_score(X_scaled, kmeans.labels_)}")
    print("-" * 30)

# Check and create directory for saving graphics
output_dir = './'  # Will save in the working directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Elbow Method Plot
plt.figure(figsize=(12, 5))

# Elbow Method - Inertia Plot
plt.subplot(1, 2, 2)
plt.plot(k_range, inertias, marker='o')
plt.axvline(x=5, color='red', linestyle='--', label="Optimal Cluster (k=4)")  # Red line for optimal cluster
plt.title("Elbow Method - Inertia")
plt.xlabel("Number of Clusters")
plt.ylabel("Total Error (Inertia)")
plt.legend()

# Silhouette Score Plot
plt.subplot(1, 2, 1)
plt.plot(k_range, silhouette_scores, marker='o', color='green')
plt.axvline(x=5, color='red', linestyle='--', label="Optimal Cluster (k=4)")  # Red line for optimal cluster
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.legend()

# Adjust layout for better visibility
plt.tight_layout()

# Save the plots
output_path = os.path.join(output_dir, "cluster_optimization.png")
plt.savefig(output_path)  # Save the plot

# Display the plot
plt.show()  # Show the plot immediately


# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:


from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import seaborn as sns

# Define the optimal number of clusters
optimal_k = 4

# Perform KMeans clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)  # Reduce the features to 2 components for visualization
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
pca_df['cluster'] = df['cluster']
pca_df['country'] = df['country']

# Plot the clusters in 2D using PCA components and label the countries
plt.figure(figsize=(12, 8))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='cluster', palette='Set1', s=100, marker='o', alpha=0.8)

# Label each point with the corresponding country
for line in range(0, pca_df.shape[0]):
    plt.text(pca_df.PCA1[line] + 0.02, pca_df.PCA2[line], pca_df['country'][line], horizontalalignment='left', size='small', color='black')

# Mark the cluster centers (centroids)
centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', marker='X', label='Centroids')

plt.title(f'PCA Visualization of Clusters (k={optimal_k})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Perform t-SNE for alternative visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Create a DataFrame for t-SNE results
tsne_df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
tsne_df['cluster'] = df['cluster']
tsne_df['country'] = df['country']

# Plot t-SNE results
plt.figure(figsize=(12, 8))
sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='cluster', palette='Set1', s=100, marker='o', alpha=0.9)

# Label each point with the corresponding country
for line in range(0, tsne_df.shape[0]):
    plt.text(tsne_df.TSNE1[line] + 0.02, tsne_df.TSNE2[line], tsne_df['country'][line], horizontalalignment='left', size='small', color='black')

# Mark the cluster centers for t-SNE (this is not straightforward, just for visualization)
plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', marker='X', label='Centroids')

plt.title(f't-SNE Visualization of Clusters (k={optimal_k})')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()

# Silhouette Score for the clusters
silhouette = silhouette_score(X_scaled, df['cluster'])
print(f"Silhouette Score for k={optimal_k}: {silhouette}")

# Davies-Bouldin Index for the clusters (lower value is better)
db_index = davies_bouldin_score(X_scaled, df['cluster'])
print(f"Davies-Bouldin Index for k={optimal_k}: {db_index}")


# In[ ]:





# In[14]:


# Print number of elements in each cluster
cluster_counts = df['cluster'].value_counts().sort_index()
print("Number of elements in each cluster:")
for cluster_id, count in cluster_counts.items():
    print(f"Cluster {cluster_id}: {count} countries")


# In[15]:


# Ülke isimlerini ve küme etiketlerini birleştir
clustered_data = df[["country", "cluster"]].sort_values(by="cluster")

# Her küme için ülke isimlerini yazdır
for cluster_id in sorted(df['cluster'].unique()):
    countries_in_cluster = clustered_data[clustered_data['cluster'] == cluster_id]["country"].tolist()
    print(f"\nCluster {cluster_id} ({len(countries_in_cluster)} countries):")
    print(", ".join(countries_in_cluster))


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
ax = sns.countplot(x='cluster', data=df, palette='pastel')
plt.title("Number of Countries in Each Cluster")
plt.xlabel("Cluster Label")
plt.ylabel("Number of Countries")

# Her çubuğun üstüne sayı etiketini yaz
for p in ax.patches:
    count = int(p.get_height())
    ax.annotate(str(count), (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10)

plt.show()


# In[ ]:





# In[17]:


cluster_counts = df['cluster'].value_counts().sort_index()
labels = [f'Cluster {i}' for i in cluster_counts.index]
sizes = cluster_counts.values

# Etiket fonksiyonu: yüzde + sayı
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        count = int(round(pct * total / 100.0))
        return f'{pct:.1f}%\n({count})'
    return my_autopct

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct=make_autopct(sizes), 
        colors=plt.cm.Paired.colors, startangle=90)
plt.title("Cluster Distribution")
plt.tight_layout()
plt.show()


# # "Cluster Profiling and Feature Importance Analysis in Socioeconomic Country Groupings"
# 

# In[18]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import f_oneway
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv("Data.csv", encoding="ISO-8859-9", sep=';')
df.columns = df.columns.str.strip()
df.rename(columns={
    "Country": "country",
    "Human Development Index (HDI)": "hdi",
    "Life expectancy at birth": "life_expectancy",
    "Expected years of schooling": "expected_school",
    "Mean years of schooling": "mean_school",
    "Gross national income (GNI) per capita (2021 PPP $)": "gni_per_capita"
}, inplace=True)
df["gni_per_capita"] = df["gni_per_capita"].astype(str).str.replace(",", ".").astype(float)

features = ["life_expectancy", "expected_school", "mean_school", "gni_per_capita"]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features])

# KMeans clustering with optimal cluster = 4
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 1. Cluster Means
cluster_means = df.groupby('cluster')[features].mean()

# 2. ANOVA Tests for each feature across clusters
anova_results = {}
for feature in features:
    samples = [df[df['cluster'] == k][feature] for k in sorted(df['cluster'].unique())]
    stat, p = f_oneway(*samples)
    anova_results[feature] = p

# 3. PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]

# 4. Decision Tree to explain clustering
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(df[features], df['cluster'])

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# PCA Plot
sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='tab10', ax=axes[0, 0])
axes[0, 0].set_title("PCA Visualization of Clusters")

# Cluster Means Heatmap
sns.heatmap(cluster_means, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[0, 1])
axes[0, 1].set_title("Cluster Feature Means")

# ANOVA Results
anova_df = pd.DataFrame(list(anova_results.items()), columns=["Feature", "p-value"])
sns.barplot(data=anova_df, x="Feature", y="p-value", palette="muted", ax=axes[1, 0])
axes[1, 0].axhline(0.05, ls='--', color='red')
axes[1, 0].set_title("ANOVA p-values by Feature")

# Decision Tree Plot
plot_tree(clf, feature_names=features, class_names=[str(i) for i in clf.classes_],
          filled=True, rounded=True, ax=axes[1, 1])
axes[1, 1].set_title("Decision Tree Explaining Clusters")

plt.tight_layout()
plt.show()

cluster_means, anova_df.round(4)


# In[ ]:





# # Machine Learning

# In[ ]:





# In[19]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv("Data.csv", encoding="ISO-8859-9", sep=';')
df.columns = df.columns.str.strip()
df.rename(columns={
    "Country": "country",
    "Human Development Index (HDI)": "hdi",
    "Life expectancy at birth": "life_expectancy",
    "Expected years of schooling": "expected_school",
    "Mean years of schooling": "mean_school",
    "Gross national income (GNI) per capita (2021 PPP $)": "gni_per_capita"
}, inplace=True)
df["gni_per_capita"] = df["gni_per_capita"].astype(str).str.replace(",", ".").astype(float)

# 2. Feature Scaling
features = ["life_expectancy", "expected_school", "mean_school", "gni_per_capita"]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features])

# 3. Clustering using K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 4. Classification using ML Models (with Train-Test Split)
X = df[features]
y = df['cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
y_train_bin = label_binarize(y_train, classes=[0, 1, 2, 3])
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = []
roc_results = []

# === BURADA MODEL DÖNGÜSÜ BİTİŞİK OLARAK EKLENMİŞTİR ===
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    y_score = model.predict_proba(X_test)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(y_test_bin.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    roc_results.append({'Model': name, 'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc})
    results.append({'Model': name, 'Accuracy': acc, 'Confusion Matrix': cm})
    print(f"\n=== {name} ===")
    print("Classification Report:")
    print(classification_report(y_test, preds))
    print("Confusion Matrix (Numeric):")
    print(cm)
    print("ROC AUC Scores:")
    for class_idx in roc_auc:
        print(f"  Class {class_idx}: AUC = {roc_auc[class_idx]:.4f}")

# 5. Accuracy Comparison
results_df = pd.DataFrame(results)[['Model', 'Accuracy']]
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="Model", y="Accuracy", palette="pastel")
plt.ylim(0, 1.05)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.xticks(rotation=45)
for index, row in results_df.iterrows():
    plt.text(index, row['Accuracy'] + 0.01, f'{row["Accuracy"]:.2f}', ha='center')
plt.tight_layout()
plt.show()

# 6. Confusion Matrices
palettes = ['Reds', 'Greens', 'Purples', 'Oranges', 'YlOrBr', 'Blues']
fig, axs = plt.subplots(3, 2, figsize=(15, 20))
axs = axs.flatten()
for i, res in enumerate(results):
    cm = res['Confusion Matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap=palettes[i % len(palettes)], ax=axs[i])
    axs[i].set_title(f"{res['Model']} - Confusion Matrix")
    axs[i].set_xlabel("Predicted")
    axs[i].set_ylabel("True")
for j in range(len(results), len(axs)):
    fig.delaxes(axs[j])
plt.tight_layout()
plt.show()

# 7. ROC Curve and AUC
fig, axs = plt.subplots(3, 2, figsize=(13, 20))
axs = axs.flatten()
for i, res in enumerate(roc_results):
    ax = axs[i]
    for j in range(len(res['roc_auc'])):
        ax.plot(res['fpr'][j], res['tpr'][j], label=f'Class {j} (AUC = {res["roc_auc"][j]:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {res["Model"]}')
    ax.legend(loc='lower right')
plt.tight_layout()
plt.show()


# # Hyperparameter Optimization and Performance Evaluation with GridSearchCV for Machine Learning Models"
# I think the most suitable improvement suggestion is Hyperparameter Tuning.
# Hyperparameter tuning is a very effective method to optimize models and can significantly improve model performance. In this process, we will try different values ​​of hyperparameters of each model and find the hyperparameters that give the best performance.
# 
# Steps:
# We will perform hyperparameter search using GridSearchCV or RandomizedSearchCV.
# 
# We will find which hyperparameters give the best results.
# 
# We will retrain our model with these hyperparameters and analyze the results.
# 
# Hyperparameter Tuning with GridSearchCV
# First, let's perform hyperparameter optimization using GridSearchCV with RandomForestClassifier. Then, we will train using the best model.
# 
# Sample Code for Random Forest with GridSearchCV:

# In[20]:


from sklearn.model_selection import GridSearchCV

#Hyperparameter grids
param_grids = {
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt'],
        'bootstrap': [True]
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    },
    "KNN": {
        'n_neighbors': [3, 5, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear'],
        'penalty': ['l2']
    },
    "Decision Tree": {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
}

# Let's redefine model classes
model_classes = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = []
roc_results = []

# === Training cycle for each model with GridSearchCV ===
for name in model_classes.keys():
    print(f"\n>>> {name} için GridSearchCV başlatılıyor...")
    model = model_classes[name]
    param_grid = param_grids[name]

    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=5,
                               n_jobs=-1,
                               verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"En İyi Parametreler: {grid_search.best_params_}")
    print(f"Test Doğruluğu: {acc:.4f}")

    y_score = best_model.predict_proba(X_test)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(y_test_bin.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    roc_results.append({'Model': name, 'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc})
    results.append({'Model': name, 'Accuracy': acc, 'Confusion Matrix': cm})
    print("Classification Report:")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(cm)
    print("ROC AUC Scores:")
    for class_idx in roc_auc:
        print(f"  Class {class_idx}: AUC = {roc_auc[class_idx]:.4f}")


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report
import numpy as np

# === Training cycle for each model with GridSearchCV ===
for name in model_classes.keys():
    print(f"\n>>> {name} için GridSearchCV başlatılıyor...")
    model = model_classes[name]
    param_grid = param_grids[name]

    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=5,
                               n_jobs=-1,
                               verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"En İyi Parametreler: {grid_search.best_params_}")
    print(f"Test Doğruluğu: {acc:.4f}")

    # Calculating ROC and AUC
    y_score = best_model.predict_proba(X_test)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(y_test_bin.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Creating a ROC chart
    plt.figure(figsize=(10, 6))
    for i in range(y_test_bin.shape[1]):
        plt.plot(fpr[i], tpr[i], label=f'{name} - Class {i} (AUC = {roc_auc[i]:.4f})')
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # random classifier line
    plt.title(f'{name} - ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Visualizing the Confusion Matrix
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Accuracy and ROC labels on AUC graph
    print("Classification Report:")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(cm)
    print("ROC AUC Scores:")
    for class_idx in roc_auc:
        print(f"  Class {class_idx}: AUC = {roc_auc[class_idx]:.4f}")
    
    # Adding labels in ROC AUC graphs
    for i in range(y_test_bin.shape[1]):
        plt.text(fpr[i][len(fpr[i])//2], tpr[i][len(tpr[i])//2], f'{roc_auc[i]:.4f}', color='black', fontsize=12, ha='center')

    # Plotting the Accuracy chart
    plt.figure(figsize=(10, 6))
    plt.bar(name, acc, color='green')
    plt.title(f'{name} - Accuracy Score')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.text(name, acc + 0.02, f'{acc:.4f}', ha='center', fontsize=12)  # Accuracy etiketleme
    plt.show()


# In[23]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# === Model definitions and hyperparameters ===
param_grids = {
    "Random Forest": {
        'n_estimators': [100],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'max_features': ['sqrt'],
        'bootstrap': [True]
    },
    "SVM": {
        'C': [1],
        'gamma': ['scale'],
        'kernel': ['rbf']
    },
    "KNN": {
        'n_neighbors': [5],
        'weights': ['uniform'],
        'metric': ['euclidean']
    },
    "Logistic Regression": {
        'C': [1],
        'solver': ['lbfgs'],
        'penalty': ['l2']
    },
    "Decision Tree": {
        'criterion': ['gini'],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    },
    "Gradient Boosting": {
        'n_estimators': [100],
        'learning_rate': [0.1],
        'max_depth': [3],
        'subsample': [1.0]
    }
}

model_classes = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# === NOTE: You must have defined the following variables before ===
# X_train, X_test, y_train, y_test, y_test_bin

for name in model_classes.keys():
    print(f"\n>>> Starting GridSearchCV for {name}...")
    model = model_classes[name]
    param_grid = param_grids[name]

    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=5,
                               n_jobs=-1,
                               verbose=0)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Test Accuracy: {acc:.4f}")

    # === ROC curve and AUC calculation ===
    y_score = best_model.predict_proba(X_test)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(y_test_bin.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # === ROC plot ===
    plt.figure(figsize=(10, 6))
    for i in range(y_test_bin.shape[1]):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    # === Dynamic Accuracy box ===
    text_x = 0.05  # fixed position near left
    text_y = 0.05
    plt.text(text_x, text_y, f'Accuracy = {acc:.4f}',
             fontsize=12,
             bbox=dict(facecolor='white', edgecolor='black'))

    plt.title(f'{name} - ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # === Confusion Matrix visualization ===
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # === Accuracy Bar Chart ===
    plt.figure(figsize=(6, 4))
    plt.bar(name, acc, color='green')
    plt.ylim(0, 1)
    plt.title(f'{name} - Accuracy Score')
    plt.ylabel('Accuracy')
    plt.text(0, acc + 0.02, f'{acc:.4f}', ha='center', fontsize=12)
    plt.show()

    # === Textual Reports ===
    print("Classification Report:")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(cm)
    print("ROC AUC Scores:")
    for class_idx in roc_auc:
        print(f"  Class {class_idx}: AUC = {roc_auc[class_idx]:.4f}")


# In[29]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# === Model definitions and hyperparameters ===
param_grids = {
    "Random Forest": {
        'n_estimators': [100],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'max_features': ['sqrt'],
        'bootstrap': [True]
    },
    "SVM": {
        'C': [1],
        'gamma': ['scale'],
        'kernel': ['rbf']
    },
    "KNN": {
        'n_neighbors': [5],
        'weights': ['uniform'],
        'metric': ['euclidean']
    },
    "Logistic Regression": {
        'C': [1],
        'solver': ['lbfgs'],
        'penalty': ['l2']
    },
    "Decision Tree": {
        'criterion': ['gini'],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    },
    "Gradient Boosting": {
        'n_estimators': [100],
        'learning_rate': [0.1],
        'max_depth': [3],
        'subsample': [1.0]
    }
}

model_classes = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# === Note: X_train, X_test, y_train, y_test, y_test_bin must be defined before running ===

num_models = len(model_classes)
cols = 2
rows = ceil(num_models / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols*7, rows*6), squeeze=False)

for idx, (name, model) in enumerate(model_classes.items()):
    print(f"\n>>> Starting GridSearchCV for {name}...")
    param_grid = param_grids[name]

    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=5,
                               n_jobs=-1,
                               verbose=0)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Test Accuracy: {acc:.4f}")

    y_score = best_model.predict_proba(X_test)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(y_test_bin.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    row_idx = idx // cols
    col_idx = idx % cols
    ax = axes[row_idx, col_idx]

    for i in range(y_test_bin.shape[1]):
        ax.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.text(0.05, 0.05, f'Accuracy = {acc:.4f}',
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(facecolor='white', edgecolor='black'))
    ax.set_title(f'{name} - ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    ax.grid(True)

# Remove any unused subplots
for empty_idx in range(num_models, rows*cols):
    fig.delaxes(axes[empty_idx // cols, empty_idx % cols])

plt.tight_layout()
plt.show()

# Other visualizations and reports: confusion matrix, accuracy bar, and classification report
for name, model in model_classes.items():
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grids[name],
                               scoring='accuracy',
                               cv=5,
                               n_jobs=-1,
                               verbose=0)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    # Confusion Matrix Plot
    plt.figure(figsize=(7, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Accuracy Bar Chart
    plt.figure(figsize=(6, 5))
    plt.bar(name, acc, color='green')
    plt.ylim(0, 1)
    plt.title(f'{name} - Accuracy Score')
    plt.ylabel('Accuracy')
    plt.text(0, acc + 0.02, f'{acc:.4f}', ha='center', fontsize=12)
    plt.show()

    # Text Reports
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, preds))
    print(f"Confusion Matrix for {name}:")
    print(cm)



# In[ ]:





# In[ ]:




