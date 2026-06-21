import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")
st.title("Mall Customer Segmentation")
st.write("Group customers into similar types using clustering techniques (K-Means, Hierarchical, DBSCAN)")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Dataset_3.csv")

le = LabelEncoder()
df['Gender_Encoded'] = le.fit_transform(df['Gender'])

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.header("Settings")
k = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=10, value=5)

# ---------------- FEATURE SETS ----------------
X1 = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
X2 = df[['Age', 'Spending Score (1-100)']].values
X3 = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

scaler1 = StandardScaler()
X1_scaled = scaler1.fit_transform(X1)

scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)

scaler3 = StandardScaler()
X3_scaled = scaler3.fit_transform(X3)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data Overview",
    "K-Means",
    "Hierarchical",
    "DBSCAN",
    "Model Comparison",
    "3D View"
])

# ================= TAB 1: DATA OVERVIEW =================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        st.write("Shape:", df.shape)
        st.write("Missing values:")
        st.write(df.isnull().sum())
    with col2:
        st.write("Gender Count")
        st.bar_chart(df['Gender'].value_counts())

    st.write("Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("Elbow Method")
    inertia = []
    K_range = range(1, 11)
    for i in K_range:
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        km.fit(X1_scaled)
        inertia.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(K_range, inertia, marker='o', color='blue')
    ax.set_title('Elbow Method - Optimal Number of Clusters', fontsize=14)
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Inertia (WCSS)')
    st.pyplot(fig, use_container_width=True)

    st.subheader("Silhouette Score by K")
    sil_scores = []
    for i in range(2, 11):
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        labels = km.fit_predict(X1_scaled)
        sil_scores.append(silhouette_score(X1_scaled, labels))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(2, 11), sil_scores, marker='s', color='green')
    ax.set_title('Silhouette Score by Number of Clusters', fontsize=14)
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Silhouette Score')
    st.pyplot(fig, use_container_width=True)

    best_k = list(range(2, 11))[sil_scores.index(max(sil_scores))]
    st.success(f"Best K according to Silhouette Score: {best_k}")

# ================= TAB 2: K-MEANS =================
with tab2:
    st.subheader(f"K-Means Clustering (K={k})")

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['KMeans_Cluster'] = kmeans.fit_predict(X1_scaled)
    km_score = silhouette_score(X1_scaled, df['KMeans_Cluster'])
    st.write(f"Silhouette Score: {km_score:.4f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(k):
        subset = df[df['KMeans_Cluster'] == i]
        ax.scatter(subset['Annual Income (k$)'], subset['Spending Score (1-100)'],
                   c=colors[i % len(colors)], label=f'Cluster {i}', s=80, alpha=0.7)
    centroids = scaler1.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='X', label='Centroids')
    ax.set_title('K-Means Clustering: Annual Income vs Spending Score')
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    st.subheader("Cluster Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        cluster_counts = df['KMeans_Cluster'].value_counts().sort_index()
        ax.pie(cluster_counts, labels=[f'Cluster {i}' for i in cluster_counts.index],
               autopct='%1.1f%%', colors=colors[:k], startangle=140, shadow=True)
        ax.set_title('Customer Distribution by Cluster')
        st.pyplot(fig, use_container_width=True)
    with col2:
        gender_cluster = df.groupby(['KMeans_Cluster', 'Gender']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 6))
        gender_cluster.plot(kind='bar', color=['salmon', 'steelblue'], edgecolor='black', ax=ax)
        ax.set_title('Gender Distribution Across Clusters')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Count')
        plt.xticks(rotation=0)
        st.pyplot(fig, use_container_width=True)

    st.subheader("Cluster Profile Summary")
    cluster_summary = df.groupby('KMeans_Cluster').agg(
        Count=('CustomerID', 'count'),
        Avg_Age=('Age', 'mean'),
        Avg_Income=('Annual Income (k$)', 'mean'),
        Avg_Spending=('Spending Score (1-100)', 'mean'),
        Female_Pct=('Gender_Encoded', lambda x: (x == 0).sum() / len(x) * 100)
    ).round(2)
    st.dataframe(cluster_summary)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cluster_summary['Avg_Age'].plot(kind='bar', ax=axes[0], color=colors[:k], edgecolor='black')
    axes[0].set_title('Average Age per Cluster')
    cluster_summary['Avg_Income'].plot(kind='bar', ax=axes[1], color=colors[:k], edgecolor='black')
    axes[1].set_title('Average Income per Cluster')
    cluster_summary['Avg_Spending'].plot(kind='bar', ax=axes[2], color=colors[:k], edgecolor='black')
    axes[2].set_title('Average Spending per Cluster')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'KMeans_Cluster']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, ax=ax)
    st.pyplot(fig, use_container_width=True)

    st.subheader("Predict Cluster for a New Customer")
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
    with col2:
        spending = st.number_input("Spending Score (1-100)", min_value=0, max_value=100, value=50)
    if st.button("Predict Cluster"):
        new_data = scaler1.transform([[income, spending]])
        cluster = kmeans.predict(new_data)[0]
        st.success(f"This customer belongs to Cluster {cluster}")

# ================= TAB 3: HIERARCHICAL =================
with tab3:
    st.subheader("Hierarchical Clustering (Dendrogram)")
    fig, ax = plt.subplots(figsize=(16, 7))
    linked = linkage(X1_scaled, method='ward')
    dendrogram(linked, truncate_mode='lastp', p=20, leaf_rotation=90, leaf_font_size=10,
               show_contracted=True, ax=ax)
    ax.set_title('Hierarchical Clustering Dendrogram (Ward Linkage)')
    st.pyplot(fig, use_container_width=True)

    hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
    df['HC_Cluster'] = hc.fit_predict(X1_scaled)
    hc_score = silhouette_score(X1_scaled, df['HC_Cluster'])
    st.write(f"Silhouette Score: {hc_score:.4f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(k):
        subset = df[df['HC_Cluster'] == i]
        ax.scatter(subset['Annual Income (k$)'], subset['Spending Score (1-100)'],
                   c=colors[i % len(colors)], label=f'Cluster {i}', s=80, alpha=0.7)
    ax.set_title('Hierarchical Clustering: Annual Income vs Spending Score')
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.legend()
    st.pyplot(fig, use_container_width=True)

# ================= TAB 4: DBSCAN =================
with tab4:
    st.subheader("DBSCAN Clustering")

    nbrs = NearestNeighbors(n_neighbors=5).fit(X1_scaled)
    distances, _ = nbrs.kneighbors(X1_scaled)
    distances = np.sort(distances[:, 4])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(distances, color='purple')
    ax.set_title('K-Distance Graph (for choosing epsilon)')
    ax.set_xlabel('Data Points (sorted)')
    ax.set_ylabel('5th Nearest Neighbor Distance')
    st.pyplot(fig, use_container_width=True)

    eps = st.slider("DBSCAN epsilon (eps)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    min_samples = st.slider("Minimum samples", min_value=2, max_value=15, value=5)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['DBSCAN_Cluster'] = dbscan.fit_predict(X1_scaled)
    n_clusters = len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'].values else 0)
    n_noise = list(df['DBSCAN_Cluster']).count(-1)
    st.write(f"Number of clusters found: {n_clusters}")
    st.write(f"Number of outliers (noise points): {n_noise}")

    if n_clusters > 1:
        db_score = silhouette_score(X1_scaled, df['DBSCAN_Cluster'])
        st.write(f"Silhouette Score: {db_score:.4f}")
    else:
        db_score = None
        st.warning("Not enough clusters to calculate Silhouette Score. Try changing eps.")

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(df['DBSCAN_Cluster'].unique())
    db_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, db_colors):
        subset = df[df['DBSCAN_Cluster'] == label]
        name = f'Cluster {label}' if label != -1 else 'Outliers/Noise'
        marker = 'x' if label == -1 else 'o'
        ax.scatter(subset['Annual Income (k$)'], subset['Spending Score (1-100)'],
                   c=[color], label=name, s=80, marker=marker, alpha=0.7)
    ax.set_title('DBSCAN Clustering: Annual Income vs Spending Score')
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    if n_noise > 0:
        st.subheader("Outlier Characteristics")
        outliers = df[df['DBSCAN_Cluster'] == -1]
        st.dataframe(outliers[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].describe())

# ================= TAB 5: MODEL COMPARISON =================
with tab5:
    st.subheader("Comparing Clustering Methods (Silhouette Score)")

    kmeans_c = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = kmeans_c.fit_predict(X1_scaled)
    km_score = silhouette_score(X1_scaled, km_labels)

    hc_c = AgglomerativeClustering(n_clusters=k, linkage='ward')
    hc_labels = hc_c.fit_predict(X1_scaled)
    hc_score = silhouette_score(X1_scaled, hc_labels)

    dbscan_c = DBSCAN(eps=0.5, min_samples=5)
    db_labels = dbscan_c.fit_predict(X1_scaled)
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    db_score_c = silhouette_score(X1_scaled, db_labels) if n_clusters_db > 1 else 0

    models = ['K-Means', 'Hierarchical (Ward)', 'DBSCAN']
    scores = [km_score, hc_score, db_score_c]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(models, scores, color=['blue', 'green', 'purple'], alpha=0.6, edgecolor='black')
    ax.set_title('Model Comparison: Silhouette Scores')
    ax.set_ylabel('Silhouette Score')
    for i, v in enumerate(scores):
        ax.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
    st.pyplot(fig, use_container_width=True)

    best_model = models[scores.index(max(scores))]
    st.success(f"Best performing model: {best_model} (Silhouette Score: {max(scores):.4f})")

# ================= TAB 6: 3D VIEW =================
with tab6:
    st.subheader("3D Customer Segmentation (Age, Income, Spending Score)")

    kmeans_3d = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['KMeans_3D_Cluster'] = kmeans_3d.fit_predict(X3_scaled)
    sil_3d = silhouette_score(X3_scaled, df['KMeans_3D_Cluster'])
    st.write(f"Silhouette Score (3D): {sil_3d:.4f}")

    fig = px.scatter_3d(df,
                        x='Age',
                        y='Annual Income (k$)',
                        z='Spending Score (1-100)',
                        color=df['KMeans_3D_Cluster'].astype(str),
                        title='3D Customer Segmentation',
                        opacity=0.8)
    st.plotly_chart(fig, use_container_width=True)

# ================= SIDEBAR: BUSINESS INSIGHT =================
st.sidebar.markdown("---")
st.sidebar.subheader("Membership Strategy Guide")
st.sidebar.write("""
- High Income, High Spending → Premium Membership
- Low Income, High Spending → Discount Membership
- High Income, Low Spending → Exclusive Perks
- Average Income/Spending → Standard Membership
- Low Income, Low Spending → Free Tier
""")
