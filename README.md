# 🛍️ Mall Customer Segmentation

An interactive web application that segments mall customers into distinct groups using unsupervised machine learning (clustering) techniques. Built to help businesses design targeted marketing strategies and improve membership card conversion rates.

🔗 **Live App:** [[Add your Streamlit Cloud link here]](https://mall-customer-segmentation-wfv2gxjajkvuemyck5qfmp.streamlit.app/)

---

## 📌 Problem Statement

How can different clustering techniques be utilized to assist a supermarket in increasing their membership card conversion rate? By performing a customer segmentation analysis, we identify groups of customers with similar shopping preferences and purchasing histories, allowing the business to tailor marketing strategies more effectively to each group.

---

## 🚀 Features

- **Data Overview** – Dataset preview, missing value checks, statistical summary, Elbow Method & Silhouette Score analysis to determine the optimal number of clusters
- **K-Means Clustering** – Interactive clustering with adjustable K, cluster visualization with centroids, cluster profiling, gender distribution, correlation heatmap, and a live prediction tool for new customers
- **Hierarchical Clustering** – Dendrogram visualization and Agglomerative Clustering results
- **DBSCAN Clustering** – Density-based clustering with adjustable epsilon and minimum samples, k-distance graph, and outlier analysis
- **Model Comparison** – Side-by-side Silhouette Score comparison across K-Means, Hierarchical, and DBSCAN to identify the best-performing model
- **3D Visualization** – Interactive 3D scatter plot (Age, Income, Spending Score) using Plotly

---

## 🛠️ Tech Stack

- **Language:** Python
- **Web Framework:** Streamlit
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, SciPy

---

## 📊 Dataset

The dataset contains 200 mall customer records with the following attributes:
- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1–100)

---

## ⚙️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/<your-username>/mall-customer-segmentation.git
cd mall-customer-segmentation

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📈 Methodology

1. **Data Preprocessing** – Encoded categorical variables and scaled numerical features using StandardScaler
2. **Cluster Optimization** – Used the Elbow Method and Silhouette Score to determine the optimal number of clusters
3. **Clustering Algorithms** – Applied K-Means, Hierarchical (Agglomerative), and DBSCAN clustering
4. **Model Evaluation** – Compared all three algorithms using Silhouette Scores
5. **Business Interpretation** – Translated each cluster into actionable membership card strategies (Premium, Discount, Exclusive Perks, Standard, Free Tier)

---

## 👥 Team Members

| Student ID | Name |
|---|---|
| KIC-HNDCSAI-261F-05 | Chethani |
| KIC-HNDCSAI-261F-07 | Isurika |
| KIC-HNDCSAI-261F-29 | Abhimani |

---

## 📄 License

This project was developed for academic purposes as part of the HND in Computing Science (AI) program.
