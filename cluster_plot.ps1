import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate and cluster data
X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X)

# Create plot
fig = px.scatter(x=X[:,0], y=X[:,1], color=clusters,
                 title="K-Means Clustering",
                 labels={'x': 'Feature 1', 'y': 'Feature 2'})
fig.write_html('cluster_plot.html')
