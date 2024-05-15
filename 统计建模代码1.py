from transformers import BertModel, BertTokenizer
from sklearn.cluster import KMeans
import jieba
import numpy as np
import torch

# 加载预训练的 BERT 模型和分词器
bert_model = BertModel.from_pretrained("bert-base-chinese")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# 读取文本文件 1.txt
with open('1.txt', 'r', encoding='utf-8') as f:
    texts = f.read()


# 对文本进行分词
tokenized_texts = [jieba.lcut(text) for text in texts]

# 使用 BERT 获取文本的向量表示
embeddings = []

for tokens in tokenized_texts:
    input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])
    input_ids = torch.tensor([input_ids])
    with torch.no_grad():
        outputs = bert_model(input_ids)
        embeddings.append(outputs.last_hidden_state[:, 0, :].numpy())
# 将 embeddings 列表中的二维数组 "flatten" 成一维数组，并堆叠成一个二维数组
embeddings = np.array([embedding.flatten() for embedding in embeddings])

# 使用 K-means 进行聚类
kmeans = KMeans(n_clusters=2)
kmeans.fit(embeddings)

# 打印聚类结果
print(kmeans.labels_)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.rcParams["font.sans-serif"]=['SimHei']
plt.rcParams['axes.unicode_minus']=False
# 使用 PCA 将 embeddings 降维到 2 维
pca = PCA(n_components=10)
reduced_embeddings = pca.fit_transform(embeddings)

# 绘制散点图
plt.scatter(reduced_embeddings[:, 3], reduced_embeddings[:, 2], c=kmeans.labels_)
plt.title('深圳市数字化企业与非数字化企业散点图')
plt.xlabel('X词向量')
plt.ylabel('Y词向量')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.rcParams["font.sans-serif"]=['SimHei']
plt.rcParams['axes.unicode_minus']=False
# 使用 PCA 将 embeddings 降维到 2 维
pca = PCA(n_components=106)
reduced_embeddings = pca.fit_transform(embeddings)

# 使用 seaborn 的 kdeplot 绘制热力图
sns.kdeplot(
    x=reduced_embeddings[:, 5],
    y=reduced_embeddings[:, 2],
    fill=True,
    cmap="viridis",  # 选择一个颜色映射，例如 "viridis", "magma", "inferno"
    thresh=0,  # 设置密度阈值，低于该值的区域将不被填充
    levels=100, # 设置颜色分级数量，越大颜色过渡越平滑
)
plt.title('数字化企业与非数字化企业热力图')
plt.xlabel('数字化企业')
plt.ylabel('非数字化企业')
plt.show()
print(len(reduced_embeddings[:, 0],)/len(reduced_embeddings[:, 1],))



from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

silhouette_coefficients = []
for i in range(2, 106):  # k 值从 2 到 50
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(embeddings)
    score = silhouette_score(embeddings, kmeans.labels_)
    silhouette_coefficients.append(score)

plt.plot(range(2, 106), silhouette_coefficients)
plt.title('Silhouette Coefficient')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Coefficient')
plt.show()
