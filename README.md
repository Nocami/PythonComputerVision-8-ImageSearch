# PythonComputerVision-7-ImageSearch
利用文本挖掘技术对基于图像视觉内容进行图像搜索，建立视觉单词（视觉码本）的概念，再建立相应数据库，最终实现在数据库中搜索图像，利用索引获取候选图像，再使用一幅图像进行查询。将上述工作最终建立为相应的演示程序以及web应用。  
## 一.基于内容的图像检索  
在大型图像数据库上，CBIR（Content-Based Image Retrieval基于内容的图像检索），用于检索在视觉上具有相似性的图像，如颜色、纹理、图像中相似的物体或场景等等。  
但是传统的数据库图像匹配时不可行的，因为数据库很大的情况下，利用特征匹配的查询方式会耗费过多的时间。所以我们引入一种模型--矢量空间模型。  
### 1.矢量空间模型  
矢量空间模型时一个用于表示和搜索文本文档的模型，它基本可以应用于任何对象类型，包括图像。这些矢量是由文本词频直方图构成的，换句话说，矢量包含了每个单词出现的次数，
而且在其他别的地方包含很多0元素。由于其忽略了单词出现的顺序及位置，该模型也被称为BOW表示模型。  
通过单词计数来构建文档直方图向量v，从而建立文档索引。通常，数据集中一个单词的重要性与它在文档中出现的次数成正比，而与它在语料库中出现的次数成反比。
