# PythonComputerVision-7-ImageSearch
利用文本挖掘技术对基于图像视觉内容进行图像搜索，建立视觉单词（视觉码本）的概念，再建立相应数据库，最终实现在数据库中搜索图像，利用索引获取候选图像，再使用一幅图像进行查询。将上述工作最终建立为相应的演示程序以及web应用。  
## 一.基于内容的图像检索  
在大型图像数据库上，CBIR（Content-Based Image Retrieval基于内容的图像检索），用于检索在视觉上具有相似性的图像，如颜色、纹理、图像中相似的物体或场景等等。  
但是传统的数据库图像匹配时不可行的，因为数据库很大的情况下，利用特征匹配的查询方式会耗费过多的时间。所以我们引入一种模型--矢量空间模型。  
### 1.矢量空间模型  
矢量空间模型时一个用于表示和搜索文本文档的模型，它基本可以应用于任何对象类型，包括图像。这些矢量是由文本词频直方图构成的，换句话说，矢量包含了每个单词出现的次数，
而且在其他别的地方包含很多0元素。由于其忽略了单词出现的顺序及位置，该模型也被称为BOW表示模型。  
通过单词计数来构建文档直方图向量v，从而建立文档索引。通常，数据集中一个单词的重要性与它在文档中出现的次数成正比，而与它在语料库中出现的次数成反比。  
最常用的权重是tf-idf（tern frequency-inverse document frequency，词频-逆向文档频率），单词w在文档d中的词频是:  
![image](01.jpg)  
逆向文件频率 (inverse document frequency, IDF)  IDF的主要思想是：如果包含词条t的文档越少, IDF越大，则说明词条具有很好的类别区分能力。某一特定词语的IDF，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取对数得到。  
![image](02.jpg)  
某一特定文件内的高词语频率，以及该词语在整个文件集合中的低文件频率，可以产生出高权重的TF-IDF。因此，TF-IDF倾向于过滤掉常见的词语，保留重要的词语。  
![image](03.jpg)   
## 二.视觉单词
将文本挖掘技术应用到图像中，首先需要建立等效视觉单词，可以用之前博文中提到的SIFT局部描述子做到。它的思想是将描述子空间量化成一些典型实例，并将图像中的每个描述子指派到其中的某个实例中。这些典型实例可以通过分析训练图像集确定，并被视为视觉单词。所有这些视觉单词构成的集合称为**视觉词汇**，有时也称为**视觉码本**。  
### 1.BOW模型
从一个很大的训练图像提取特征描述子，利用一些聚类算法可以构建出视觉单词。聚类算法最常用的是**K-means**，这里也采用K-means。视觉单词并不抽象，它只是在给定特殊描述子空间中的一组向量集，在采用K-means进行聚类时得到的视觉单词时聚类质心。用视觉单词直方图来表示图像，则该模型称为BOW模型。这里展示一个示例数据集，用它可以说明BOW概念。文件first1000.zip包含了肯塔基大学物体识别数据集（ukbench）的前1000幅图片，完整数据集及配套代码可以去 http://www.bis.uky.edu/~stewe/ukbench/ 找到。这个数据集有很多子集，每个子集包括四幅图像，具有相同的场景或物体，而且存储的文件名时连续的。如下图所示：  
![image](04.jpg)  
### 2.创建词汇
为创建视觉单词词汇，我们需要提取特征描述子，这里，使用之前博文中介绍过的SIFT特征描述子。imlist包含的是图像的文件名，运行相应的代码（后文给出），可以得到每幅图的描述子，并且将每幅图像的描述子保存在一个文件中。创建一个Vocabulary类，其中包含了一个由单词聚类中心VOC与每个单词对应的逆向文档频率构成的向量，为了在某些图像集上训练词汇，train()方法获取包含有.sift后缀的描述子文件列表和词汇单词数k.在K-means聚类阶段可以对训练数据下采样，因为如果使用过多特征，会耗费很长时间。现在在计算机的某个文件夹中，保存了图像及提取出来的sift特征文件，利用pickle模块保存整个词汇对象以便后面使用。  
## 三.图像索引
### 1.建立数据库
在索引图像前，我们需要建立一个数据库。对图像进行索引就是从这些图像中提取描述子，利用词汇将描述子转换成视觉单词，并保存视觉单词及对应图像的单词直方图，从而利用图像对数据库进行查询，并返回相似的图像作为搜索结果。  
### 2.添加图像
有了数据库表单，我们可以在索引中添加图像。
## 四.在数据库中搜索图像
建立好图像的索引，我们就可以在数据库中搜索相似的图像了。这里使用BoW（Bag-of-Word,词袋模型）来表示整个图像，不过这里介绍的过程是通用的，可以应用于寻找相似的物体、相似的脸、颜色等，这取决于图像及所用的描述子。如果图像数据库很大，逐一比较整个数据库中的所有直方图往往是不可行的，要找到一个大小合理的候选集，单词索引的作用就是这个：可以利用单词索引获得候选集，然后只需在候选集上进行逐一比较。  
### 1.利用索引获取候选图像
可以利用建立起来的索引找到包含特定单词的所有图像，这不过是对数据库做一次简单的查询。
### 2.用一幅图像进行查询
利用一幅图像进行查询时，没有必要进行完全的搜索。对于每个候选图像，我们用标准的欧氏距离比较它和查询图像间的直方图，并返回一个经排序的包含距离及图像ID的元组列表。
### 3.确定对比基准并绘制结果
为了评价搜索结果好坏，可以计算前4个位置中搜索到的相似图象数。这是在ukbench图像集上评价搜索性能常采用的评价方式。
## 五.使用几何特性对结果排序
Bow模型的一个主要缺点是在用视觉单词表示图像时不包含图像特征的位置信息，这是为获取速度和可伸缩性而付出的代价。利用一些考虑特征集合关系的准则重排搜索到的考前结果，可以提高准确率，最常用的方法是在查询图像与靠前图像的特征位置间拟合单应性。  
为了提高效率，可以将特征位置存储在数据库中，并由特征的单词ID决定它们之间的关联。
## 六.演示程序及Web应用
之前介绍了图像检索的原理与基础步骤，这里将采用上述的知识进行一个整合，给出一个完整的图像搜索引擎。
### 1.所需运行环境及相应配置
代码运行于Python 3平台，需要相应的PCV，之前博文已经给出。  
另外需要安装  
**安装pyqt5  ：pip install PyQt5**  
为了建立演示程序，采用CherryPy包，参见http://www.cherrypy.org 。这是一个纯Python轻量级Web服务器，使用面向对象模型。  

**安装cherrypy : pip install cherrypy**  
所需的图像数据集first1000自行下载。  
### 2.源码及流程  
#### 1)生成所需模型文件
~~~python
# -*- coding: utf-8 -*-
import pickle
from PCV.imagesearch import vocabulary
from PCV.tools.imtools import get_imlist
from PCV.localdescriptors import sift

#获取图像列表
imlist = get_imlist('first1000/')
nbr_images = len(imlist)
#获取特征列表
featlist = [imlist[i][:-3]+'sift' for i in range(nbr_images)]

#提取文件夹下图像的sift特征
for i in range(nbr_images):
    sift.process_image(imlist[i], featlist[i])

#生成词汇
voc = vocabulary.Vocabulary('ukbenchtest')
voc.train(featlist, 1000, 10)
#保存词汇
# saving vocabulary
with open('first1000/vocabulary.pkl', 'wb') as f:
    pickle.dump(voc, f)
print ('vocabulary is:', voc.name, voc.nbr_words)
~~~  
这里用的数据集是之前下载的data文件夹中的 ﬁrst1000 数据集，里面有1000张图片，建议把这个数据集文件整个 放到要运行的代码的当前目录下，比较方便，且不容易出错：这个过程很缓慢，根据机器配置不同，需要10-30分钟时间。  
![image](1.JPG)  
![image](2.JPG)  
#### 2)将模型数据导入数据库
~~~python
# -*- coding: utf-8 -*-
import pickle
from PCV.imagesearch import imagesearch
from PCV.localdescriptors import sift
from sqlite3 import dbapi2 as sqlite
from PCV.tools.imtools import get_imlist

#获取图像列表
imlist = get_imlist('first1000/')
nbr_images = len(imlist)
#获取特征列表
featlist = [imlist[i][:-3]+'sift' for i in range(nbr_images)]

# load vocabulary
#载入词汇
with open('first1000/vocabulary.pkl', 'rb') as f:
    voc = pickle.load(f)
#创建索引
indx = imagesearch.Indexer('testImaAdd.db',voc)
indx.create_tables()
# go through all images, project features on vocabulary and insert
#遍历所有的图像，并将它们的特征投影到词汇上
for i in range(nbr_images)[:1000]:
    locs,descr = sift.read_features_from_file(featlist[i])
    indx.add_to_index(imlist[i],descr)
# commit to database
#提交到数据库
indx.db_commit()

con = sqlite.connect('testImaAdd.db')
print (con.execute('select count (filename) from imlist').fetchone())
print (con.execute('select * from imlist').fetchone())
 

~~~  
运行后，会生成相应文件：   
![image](5.jpg)  
#### 3).索引测试
将数据放进数据库中之后就可以开始测试我们的图片索引。
下面直接上代码：  
~~~python
# -*- coding: utf-8 -*-
import pickle
from PCV.localdescriptors import sift
from PCV.imagesearch import imagesearch
from PCV.geometry import homography
from PCV.tools.imtools import get_imlist

# load image list and vocabulary
#载入图像列表
imlist = get_imlist('first1000/')
nbr_images = len(imlist)
#载入特征列表
featlist = [imlist[i][:-3]+'sift' for i in range(nbr_images)]

#载入词汇
with open('first1000/vocabulary.pkl', 'rb') as f:
    voc = pickle.load(f)

src = imagesearch.Searcher('testImaAdd.db',voc)

# index of query image and number of results to return
#查询图像索引和查询返回的图像数
q_ind = 0
nbr_results = 20

# regular query
# 常规查询(按欧式距离对结果排序)
res_reg = [w[1] for w in src.query(imlist[q_ind])[:nbr_results]]
print ('top matches (regular):', res_reg)

# load image features for query image
#载入查询图像特征
q_locs,q_descr = sift.read_features_from_file(featlist[q_ind])
fp = homography.make_homog(q_locs[:,:2].T)

# RANSAC model for homography fitting
#用单应性进行拟合建立RANSAC模型
model = homography.RansacModel()
rank = {}

# load image features for result
#载入候选图像的特征
for ndx in res_reg[1:]:
    locs,descr = sift.read_features_from_file(featlist[ndx])  # because 'ndx' is a rowid of the DB that starts at 1
    # get matches
    matches = sift.match(q_descr,descr)
    ind = matches.nonzero()[0]
    ind2 = matches[ind]
    tp = homography.make_homog(locs[:,:2].T)
    # compute homography, count inliers. if not enough matches return empty list
    try:
        H,inliers = homography.H_from_ransac(fp[:,ind],tp[:,ind2],model,match_theshold=4)
    except:
        inliers = []
    # store inlier count
    rank[ndx] = len(inliers)

# sort dictionary to get the most inliers first
sorted_rank = sorted(rank.items(), key=lambda t: t[1], reverse=True)
res_geom = [res_reg[0]]+[s[0] for s in sorted_rank]
print ('top matches (homography):', res_geom)

# 显示查询结果
imagesearch.plot_results(src,res_reg[:8]) #常规查询
imagesearch.plot_results(src,res_geom[:8]) #重排后的结果
~~~
执行完后会出现两张图片:  
![image](3.JPG)  
![image](4.JPG)  
#### 4).建立Demo和Web应用  
安装CherryPy包后，直接运行下述代码：  
~~~python
# -*- coding: utf-8 -*-
import cherrypy
import pickle
import urllib
import os
from numpy import *
#from PCV.tools.imtools import get_imlist
from PCV.imagesearch import imagesearch
import random

"""
This is the image search demo in Section 7.6.
"""


class SearchDemo:

    def __init__(self):
        # 载入图像列表
        self.path = 'first1000/'
        #self.path = 'D:/python_web/isoutu/first500/'
        self.imlist = [os.path.join(self.path,f) for f in os.listdir(self.path) if f.endswith('.jpg')]
        #self.imlist = get_imlist('./first500/')
        #self.imlist = get_imlist('E:/python/isoutu/first500/')
        self.nbr_images = len(self.imlist)
        print (self.imlist)
        print (self.nbr_images)
        self.ndx = list(range(self.nbr_images))
        print (self.ndx)

        # 载入词汇
        # f = open('first1000/vocabulary.pkl', 'rb')
        with open('first1000/vocabulary.pkl','rb') as f:
            self.voc = pickle.load(f)
        #f.close()

        # 显示搜索返回的图像数
        self.maxres = 10

        # header and footer html
        self.header = """
            <!doctype html>
            <head>
            <title>Image search</title>
            </head>
            <body>
            """
        self.footer = """
            </body>
            </html>
            """

    def index(self, query=None):
        self.src = imagesearch.Searcher('testImaAdd.db', self.voc)

        html = self.header
        html += """
            <br />
            Click an image to search. <a href='?query='> Random selection </a> of images.
            <br /><br />
            """
        if query:
            # query the database and get top images
            #查询数据库，并获取前面的图像
            res = self.src.query(query)[:self.maxres]
            for dist, ndx in res:
                imname = self.src.get_filename(ndx)
                html += "<a href='?query="+imname+"'>"
                
                html += "<img src='"+imname+"' alt='"+imname+"' width='100' height='100'/>"
                print (imname+"################")
                html += "</a>"
            # show random selection if no query
            # 如果没有查询图像则随机显示一些图像
        else:
            random.shuffle(self.ndx)
            for i in self.ndx[:self.maxres]:
                imname = self.imlist[i]
                html += "<a href='?query="+imname+"'>"
                
                html += "<img src='"+imname+"' alt='"+imname+"' width='100' height='100'/>"
                print (imname+"################")
                html += "</a>"

        html += self.footer
        return html

    index.exposed = True

#conf_path = os.path.dirname(os.path.abspath(__file__))
#conf_path = os.path.join(conf_path, "service.conf")
#cherrypy.config.update(conf_path)
#cherrypy.quickstart(SearchDemo())

cherrypy.quickstart(SearchDemo(), '/', config=os.path.join(os.path.dirname(__file__), 'service.conf'))
~~~  

运行这个代码首先需要配置 service.conf文件：  
![image](6.JPG)  
内容如下：  
~~~
[global] 
server.socket_host = "127.0.0.1" 
server.socket_port = 8080 
server.thread_pool = 50 
tools.sessions.on = True 
[/] 
tools.staticdir.root = "E:/Python/pythonwatch/pcv-book-code-master/ch07" 
tools.staticdir.on = True 
tools.staticdir.dir = ""
~~~
配置文件中的第一部分为IP地址和端口，第二部分为我们的图库的地址,我的图库是在 E:/Study/pythonProject/ch07/ﬁrst1000/ 地址下，然后我在数据库中保存 的路径是 ﬁrst1000/xxx.jpg 所以我只要将图库的地址设置成 E:/Study/pythonProject/ch07 就行。
最后我们运行的时候会将我们设置的图库的地址（也就是 E:/Study/pythonProject/ch07） 和我们保存在数据库中的地址（ﬁrst1000/xxx.jpg）连接起来，用于显示图片。
最后效果类似于这样
打开我们的浏览器，输入相应端口：  
![image](7.jpg)  
就会显示我们的Web搜索引擎：  
![image](8.jpg)  
![image](9.jpg)  
![image](10.jpg)  
第一张图是随机选择了一些图像，下面是两个查询示例：页面内最左边为查询原图，之后是结果靠前的图像。
