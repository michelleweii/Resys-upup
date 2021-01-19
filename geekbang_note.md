https://time.geekbang.org/column/article/295939

## 特征工程

### 04 ###

特征是对某个行为过程相关信息的抽象表达：因为一个行为过程必须转换成某种数学形式才能被机器学习模型所学习。

但也不能所有情节都当做特征放进模型学习。具体的推荐场景中包含大量冗余的、无用的信息，把它们都考虑进来甚至会损害模型的泛化能力。

> 构建推荐系统特征工程的原则：尽可能地让特征工程抽取出的一组特征，能够保留推荐环境及用户行为过程中的所有“有用“信息，并且尽量摒弃冗余信息。



电影推荐的要素和特征化方式

![img](https://static001.geekbang.org/resource/image/af/5d/af921c7e81984281621729f6e75c1b5d.jpeg)



#### 推荐系统中的常用特征 ####

1. 用户行为数据（最常用、最关键）
2. 用户关系数据
3. 属性、标签类数据
4. 内容类数据
5. 场景信息（上下文信息）



1、用户行为数据

用户行为在推荐系统中一般分为显性反馈行为（Explicit Feedback）和隐性反馈行为（Implicit Feedback）两种。

不同业务场景下用户行为数据的例子

![img](https://static001.geekbang.org/resource/image/75/06/7523075958d83e9bd08966b77ea23706.jpeg)

隐性反馈行为更重要，主要原因是显性反馈行为的收集难度过大，数据量小。



2、用户关系数据

物以类聚，人以群分

用户行为数据是人与物之间的“连接”日志；

用户关系数据就是人与人之间连接的记录

用户关系数据也可以分为“显性”和“隐性”两种，或者称为“强关系”和“弱关系”。用户与用户之间可以通过“关注”“好友关系”等连接建立“强关系”，也可以通过“互相点赞”“同处一个社区”，甚至“同看一部电影”建立“弱关系”。

- 比如可以将用户关系作为召回层的一种物品召回方式；（user-cf）
- 也可以通过用户关系建立关系图，使用 Graph Embedding 的方法生成用户和物品的 Embedding；
- 还可以直接利用关系数据，通过“好友”的特征为用户添加新的属性特征；
- 甚至可以利用用户关系数据直接建立社会化推荐系统。



3、属性、标签类数据

推荐系统中另外一大类特征来源是属性、标签类数据，本质上都是直接描述用户或者物品的特征。属性和标签的主体可以是用户，也可以是物品

图：属性、标签类数据的分类和来源

![img](https://static001.geekbang.org/resource/image/ba/69/ba044e0033b513d996633de77e11f969.jpeg)



在推荐系统中使用属性、标签类数据，一般是通过 Multi-hot 编码的方式将其转换成特征向量，一些重要的属性标签类特征也可以先转换成 Embedding，比如业界最新的做法是将标签属性类数据与其描述主体一起构建成知识图谱（Knowledge Graph），在其上施以 Graph Embedding 或者 GNN（Graph Neural Network，图神经网络）生成各节点的 Embedding，再输入推荐模型。



4、内容类数据

相比标签类特征，内容类数据往往是大段的描述型文字、图片，甚至视频。文字信息则更多是通过自然语言处理的方法提取关键词、主题、分类等信息，一旦这些特征被提取出来，就跟处理属性、标签类特征的方法一样，通过 Multi-hot 编码，Embedding 等方式输入推荐系统进行训练。



5、场景信息（上下文信息）

最后一大类是场景信息，或称为上下文信息（Context），它是描述推荐行为产生的场景的信息。最常用的上下文信息是“时间”和通过 GPS、IP 地址获得的“地点”信息。根据推荐场景的不同，上下文信息的范围极广，除了我们上面提到的时间和地点，还包括“当前所处推荐页面”“季节”“月份”“是否节假日”“天气”“空气质量”“社会大事件”等等。





### 05 ###

> 知道了推荐系统要使用的常用特征有哪些。但这些原始的特征是无法直接提供给推荐模型使用的，因为推荐模型本质上是一个函数，输入输出都是数字或数值型的向量。那么问题来了，像动作、喜剧、爱情、科幻这些电影风格，是怎么转换成数值供推荐模型使用的呢？用户的行为历史又是怎么转换成数值特征的呢？



#### spark ####

Spark 是一个分布式计算平台。所谓分布式，指的是计算节点之间不共享内存，需要通过网络通信的方式交换数据。Spark 最典型的应用方式就是建立在大量廉价的计算节点上，这些节点可以是廉价主机，也可以是虚拟的 Docker Container（Docker 容器）。

Spark 程序由 Manager Node（管理节点）进行调度组织，由 Worker Node（工作节点）进行具体的计算任务执行，最终将结果返回给 Drive Program（驱动程序）。在物理的 Worker Node 上，数据还会分为不同的 partition（数据分片），可以说 partition 是 Spark 的基础数据单元。

![img](https://static001.geekbang.org/resource/image/4a/9b/4ae1153e4daee39985c357ed796eca9b.jpeg)





如何利用 One-hot 编码处理类别型特征

广义上来讲，所有的特征都可以分为两大类。

第一类是类别、ID 型特征——影的风格、ID、标签、导演演员等信息，用户看过的电影 ID、用户的性别、地理位置信息、当前的季节、时间（上午，下午，晚上）、天气等等，这些无法用数字表示的信息全都可以被看作是类别、ID 类特征。

第二类是数值型特征——能用数字直接表示的特征就是数值型特征，典型的包括用户的年龄、收入、电影的播放时长、点击量、点击率等。

> 进行特征处理的目的，是把所有的特征全部转换成一个数值型的特征向量。

对于数值型特征，这个过程非常简单，直接把这个数值放到特征向量上相应的维度上就可以了。但是对于类别、ID 类特征，我们应该怎么处理它们呢？



one-hot——将类别、**ID 型特征**转换成数值向量的一种最典型的编码方式

类别转换：

![img](https://static001.geekbang.org/resource/image/94/15/94f78685d98671648638e330a461ab15.jpeg)

id类转换：

ID 型特征也经常使用 One-hot 编码。比如，用户 U 观看过电影 M，这个行为是一个非常重要的用户特征，那我们应该如何向量化这个行为呢？其实也是使用 One-hot 编码。**假设，我们的电影库中一共有 1000 部电影，电影 M 的 ID 是 310（编号从 0 开始），那这个行为就可以用一个 1000 维的向量来表示，让第 310 维的元素为 1，其他元素都为 0。**

One-hot 编码也可以自然衍生成 **Multi-hot 编码（多热编码）**。比如，对于历史行为序列类、标签特征等数据来说，用户往往会与多个物品产生交互行为，或者一个物品被打上多个标签，这时最常用的特征向量生成方式就是把其转换成 Multi-hot 编码。因为每个电影都是有多个 Genre（风格）类别的，所以我们就可以用 Multi-hot 编码完成标签到向量的转换。

> multi-hot e.g. :
>
> 用户行为特征是multi-hot的，即多值离散特征。针对这种特征，由于每个涉及到的非0值个数是不一样的，常见的做法就是将id转换成embedding之后，加一层pooling层，比如average-pooling，sum-pooling，max-pooling。DIN中使用的是weighted-sum，其实就是加权的sum-pooling，权重经过一个activation unit计算得到。



数值型特征的处理 - *归一化和分桶*

一是特征的尺度，二是特征的分布



> 用分桶的方式来解决特征值分布极不均匀的问题。所谓“分桶（Bucketing）”，就是将样本按照某特征的值从高到低排序，然后按照桶的数量找到分位数，将样本分到各自的桶中，再用桶 ID 作为特征值。

怎么理解？比如用户对电影的打分都是3-4之间（5分制）比如[3,3.3)、 [3.3,3.6) 和 [3.6,5]三个区间对应分成三类。这样用户所打的分数就更有区分性。

> **1）分桶。**比如视频一周内被播放次数应该是一个有用的特征，因为播放次数跟视频的热度有很强的相关性，**但是如果不同视频的播放次数跨越不同的数量级，则很难发挥想要的作用。**例如 LR 模型，模型往往只对比较大的特征值敏感。对于这种情况，通常的解决方法是进行分桶。***分桶操作可以看作是对数值变量的离散化，之后通过二值化进行 one-hot 编码。***
>
> 分桶的数量和宽度可以根据业务领域的经验来指定，也有一些常规做法。(1)等距分桶，每个桶的值域是固定的，这种方式适用于样本分布较为均匀的情况；(2)等频分桶，即使得每个桶里数据一样多，这种方式可以保证每个桶有相同的样本数，但也会出现特征值差异非常大的样本被放在一个桶中的情况；(3)模型分桶，使用模型找到最佳分桶，例如利用聚类的方式将特征分成多个类别，或者树模型，这种非线性模型天生具有对连续型特征切分的能力，利用特征分割点进行离散化。
>
> **分桶是离散化的常用方法，将连续特征离散化为一系列 0/1 的离散特征，离散化之后得到的稀疏向量，内积乘法运算速度更快，计算结果方便存储。**离散化之后的特征对于异常数据也具有很强的鲁棒性。需要注意的是：1)要使得桶内的属性取值变化对样本标签的影响基本在一个不大的范围，即不能出现单个分桶的内部，样本标签输出变化很大的情况；2)使每个桶内都有足够的样本，如果桶内样本太少，则随机性太大，不具有统计意义上的说服力；3)每个桶内的样本尽量分布均匀。
>
> ​    常用的行为次数与曝光次数比值类的特征，由于数据的稀疏性，这种计算方式得到的统计量通常具有较大的偏差，需要做**平滑处理**，比如广告点击率常用的贝叶斯平滑技术。而 <u>在我们推荐场景中，也会用到很多统计类特征、比率特征。如果直接使用，比如由于不同item的下发量是不同的，这会让推荐偏向热门的类目，使得越推越窄，无法发现用户的个体差异，也不利于多样性的探索。</u> 常见的有贝叶斯平滑和威尔逊区间平滑等。



- 离散特征：比如西瓜的色泽有 [青绿、乌黑、浅白] 三种颜色；或许有时候也称为属性类;
- 标签类特征：也是离散的类别特征；
- 连续特征：比如西瓜的含糖率有 0.460、0.376、0.263、0.821、0.102 ... 等等等；
- 数值型特征：特征以数字形式表示？
- ID类特征：也是离散特征，电商领域为例，存在大量ID类特征，比如user ID, item ID, product ID, store ID, brand ID和category ID等。



spark分桶的特征处理 QuantileDiscretizer

![img](https://static001.geekbang.org/resource/image/b3/7b/b3b8c959df72ce676ae04bd8dd987e7b.jpeg)



### 06 word2vec ###

Embedding 就是用一个数值向量“表示”一个对象（Object）的方法。对序列数据进行了 Embedding 化。

大量使用 One-hot 编码会导致样本特征向量极度稀疏，而深度学习的结构特点又不利于稀疏特征向量的处理，<u>*因此几乎所有深度学习推荐模型都会由 Embedding 层负责将稀疏高维特征向量转换成稠密低维特征向量。*</u>

![img](https://static001.geekbang.org/resource/image/99/39/9997c61588223af2e8c0b9b2b8e77139.jpeg)

它的输入层和输出层的维度都是 V，这个 V 其实就是语料库词典的大小。假设语料库一共使用了 10000 个词，那么 V 就等于 10000。根据生成的训练样本，这里的输入向量自然就是由输入词转换而来的 One-hot 编码向量，**输出向量则是由多个输出词转换而来的 Multi-hot 编码向量**，显然，基于 Skip-gram 框架的 Word2vec 模型解决的是一个多分类问题。

最后是激活函数的问题，这里我们需要注意的是，隐层神经元是没有激活函数的，或者说采用了输入即输出的恒等函数作为激活函数，而输出层神经元采用了 **<u>*softmax 作为激活函数*</u>**。

为什么要这样设置 Word2vec 的神经网络，以及我们为什么要这样选择激活函数呢？因为这个神经网络其实是为了表达从输入向量到输出向量的这样的一个条件概率关系，我们看下面的式子：

![image-20210118233633279](/Users/michelle/Library/Application Support/typora-user-images/image-20210118233633279.png)

这个由输入词 WI 预测输出词 WO 的条件概率，其实就是 Word2vec 神经网络要表达的东西。我们通过极大似然的方法去最大化这个条件概率，就能够让相似的词的内积距离更接近，这就是我们希望 Word2vec 神经网络学到的。

> 多分类问题：



怎样把词向量从 Word2vec 模型中提取出来？

Embedding 藏在输入层到隐层的权重矩阵 W VxN 中。在训练完成后，模型输入向量矩阵的行向量，就是我们要提取的词向量。

![img](https://static001.geekbang.org/resource/image/0d/72/0de188f4b564de8076cf13ba6ff87872.jpeg)

在实际的使用过程中，我们往往会**把输入向量矩阵转换成词向量查找表（Lookup table）**。例如，输入向量是 10000 个词组成的 One-hot 向量，隐层维度是 300 维，那么输入层到隐层的权重矩阵为 10000x300 维。在转换为词向量 Lookup table 后，每行的权重即成了对应词的 Embedding 向量。如果我们把这个查找表存储到线上的数据库中，就可以轻松地在推荐物品的过程中使用 Embedding 去计算相似性等重要的特征了。

![img](https://static001.geekbang.org/resource/image/1e/96/1e6b464b25210c76a665fd4c34800c96.jpeg)

​                                                                      Word2vec的Lookup table



#### item2vec

既然 Word2vec 可以对词“序列”中的词进行 Embedding，那么对于用户购买“序列”中的一个商品，用户观看“序列”中的一个电影，也应该存在相应的 Embedding 方法。

![img](https://static001.geekbang.org/resource/image/d8/07/d8e3cd26a9ded7e79776dd31cc8f4807.jpeg)

图8 不同场景下的序列数据



### 07 利用图结构数据生成Graph Embedding？

![img](https://static001.geekbang.org/resource/image/54/91/5423f8d0f5c1b2ba583f5a2b2d0aed91.jpeg)

（1）从社交网络中，我们可以发现意见领袖，可以发现社区，再根据这些“社交”特性进行社交化的推荐，如果我们可以对社交网络中的节点进行 Embedding 编码，社交化推荐的过程将会非常方便。

（2）知识图谱中包含了不同类型的知识主体（如人物、地点等），附着在知识主体上的属性（如人物描述，物品特点），以及主体和主体之间、主体和属性之间的关系。如果我们能够对知识图谱中的主体进行 Embedding 化，就可以发现主体之间的潜在关系，这对于基于内容和知识的推荐系统是非常有帮助的。

（3）行为关系类图数据。这类数据几乎存在于所有互联网应用中，它事实上是由用户和物品组成的“二部图”（也称二分图，如图 1c）。用户和物品之间的相互行为生成了行为关系图。借助这样的关系图，我们自然能够利用 Embedding 技术发掘出物品和物品之间、用户和用户之间，以及用户和物品之间的关系，从而应用于推荐系统的进一步推荐。



#### 基于随机游走的 Graph Embedding 方法：Deep Walk

> 它的主要思想是在由物品组成的图结构上进行随机游走，产生大量物品序列，然后将这些物品序列作为训练样本输入 Word2vec 进行训练，最终得到物品的 Embedding。因此，DeepWalk 可以被看作连接序列 Embedding 和 Graph Embedding 的一种过渡方法。

![img](https://static001.geekbang.org/resource/image/1f/ed/1f28172c62e1b5991644cf62453fd0ed.jpeg)

> 我们基于原始的用户行为序列（图 2a），比如用户的购买物品序列、观看视频序列等等，来构建物品关系图（图 2b）。从中，我们可以看出，因为用户 Ui先后购买了物品 A 和物品 B，所以产生了一条由 A 到 B 的有向边。如**果后续产生了多条相同的有向边，则有向边的权重被加强**。在将所有用户行为序列都转换成物品相关图中的边之后，全局的物品相关图就建立起来了。
>
> 采用随机游走的方式随机选择起始点，重新产生物品序列（图 2c）。其中，**随机游走采样的次数、长度等都属于超参数**，需要我们根据具体应用进行调整。
>
> 将这些随机游走生成的物品序列输入图 2d 的 Word2vec 模型，生成最终的物品 Embedding 向量。



DeepWalk 的算法流程中，唯一需要**形式化定义的就是随机游走的跳转概率**，也就是到达节点 vi后，下一步遍历 vi 的邻接点 vj 的概率。如果物品关系图是有向有权图，那么**从节点 vi 跳转到节点 vj** 的概率定义如下：

![image-20210119214748225](/Users/michelle/Library/Application Support/typora-user-images/image-20210119214748225.png)



其中，N+(vi) 是节点 vi所有的出边集合，Mij是节点 vi到节点 vj边的权重，即 DeepWalk 的跳转概率就是跳转边的权重占所有相关出边权重之和的比例。<u>如果物品相关图是无向无权重图，那么跳转概率将是上面这个公式的一个特例，即权重 Mij将为常数 1，且 N+(vi) 应是节点 vi所有“边”的集合，而不是所有“出边”的集合。(咦？那不是所有的边跳转概率都一样？)</u>

- 物品关系图是有向有权图：节点 vi 跳转到节点 vj 的概率定义：i到j的权重比上**i出边**的权重之和。
- 无向无权重图：1/i的所有边的和。



#### 在同质性和结构性间权衡的方法，Node2vec

Node2vec 通过调整随机游走跳转概率的方法，让 Graph Embedding 的结果在网络的同质性（Homophily）和结构性（Structural Equivalence）中进行权衡，可以进一步把不同的 Embedding 输入推荐模型。

结构性和同质性是什么？Graph Embedding 的结果究竟是怎么表达结构性和同质性的呢？

> <u>**“同质性”**指的是距离相近节点的 Embedding 应该尽量近似</u>，如图 3 所示，节点 u 与其相连的节点 s1、s2、s3、s4的 Embedding 表达应该是接近的，这就是网络“同质性”的体现。在电商网站中，同质性的物品很可能是同品类、同属性，或者经常被一同购买的物品。
>
> 为了表达“同质性”，随机游走要更倾向于 **DFS（Depth First Search，深度优先搜索）**才行，因为 DFS 更有可能通过多次跳转，游走到远方的节点上。但无论怎样，DFS 的游走更大概率会在一个大的集团内部进行，这就使得一个集团或者社区内部节点的 Embedding 更为相似，从而更多地表达网络的“同质性”。



> <u>**“结构性”**指的是结构上相似的节点的 Embedding 应该尽量接近</u>，比如图 3 中节点 u 和节点 s6都是各自局域网络的中心节点，它们在结构上相似，所以它们的 Embedding 表达也应该近似，这就是“结构性”的体现。在电商网站中，结构性相似的物品一般是各品类的爆款、最佳凑单商品等拥有类似趋势或者结构性属性的物品。
>
> 为了使 Graph Embedding 的结果能够表达网络的“结构性”，在随机游走的过程中，我们需要让游走的过程更倾向于 **BFS（Breadth First Search，宽度优先搜索）**，因为 BFS 会更多地在当前节点的邻域中进行游走遍历，相当于对当前节点周边的网络结构进行一次“微观扫描”。当前节点是“局部中心节点”，还是“边缘节点”，亦或是“连接性节点”，其生成的序列包含的节点数量和顺序必然是不同的，从而让最终的 Embedding 抓取到更多结构性信息。

![img](https://static001.geekbang.org/resource/image/e2/82/e28b322617c318e1371dca4088ce5a82.jpeg)

​                                                                                       图3 网络的BFS和 DFS示意图



在 Node2vec 算法中，究竟是怎样控制 BFS 和 DFS 的倾向性的呢？

主要是通过节点间的跳转概率来控制跳转的倾向性。图 4 所示为 **Node2vec 算法从节点 t 跳转到节点 v 后，再从节点 v 跳转到周围各点的跳转概率**。这里，你要注意这几个节点的特点。比如，节点 t 是随机游走上一步访问的节点，节点 v 是当前访问的节点，节点 x1、x2、x3是与 v 相连的非 t 节点，但节点 x1还与节点 t 相连，这些不同的特点决定了随机游走时下一次跳转的概率。![img](https://static001.geekbang.org/resource/image/6y/59/6yyec0329b62cde0a645eea8dc3a8059.jpeg)                 

​                                                                                         图4 Node2vec的跳转概率

从当前节点 v 跳转到下一个节点 x 的概率![image-20210119215702542](/Users/michelle/Library/Application Support/typora-user-images/image-20210119215702542.png)，其中 wvx 是边 vx 的原始权重，αpq(t,x) 是 Node2vec 定义的一个跳转权重。**跳转权重决定是倾向于 DFS 还是 BFS**。

![image-20210119215903301](/Users/michelle/Library/Application Support/typora-user-images/image-20210119215903301.png)

![image-20210119215924725](/Users/michelle/Library/Application Support/typora-user-images/image-20210119215924725.png)里的 dtx是指节点 t 到节点 x 的距离，比如节点 x1其实是与节点 t 直接相连的，所以这个距离 dtx就是 1，节点 t 到节点 t 自己的距离 dtt就是 0，而 x2、x3这些不与 t 相连的节点，dtx就是 2。此外，αpq(t,x) 中的参数 p 和 q 共同控制着随机游走的倾向性。**参数 p 被称为返回参数（Return Parameter），p 越小，随机游走回节点 t 的可能性越大，Node2vec 就更注重表达网络的结构性。**  **参数 q 被称为进出参数（In-out Parameter），q 越小，随机游走到远方节点的可能性越大，Node2vec 更注重表达网络的同质性。**反之，当前节点更可能在附近节点游走。你可以自己尝试给 p 和 q 设置不同大小的值，算一算从 v 跳转到 t、x1、x2和 x3的跳转概率。这样一来，应该就不难理解我刚才所说的随机游走倾向性的问题。

| x_1  | dtx=1 | αpq(t,x) = 1   |
| ---- | ----- | -------------- |
| x_2  | dtx=2 | αpq(t,x) = 1/q |
| x_32 | dtx=1 | αpq(t,x) = 1/q |
| t    | dtt=0 | αpq(t,t) = 1/p |

Node2vec 这种灵活表达同质性和结构性的特点也得到了实验的证实，我们可以通过调整 p 和 q 参数让它产生不同的 Embedding 结果。图 5 上就是 Node2vec 更注重同质性的体现，从中我们可以看到，距离相近的节点颜色更为接近，图 5 下则是更注重结构性的体现，其中结构特点相近的节点的颜色更为接近。

![img](https://static001.geekbang.org/resource/image/d2/3a/d2d5a6b6f31aeee3219b5f509a88903a.jpeg)

Node2vec 所体现的网络的同质性和结构性，在推荐系统中都是非常重要的特征表达。**由于 Node2vec 的这种灵活性，以及发掘不同图特征的能力，我们甚至可以把不同 Node2vec 生成的偏向“结构性”的 Embedding 结果，以及偏向“同质性”的 Embedding 结果共同输入后续深度学习网络，以保留物品的不同图特征信息。**



#### Embedding 是如何应用在推荐系统的特征工程中的？

（1）**“直接应用”**最简单，就是在我们得到 Embedding 向量之后，直接利用 Embedding 向量的相似性实现某些推荐系统的功能。典型的功能有，利用物品 Embedding 间的相似性实现相似物品推荐，利用 *物品 Embedding 和用户 Embedding 的相似性实现 “猜你喜欢 ”* 等经典推荐功能，还可以利用物品 Embedding 实现推荐系统中的召回层等。

（2）**“预训练应用”**指的是在我们预先训练好物品和用户的 Embedding 之后，不直接应用，而是把这些 Embedding 向量作为特征向量的一部分，跟其余的特征向量拼接起来，作为推荐模型的输入参与训练。这样做能够更好地把其他特征引入进来，让推荐模型作出更为全面且准确的预测。

（3）**“End2End 应用”**，也就是端到端训练，就是指我们不预先训练 Embedding，而是把 Embedding 的训练与深度学习推荐模型结合起来，采用统一的、端到端的方式一起训练，直接得到包含 Embedding 层的推荐模型。这种方式非常流行，比如图 6 就展示了三个包含  Embedding 层的经典模型，分别是微软的 Deep Crossing，UCL 提 出的 FNN 和 Google 的 Wide&Deep。

![img](https://static001.geekbang.org/resource/image/e9/78/e9538b0b5fcea14a0f4bbe2001919978.jpg)

​                                                                 图6 带有Embedding层的深度学习模型

总结

- Deep Walk ：首先，我们基于原始的用户行为序列来构建物品关系图，然后采用随机游走的方式随机选择起始点，重新产生物品序列，最后将这些随机游走生成的物品序列输入 Word2vec 模型，生成最终的物品 Embedding 向量。

- Node2vec 相比于 Deep Walk，增加了随机游走过程中跳转概率的倾向性。如果倾向于宽度优先搜索，则 Embedding 结果更加体现“结构性”。如果倾向于深度优先搜索，则更加体现“同质性”。

![img](https://static001.geekbang.org/resource/image/d0/e6/d03ce492866f9fb85b4fbf5fa39346e6.jpeg)



### 08 如何使用Spark生成Item2vec和Graph Embedding？

Item2vec 是基于自然语言处理模型 Word2vec 提出的，所以 Item2vec 要处理的是类似文本句子、观影序列之类的序列数据。那在真正开始 Item2vec 的训练之前，还要先为它准备好训练用的序列数据。

在 MovieLens 数据集中，有一张叫 rating（评分）的数据表，里面包含了用户对看过电影的评分和评分的时间。

数据处理：只放用户打分比较高的电影（因为item2vec 模型本质上是要学习到物品之间的近似性。既然这样，我们当然是希望评分好的电影靠近一些，评分差的电影和评分好的电影不要在序列中结对出现。）

先过滤掉他评分低的电影，再把他评论过的电影按照时间戳排序。这样，我们就得到了一个用户的观影序列，所有用户的观影序列就组成了 Item2vec 的训练样本集。

1. 读取 ratings 原始数据到 Spark 平台；
2. 用 where 语句过滤评分低的评分记录；
3. 用 groupBy userId 操作聚合每个用户的评分记录，DataFrame 中每条记录是一个用户的评分序列；
4. 定义一个自定义操作 sortUdf，用它实现每个用户的评分记录按照时间戳进行排序；
5. 把每个用户的评分记录处理成一个字符串的形式，供后续训练过程使用。

```scala
def processItemSequence(sparkSession: SparkSession): RDD[Seq[String]] ={
  //设定rating数据的路径并用spark载入数据
  val ratingsResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
  val ratingSamples = sparkSession.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)


  //实现一个用户定义的操作函数(UDF)，用于之后的排序
  val sortUdf: UserDefinedFunction = udf((rows: Seq[Row]) => {
    rows.map { case Row(movieId: String, timestamp: String) => (movieId, timestamp) }
      .sortBy { case (movieId, timestamp) => timestamp }
      .map { case (movieId, timestamp) => movieId }
  })


  //把原始的rating数据处理成序列数据
  val userSeq = ratingSamples
    .where(col("rating") >= 3.5)  //过滤掉评分在3.5一下的评分记录
    .groupBy("userId")            //按照用户id分组
    .agg(sortUdf(collect_list(struct("movieId", "timestamp"))) as "movieIds")     //每个用户生成一个序列并用刚才定义好的udf函数按照timestamp排序
    .withColumn("movieIdStr", array_join(col("movieIds"), " "))
                //把所有id连接成一个String，方便后续word2vec模型处理


  //把序列数据筛选出来，丢掉其他过程数据
  userSeq.select("movieIdStr").rdd.map(r => r.getAs[String]("movieIdStr").split(" ").toSeq)
```

ID 为 11888 用户的观影序列：296 380 344 588 593 231 595 318 480 110 253 288 47 364 377 589 410 597 539 39 160 266 350 553 337 186 736 44 158 551 293 780 353 368 858

Item2vec：模型训练

```scala

def trainItem2vec(samples : RDD[Seq[String]]): Unit ={
    //设置模型参数
    val word2vec = new Word2Vec()
    .setVectorSize(10) // 用于设定生成的 Embedding 向量的维度
    .setWindowSize(5)  // 用于设定在序列数据上采样的滑动窗口大小
    .setNumIterations(10)  // 用于设定训练时的迭代次数


  //训练模型
  val model = word2vec.fit(samples)


  //训练结束，用模型查找与item"592"最相似的20个item
  val synonyms = model.findSynonyms("592", 20)
  for((synonym, cosineSimilarity) <- synonyms) {
    println(s"$synonym $cosineSimilarity")
  }
 
  //保存模型
  val embFolderPath = this.getClass.getResource("/webroot/sampledata/")
  val file = new File(embFolderPath.getPath + "embedding.txt")
  val bw = new BufferedWriter(new FileWriter(file))
  var id = 0
  //用model.getVectors获取所有Embedding向量
  for (movieId <- model.getVectors.keys){
    id+=1
    // 调用 getVectors 接口就可以提取出某个电影 ID 对应的 Embedding 向量
    bw.write( movieId + ":" + model.getVectors(movieId).mkString(" ") + "\n")
  }
  bw.close()
```





Graph embedding

数据准备：最关键数据是物品之间的转移概率矩阵。转移概率矩阵表达了下图中的物品关系图，它定义了随机游走过程中，从物品 A 到物品 B 的跳转概率。

![image-20210119230143875](/Users/michelle/Library/Application Support/typora-user-images/image-20210119230143875.png)

> 生成转移概率矩阵的函数输入是在训练 Item2vec 时处理好的观影序列数据。
>
> 输出的是转移概率矩阵。
>
> 由于转移概率矩阵比较稀疏，没有采用比较浪费内存的二维数组的方法，而是采用了一个双层 Map 的结构去实现它。比如说，**要得到物品 A 到物品 B 的转移概率，那么 transferMatrix(itemA)(itemB) 就是这一转移概率。**

```scala
// Spark 生成转移概率矩阵

//samples 输入的观影序列样本集
def graphEmb(samples : RDD[Seq[String]], sparkSession: SparkSession): Unit ={
  //通过flatMap操作把观影序列打碎成一个个影片对
  val pairSamples = samples.flatMap[String]( sample => {
    var pairSeq = Seq[String]()
    var previousItem:String = null
    sample.foreach((element:String) => {
      if(previousItem != null){
        pairSeq = pairSeq :+ (previousItem + ":" + element)
      }
      previousItem = element
    })
    pairSeq
  })
  //统计影片对的数量(再利用 countByValue 操作统计这些影片对的数量)
  val pairCount = pairSamples.countByValue()
  //转移概率矩阵的双层Map数据结构
  val transferMatrix = scala.collection.mutable.Map[String, scala.collection.mutable.Map[String, Long]]()
  val itemCount = scala.collection.mutable.Map[String, Long]()

// ***
  // 根据这些影片对的数量求取每两个影片之间的转移概率
  *** //
  
  //求取转移概率矩阵
  pairCount.foreach( pair => {
    val pairItems = pair._1.split(":")
    val count = pair._2
    lognumber = lognumber + 1
    println(lognumber, pair._1)


    if (pairItems.length == 2){
      val item1 = pairItems.apply(0)
      val item2 = pairItems.apply(1)
      if(!transferMatrix.contains(pairItems.apply(0))){
        transferMatrix(item1) = scala.collection.mutable.Map[String, Long]()
      }


      transferMatrix(item1)(item2) = count
      itemCount(item1) = itemCount.getOrElse[Long](item1, 0) + count
    }
  

```

Graph Embedding：随机游走采样过程

> **随机游走采样的过程是利用转移概率矩阵生成新的序列样本的过程。**这怎么理解呢？首先，我们要根据物品出现次数的分布随机选择一个起始物品，之后就进入随机游走的过程。在每次游走时，我们根据转移概率矩阵查找到两个物品之间的转移概率，然后根据这个概率进行跳转。比如当前的物品是 A，从转移概率矩阵中查找到 A 可能跳转到物品 B 或物品 C，转移概率分别是 0.4 和 0.6，那么我们就按照这个概率来随机游走到 B 或 C，依次进行下去，直到样本的长度达到了我们的要求。

```scala

//随机游走采样函数
//transferMatrix 转移概率矩阵
//itemCount 物品出现次数的分布
def randomWalk(transferMatrix : scala.collection.mutable.Map[String, scala.collection.mutable.Map[String, Long]], itemCount : scala.collection.mutable.Map[String, Long]): Seq[Seq[String]] ={
  //样本的数量
  val sampleCount = 20000
  //每个样本的长度
  val sampleLength = 10
  val samples = scala.collection.mutable.ListBuffer[Seq[String]]()
  
  //物品出现的总次数
  var itemTotalCount:Long = 0
  for ((k,v) <- itemCount) itemTotalCount += v


  //随机游走sampleCount次，生成sampleCount个序列样本
  for( w <- 1 to sampleCount) {
    samples.append(oneRandomWalk(transferMatrix, itemCount, itemTotalCount, sampleLength))
  }


  Seq(samples.toList : _*)
}


//通过随机游走产生一个样本的过程
//transferMatrix 转移概率矩阵
//itemCount 物品出现次数的分布
//itemTotalCount 物品出现总次数
//sampleLength 每个样本的长度
def oneRandomWalk(transferMatrix : scala.collection.mutable.Map[String, scala.collection.mutable.Map[String, Long]], itemCount : scala.collection.mutable.Map[String, Long], itemTotalCount:Long, sampleLength:Int): Seq[String] ={
  val sample = scala.collection.mutable.ListBuffer[String]()


  //决定起始点
  val randomDouble = Random.nextDouble()
  var firstElement = ""
  var culCount:Long = 0
  //根据物品出现的概率，随机决定起始点
  breakable { for ((item, count) <- itemCount) {
    culCount += count
    if (culCount >= randomDouble * itemTotalCount){
      firstElement = item
      break
    }
  }}


  sample.append(firstElement)
  var curElement = firstElement
  //通过随机游走产生长度为sampleLength的样本
  breakable { for( w <- 1 until sampleLength) {
    if (!itemCount.contains(curElement) || !transferMatrix.contains(curElement)){
      break
    }
    //从curElement到下一个跳的转移概率向量
    val probDistribution = transferMatrix(curElement)
    val curCount = itemCount(curElement)
    val randomDouble = Random.nextDouble()
    var culCount:Long = 0
    //根据转移概率向量随机决定下一跳的物品
    breakable { for ((item, count) <- probDistribution) {
      culCount += count
      if (culCount >= randomDouble * curCount){
        curElement = item
        break
      }
    }}
    sample.append(curElement)
  }}
  Seq(sample.toList : _

```

通过随机游走产生了我们训练所需的 sampleCount 个样本之后，下面的过程就和 Item2vec 的过程完全一致了，就是把这些训练样本输入到 Word2vec 模型中，完成最终 Graph Embedding 的生成。

总结

![img](https://static001.geekbang.org/resource/image/02/a7/02860ed1170d9376a65737df1294faa7.jpeg)





未完。。。

scala代码这里没有很好理解。