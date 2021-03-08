# 函数式编程与面向对象编程的区别

- 面向对象调用：对象.方法（参数）
- 函数式调用：函数（参数）





SparkContext:  spark应用程序的主要入口

sparksession：用于操作dataframe的入口点

查看结果用rdd.collect()



什么时候用下划线_来操作？

如果参数是按照顺序只使用了一次的话，参数可以用下划线代替。



# spark 运行环境



## local模式

**就是不需要其他任何节点资源就可以在本地执行 Spark 代码的环境**，一般用于教学，调试，演示等；在 IDEA 中运行代码的环境我们称之为开发环境，不太一样。

```shell
bin/spark-submit \
--class org.apache.spark.examples.SparkPi \
--master local[2] \
./examples/jars/spark-examples_2.12-3.0.0.jar \
10
```

1) --class 表示要执行程序的主类，此处可以更换为咱们自己写的应用程序;
2) **--master local[2] 部署模式，默认为本地模式，数字表示分配的虚拟 CPU 核数量**;
3) spark-examples_2.12-3.0.0.jar 运行的应用类所在的 jar 包， 实际使用时，可以设定为咱
们自己打的 jar 包;
4) 数字 10 表示程序的入口参数，用于设定当前应用的任务数量。

- **-- master表示的是环境的意思！**

## Standalone 模式

- local 本地模式毕竟只是用来进行练习演示的，真实工作中还是要将应用提交到对应的
  集群中去执行

- 只使用 Spark 自身节点运行的集群模式，也就是我们所谓的独立部署（Standalone）模式。 

- Spark 的 Standalone 模式体现了经典的 master-slave 模式。



参数说明

![image-20210308184540643](/Users/michelle/Library/Application Support/typora-user-images/image-20210308184540643.png)

- --executor-memory 计算节点，内存越大，计算能力越强；

## yarn模式

独立部署（Standalone）模式由 Spark 自身提供计算资源，无需其他框架提供资源。 这
种方式降低了和其他第三方资源框架的耦合性，独立性非常强。但是你也要记住， Spark 主
要是计算框架，而不是资源调度框架，所以本身提供的资源调度并不是它的强项，所以还是
和其他专业的资源调度框架集成会更靠谱一些。 所以接下来我们来学习在强大的 Yarn 环境
下 Spark 是如何工作的（其实是因为在国内工作中， Yarn 使用的非常多） 。

> 总结：standalone模式由spark本身提供计算资源，但是spark是计算框架，一般采用yarn环境作为资源调度框架。

提交应用 

```shell
bin/spark-submit \
--class org.apache.spark.examples.SparkPi \
--master yarn \
--deploy-mode client \ # 注意这里是client，这样就能看到输出的结果了。
./examples/jars/spark-examples_2.12-3.0.0.jar \
10
```



课程P18，ppt 22页，windows环境安装。

## 部署模式对比

![image-20210308193332023](/Users/michelle/Library/Application Support/typora-user-images/image-20210308193332023.png)







# 4、spark运行架构

## 4.1 运行架构

Spark 框架的核心是一个计算引擎，整体来说，它采用了标准 master-slave 的结构。
如下图所示，它展示了一个 Spark 执行时的基本结构。 图形中的 Driver 表示 master，
负责管理整个集群中的作业任务调度。图形中的 Executor 则是 slave，负责实际执行任务。

![image-20210308193646183](/Users/michelle/Library/Application Support/typora-user-images/image-20210308193646183.png)

## 4.2 核心组件

对于 Spark 框架有两个核心组件：

### 4.2.1 Driver

Spark 驱动器节点，用于执行 Spark 任务中的 main 方法，**负责实际代码的执行工作。**
Driver 在 Spark 作业执行时主要负责：
➢ 将用户程序转化为作业（job）
➢ 在 Executor 之间调度任务(task)
➢ 跟踪 Executor 的执行情况
➢ 通过 UI 展示查询运行情况
实际上，我们无法准确地描述 Driver 的定义，因为在整个的编程过程中没有看到任何有关
Driver 的字眼。**所以简单理解，所谓的 Driver 就是驱使整个应用运行起来的程序，也称之为**
**Driver 类。**

### 4.2.2 Executor

Spark Executor 是集群中工作节点（Worker）中的一个 JVM 进程，负责在 Spark 作业中运行具体任务（Task） ，任务彼此之间相互独立。 Spark 应用启动时， Executor 节点被同时启动，并且始终伴随着整个 Spark 应用的生命周期而存在。如果有 Executor 节点发生了故障或崩溃， Spark 应用也可以继续执行，会将出错节点上的任务调度到其他 Executor 节点上继续运行。





p19

