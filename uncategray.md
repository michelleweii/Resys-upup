- gbdt+lr模型部署，只需要部署gbdt即可。gbdt导出的特征交叉数据直接传给java端。
  权值w和bias是不变的。直接wx+b即可。

训练的时候这两个模型都要训练。

- logits如何理解？

softmax的目的是把logits映射到0,1之间，因此logits可以理解为原生概率；

