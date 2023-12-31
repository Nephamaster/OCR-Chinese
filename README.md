# **OCR-Chinese**

本项目是对论文 **Robust Scene Text Recognition with Automatic Rectification** ( https://arxiv.org/abs/1603.03915v2 ) 的具体实现

## **汉字文本识别模型**

+ 能够识别**黑体楷体宋体汉字**、**数字**、**英文字符**、所有**标点符号**以及**特殊符号（@#$%&^等）**
+ 模型对于绝大部分**不规则的文本图像**（图片亮度不足、清晰度较低、文本倾斜或弯曲）的识别效果较好；对于现实情境下的文本识别有较强的鲁棒性
+ 准确率（平均）：**93%**
+ 置信度（平均）：**0.587**

## **网络框架**

**Spatial Transformer Network ( STN ) + Sequence Recognition Network ( SRN )**

+ **STN**:
  1. **Localization Network**：CNN 网络，预测**基准点集**；
  2. **Grid Generator**：网格生成器，利用 **TPS 变换**生成**采样网格**；
  3. **Smapler**： 利用采样网格将输入的带文本图像校正为较规则的文本图像；
  4. **技术要点**：**TPS ( Thin-Plate-Spline ) 变换**：变换矩阵(T)存储 TPS 参数，将输入图像的各点坐标映射至基准坐标系下的**规范点坐标**；由于是矩阵乘法，自然可以传播梯度；

+ **SRN**
  1. **Encoder**：CRNN 网络，对输入图像生成特征向量序列 ( **CNN+BLSTM** )；
  2. **Decoder**：**GRU + Attention**，每一步解码由 attention 机制决定，循环生成一个基于输入序列的字符序列
  3. STN 和 SRN 的训练是一体贯通的，即 **STN 没有对基准点的 label**，STN 接收从 SRN 反向传播回的梯度，对基准点的位置进行调整
