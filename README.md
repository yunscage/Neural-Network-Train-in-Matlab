# Neural-Network-Train-in-Matlab
This is a self-coded basic network training program in Matlab 2023b. You can add your ideas to improve it.
The basic structure is as follows:
1.net construction. using "dlnetwork" function of Matlab.
    Prenet=dlnetwork(Mylayers);
2.Training process. 
  2.1 add a monitor for viewable.
  2.2 Define "modelGradients" function for forwarding. Calculate loss and gradients.
  2.3 Use "adamupdate" function to update the gradients.
3. GPU acceleration.
  If GPU is supported in your computer, keep the code unchanged.


My contact email: fanyuns@csu.edu.cn Or 15111046347@163.com


这是一个自编的基础网络训练程序，基于Matlab 2023b。你可以添加自己的想法来改进它。

基本结构如下：

1.网络构建：使用Matlab的“dlnetwork”函数。 Prenet=dlnetwork(Mylayers);
2.训练过程。 
    2.1 添加一个监视器以便查看。 
    2.2 定义“modelGradients”函数进行前向传播，计算损失和梯度。 
    2.3 使用“adamupdate”函数更新梯度。
3.GPU加速。 如果你的电脑支持GPU，则无需修改代码。

联系邮箱：fanyuns@csu.edu.cn 或 15111046347@163.com
