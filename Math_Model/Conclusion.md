# python浮点数
- [x] 整数后面需要加.将整数转换成float
- [x] 浮点数相加时尽量避免两个小数加一个大数
- [x] 避免float 和 int 相加
  
# 四阶龙格库塔
对于一般的一阶微分方程$f^{'}(t,y) , y(t_0)=y_0$的利用下列方程

$$
\begin{gathered}
k_1=f\left(t_n, y_n\right) \\
k_2=f\left(t_n+\frac{h}{2}, y_n+\frac{h}{2} k_1\right) \\ k_3=f\left(t_n+\frac{h}{2}, y_n+\frac{h}{2} k_2\right) \\ k_4=f\left(t_n+h, y_n+h k_3\right)\\
y_{n+1}=y_n+\frac{h}{6}\left(k_1+2 k_2+2 k_3+k_4\right) 
\end{gathered}
$$

同理对于一阶的微分方程组，形如$Y^{'} = F(Y,t) , Y(t_0)=Y_0$，只需要将上面的替换成向量的形式即可


