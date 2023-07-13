import math
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# 龙格库塔

def func(u1, u2, z1, z2, t):
    a = u1
    b = u2
    c = (-C_z * (u1 - u2) - k_t * (z1 - z2)) / m1
    d = (f_w * np.cos(w * t) + C_z * (u1 - u2) + k_t * (z1 - z2) - C_p * u2 - rho * g * np.pi * z2) / S

    return a, b, c, d


if __name__ == '__main__':
    t = 0.  # 自变量初始值
    w = 1.4005  # 入射频率
    T = 2 * np.pi / w * 40  # 自变量终值
    u1 = 0.  # 因变量初始值
    u2 = 0.  # 因变量初始值
    z1 = 0.  # 因变量初始值
    z2 = 0.  # 因变量初始值
    dt = 1e-4  # 步长
    m1 = 2433.  # 振子质量
    m2 = 4866.  # 浮子质量
    m3 = 1335.535  # 附加质量
    S = 6201.535
    k_t = 80000.  # 弹簧刚度
    rho = 1025.  # 海水密度
    g = 9.8  # 重力加速度
    C_p = 656.3616  # 阻尼系数
    f_w = 6250.  # 垂荡激励力振幅
    C_z = 10000.  # 直线阻尼器
    z1s, z2s, u1s, u2s, ts = [], [], [], [], []

    while t <= T:
        k1 = func(u1, u2, z1, z2, t)
        k2 = func(u1 + dt / 2. * k1[2],
                  u2 + dt / 2. * k1[3],
                  z1 + dt / 2. * k1[0],
                  z2 + dt / 2. * k1[1],
                  t + dt / 2.)
        k3 = func(u1 + dt / 2. * k2[2],
                  u2 + dt / 2. * k2[3],
                  z1 + dt / 2. * k2[0],
                  z2 + dt / 2. * k2[1],
                  t + dt / 2.)
        k4 = func(u1 + dt * k3[2], u2 + dt * k3[3],
                  z1 + dt * k3[0], z2 + dt * k3[1], t + dt)

        z1 += dt/6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        z2 += dt/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        u1 += dt/6 * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        u2 += dt/6 * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

        t += dt
        z1s.append(np.float64(z1))
        z2s.append(z2)
        u1s.append(np.float64(u1))
        u2s.append(np.float64(u2))
        ts.append(t)


    plt.plot(ts, z1s, label='振子', c='r')
    plt.plot(ts, z2s, label='浮子', c='b')
    plt.legend()
    plt.show()

    print(z2s[100000 - 1], z2s[200000 - 1])

