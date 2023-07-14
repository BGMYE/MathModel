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
    # C_z = 10000.  # 直线阻尼器
    z1s, z2s, z3s, u1s, u2s, u3s, ts = [], [], [], [], [], [], []

    while t <= T:
        C_z = (abs(u1 - u2) ** 0.5) * 10000.
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
        z1s.append(z1)
        z2s.append(z2)
        u1s.append(u1)
        u2s.append(u2)
        u3s.append(u1-u2)
        z3s.append(z1-z2)
        ts.append(t)


    plt.figure(figsize=(14, 4), dpi=100)
    ax1 = plt.subplot(1, 2, 1)
    ax2 = ax1.twinx()
    M1, = ax1.plot(ts, z1s, c='b')
    M2, = ax2.plot(ts, z2s, c='g')
    M5, = ax1.plot(ts, z3s, c='r')
    plt.legend(handles=[M1, M2, M5, ], labels=["振子", "浮子", "振子与浮子差"], loc="best")
    ax1.set_xlabel("时间(s)")
    ax1.set_ylabel("位移（m）")

    ax3 = plt.subplot(1, 2, 2)
    ax4 = ax3.twinx()
    M3, = ax3.plot(ts, u1s, c='b')
    M4, = ax4.plot(ts, u2s, c='g')
    M6, = ax1.plot(ts, u3s, c='r')
    plt.legend(handles=[M3, M4, M6, ], labels=["振子", "浮子", "振子与浮子差"], loc="best")
    ax3.set_xlabel("时间(s)")
    ax3.set_ylabel("速度（m/s）")
    plt.show()

    print(z1s[100000+1],z1s[200000+1])
