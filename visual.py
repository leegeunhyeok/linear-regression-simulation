"""
[ 딥러닝 기초 ]
단순 선형회귀 그래프의 학습 과정을 시각화한 예제입니다.
Author: Leegeunhyeok
"""

# pip install matplotlib numpy
import matplotlib.pyplot as plt
import numpy as np
import math
import random

def hypothesis(a, x, b=0):
    return a * x + b

def cost_function(a, x_data, y_data):
    x_len = len(x_data)
    y_len = len(y_data)
    if x_len != y_len:
        raise ValueError("X 데이터와 Y 데이터의 수가 일치하지 않습니다.")

    s = 0
    for i in range(x_len):
        s += (hypothesis(a, x_data[i]) - y_data[i]) ** 2

    return s / x_len


# 임시 x, y 데이터 생성
# 1 ~ 100
x_data = [n for n in range(1, 101)]
# n = 1 ~ 100, y 범위 n ~ n +40
y_data = [random.choice(list(range(n, n + 40))) for n in range(1, 101)]


# -50 ~ 50
cost_x = [n for n in range(-50, 51)]
# n^2
cost_y = [n ** 2 for n in range(-50, 51)]

# 선형회귀 그래프
# f(x) = ax + b
a = 1.0
da = hypothesis(a, 2) - hypothesis(a, 1)

# 선형회귀 그래프
linear_x = [n for n in range(1, 101)]
linear_y = np.linspace(1, da * 100, 100)

# Interactive mode on
plt.ion()

# 컨테이너 가져오기
fig = plt.figure("Deep Learning", figsize=(8, 6))


# 선형회귀 그래프
linear = fig.add_subplot(221, title="Hypothesis")
linear.text(50, 90, r"$H(x)=ax + b$", horizontalalignment="center", verticalalignment="top")
linear_text = linear.text(50, 70, r"$a = 1, b = 0$", horizontalalignment="center", verticalalignment="top")
linear_plot, = linear.plot(linear_x, linear_y, color="b")


# Cost 함수 그래프 및 텍스트
cost = fig.add_subplot(222, title="Cost Function")
plt.xlabel("E (Error)")
plt.ylabel("C (Cost)")
cost_plot, = cost.plot(cost_x, cost_y, color="g")
cost_marker = cost.scatter([0], [0], s=20, color="r")
cost.text(0, 2000, r"$cost(a, b)=\frac{1}{n}\sum_{i=1}^n (H(x_i) - y_i)^2$", horizontalalignment="center", verticalalignment="top")
cost_text = cost.text(0, 1000, r"$cost=0$", horizontalalignment="center", verticalalignment="top")


# 임시 데이터 그래프
data = fig.add_subplot(223, title="Sample DataSet")
plt.xlabel("X")
plt.ylabel("Y")
data.scatter(x_data, y_data, s=10, color="g")


# 학습 상태 그래프
status = fig.add_subplot(224, title="Status")
status.scatter(x_data, y_data, s=10, color="r")
status_linear, = status.plot(linear_x, linear_y, color="b")


# 레이아웃 맞춤
fig.tight_layout()

# 데이터 실시간 변경 (임시)
for phase in np.linspace(0, 1e-02, 500):
    # 1e-02 만큼 증가
    a += phase

    # Y 데이터 계산
    linear_y = np.linspace(1, a * 100, 100)

    # 오차비용 구하기
    c = cost_function(a, x_data, y_data)
    print("cost: {}, a: {}, x: {}".format(c, a, math.sqrt(c)))

    # 텍스트 및 그래프 데이터 업데이트
    linear_text.set_text(r"$a = {:.3f}, b = 0$".format(a))
    linear_plot.set_ydata(linear_y)
    status_linear.set_ydata(linear_y)
    cost_marker.set_offsets((math.sqrt(c), c))
    cost_text.set_text(r"$cost={:.3f}$".format(c))
    fig.canvas.draw()
    fig.canvas.flush_events()