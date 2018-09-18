"""
[ 딥러닝 기초 ]
단순 선형회귀 그래프의 학습 과정을 시각화한 예제입니다.
Author: Leegeunhyeok
"""

# pip install matplotlib numpy
import matplotlib.pyplot as plt
import numpy as np
import random

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
linear_x = np.linspace(0, 1, 100)
linear_y = np.linspace(0, 1, 100)

# Interactive mode on
plt.ion()

# 컨테이너 가져오기
fig = plt.figure("Deep Learning")

# 그래프 추가
data = fig.add_subplot(221, title="Sample DataSet")
plt.xlabel("X (Input)")
plt.ylabel("Y (Output)")

cost = fig.add_subplot(222, title="Cost Function")
plt.xlabel("E (Error)")
plt.ylabel("C (Cost)")

linear = fig.add_subplot(223, title="Linear egression")
status = fig.add_subplot(224, title="Status")

# 데이터 그래프
data.scatter(x_data, y_data, s=10, color="g")

# Cost 함수 그래프 및 텍스트
cost_plot = cost.plot(cost_x, cost_y, color="g")
cost.text(0, 2000, r'$COST=\sqrt{\frac{1}{n}\sum_{i=0}^n (t_i - y_i)^2}$', horizontalalignment="center", verticalalignment="top")

# 선형회귀 그래프
linear_plot, = linear.plot(linear_x, linear_y, color="b")

# 학습 상태 그래프
status_plot = status.scatter(x_data, y_data, s=10, color="r")
status_linear = status.plot(linear_x, linear_y, color="b")

# 레이아웃 맞춤
fig.tight_layout()

# 데이터 실시간 변경 (임시)
for phase in np.linspace(0, 10 * np.pi, 500):
    linear_plot.set_ydata(linear_x + phase)
    fig.canvas.draw()
    fig.canvas.flush_events()