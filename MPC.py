import numpy as np
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt


# 연속 상태방정식의 행렬

def mpcgain(Ap, Bp, Cp, Nc, Np):
    # Augmented discrete
    [m1, n1] = Cp.shape
    [n1, n_in] = Bp.shape

    A_e = np.eye(n1 + m1)
    A_e[0:n1, 0:n1] = Ap
    A_e[n1:n1 + m1, 0:n1] = Cp @ Ap

    B_e = np.zeros((n1 + m1, n_in))
    B_e[0:n1, :] = Bp
    B_e[n1:n1 + m1, :] = Cp @ Bp

    C_e = np.zeros((m1, n1 + m1))
    C_e[:, n1:n1 + m1] = np.eye(m1)

    # Predictive model

    h = np.zeros((Np, C_e.shape[1]))
    F = np.zeros((Np, C_e.shape[1]))

    h[0, :] = C_e
    F[0, :] = C_e @ A_e

    for kk in range(1, Np):
        h[kk, :] = h[kk - 1, :] @ A_e
        F[kk, :] = F[kk - 1, :] @ A_e

    v = h @ B_e
    Phi = np.zeros((Np, Nc))
    Phi[:, 0] = v.flatten()

    for i in range(1, Nc):
        if i < Np:
            Phi[:, i] = np.concatenate((np.zeros((i, 1)), v[0:Np - i])).flatten()

    BarRs = np.ones((Np, 1))
    Phi_Phi = Phi.T @ Phi
    Phi_F = Phi.T @ F
    Phi_R = Phi.T @ BarRs

    return Phi_Phi, Phi_F, Phi_R, A_e, B_e, C_e, Phi


# MPC_gain

# Np, 예측하려는 결과의 개수
Np = 10
# Nc, 예측하려는 입력 제어의 개수
Nc = 4
# W_con, Weighting matrix 가중치
W_con = 0
# 사인 레퍼런스 유무
sinwave = 1


# 저항
R = 8.4
# 현재 토크
kt = 0.042
# Back-emf constant
km = 0.042
# 인덕턴스
L = 1.16
# 2차 모멘트
J = 2.09 * (10 ** (5))
# 점성계수
B = 0

# 연속 방정식
Ac = np.array([[0, 1, 0], [0, -B / J, kt / J], [0, -km / L, -R / L]])
Bc = np.array([[0], [0], [1 / L]])
Cc = np.array([[1, 0, 0]])
Dc = np.zeros((1, 1))

dt = 1

# 연속 => 이산화 함수
sys_p = cont2discrete((Ac, Bc, Cc, Dc), dt)

Ap, Bp, Cp, Dp = sys_p[0], sys_p[1], sys_p[2], sys_p[3]

[m1, n1] = Cp.shape
[n1, n_in] = Bp.shape



Phi_Phi, Phi_F, Phi_R, A_e, B_e, C_e, Phi = mpcgain(Ap, Bp, Cp, Nc, Np)
[n, n_in] = B_e.shape

# 모델의 초기값
xm = np.zeros((n1 + m1 - 1, 1))
# 피드백 받는 정보의 초기값, 출력에 대한 초기값, 변화량의 초기값
Xf = np.zeros((n1 + m1, 1))
N_sim = 100

# Reference
r = 10 * np.ones((N_sim, 1))

u = 0
y = 0

ref = np.ones((N_sim, 1))

########사인파 reference###########
if sinwave == 1:
    sineref = np.ones((N_sim, 1))
    for kk in range(N_sim):
        for k in range(N_sim):
            ref[k] = 10 * np.sin((k + kk - 2) * 0.1)
        sineref[kk] = ref[kk]
    r = ref

###################################

u1 = np.zeros((N_sim, 1))
y1 = np.zeros((N_sim, 1))

for kk in range(N_sim):
    DeltaU = np.linalg.inv(Phi_Phi + W_con * np.eye(Nc)) @ (Phi_R * r[kk] - Phi_F @ Xf)
    deltau = DeltaU[0, 0]
    u = u + deltau
    u1[kk] = u
    y1[kk] = y
    xm_old = xm
    xm = Ap @ xm + Bp * u
    y = Cp @ xm
    Xf = np.vstack((xm - xm_old, y))

# x 축
k = range(N_sim)

plt.subplot(211)
plt.plot(k, r, 'k', linewidth=2)
plt.plot(k, y1, 'r--', linewidth=2)
# plt.ylim(-1, 18)
plt.grid(True)
plt.xlabel('Sampling Instant')
plt.legend(['ref', 'Output'])

plt.subplot(212)
plt.plot(k, u1, 'b', linewidth=2)
# plt.ylim(-0.1, 1)
plt.grid(True)
plt.xlabel('Sampling Instant')
plt.legend(['Control input'])

plt.tight_layout()
plt.show()

print()