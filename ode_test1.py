from src.samplers import ODEIVPCase1Sampler,get_data_sampler
from src.tasks import get_task_sampler
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# 测试构造函数的正常情况
def test_odeivp_case1_sampler_construction():
    sampler = ODEIVPCase1Sampler(n_dims=5)
    assert sampler.n_dims == 5

# 测试 scale 和 bias 的设置
def test_odeivp_case1_sampler_scale_bias():
    sampler = ODEIVPCase1Sampler(n_dims=3, scale=2.0, bias=1.0)
    assert sampler.scale == 2.0
    assert sampler.bias == 1.0

n_dim = 20
bs = 64
n_p = 41
sampler = ODEIVPCase1Sampler(n_dims=n_dim)
sx = sampler.sample_xs(b_size=bs,n_points=n_p)


sampler2 = get_data_sampler("ode_ivp_case1",n_dim)
sx2 = sampler2.sample_xs(b_size=bs,n_points=n_p)

print(f"stage1 shape: {sx2.shape}")

task_sampler = get_task_sampler("ode_ivp_case1",n_dims=n_dim,batch_size=bs)
task_sampler_args = {}
task = task_sampler(**task_sampler_args)

print(sx2[0,0,3])
print(sx2[0,0,3].item())
ys = task.evaluate(sx2)

print("stage2")

def dydt(t, y):
    return sx2[0,0,0].item() * y + sx2[0,0,1].item()

y0 = sx2[0,0,2].item()
steps = int(sx2[0,0,3].item())
# Time span
t_span = (0, 5)
t_eval = np.linspace(0, 5, steps)
print(t_eval)

# Solve the IVP
sol = solve_ivp(dydt, t_span, [y0], t_eval=t_eval)
print(len(sol.t), len(sol.y[0]))
print('t', sol.t)
print('y', sol.y[0])
tru_y = ys[0,0,:steps]
print("truy_shape",tru_y.shape)

tru_y = tru_y.tolist()
# print("truy_new_shape",tru_y.shape)


# Plot the solution
plt.figure(figsize=(10, 6))

# First subplot for the solution
plt.subplot(2, 1, 1)
plt.plot(sol.t, sol.y[0], label='IVP Solution', marker='o')
plt.plot(sol.t, tru_y, label='Ground Truth', marker='s')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution to the IVP')
plt.legend()

# Second subplot for the difference
plt.subplot(2, 1, 2)
plt.plot(sol.t, tru_y- sol.y[0], label='Difference', color='red')
plt.xlabel('t')
plt.ylabel('Difference')
plt.title('Difference between Ground Truth and IVP Solution')
plt.legend()

plt.tight_layout()
plt.show()
print("stage3")