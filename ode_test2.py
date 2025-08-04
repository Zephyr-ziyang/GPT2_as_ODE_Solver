from src.samplers import ODEIVPCase2Sampler,get_data_sampler
from src.tasks import get_task_sampler
import numpy as np
from scipy.integrate import solve_ivp,quad
import matplotlib.pyplot as plt


n_dim = 20
bs = 64
n_p = 41
sampler = ODEIVPCase2Sampler(n_dims=n_dim)
sx = sampler.sample_xs(b_size=bs,n_points=n_p)


sampler2 = get_data_sampler("ode_ivp_case2",n_dim)
sx2 = sampler2.sample_xs(b_size=bs,n_points=n_p)
print("stage1")
print(f"x shape: {sx2.shape}")
# sampler3 = get_data_sampler("ode_ivp_case2",6)
# sx3 = sampler3.sample_xs(b_size=bs,n_points=n_p)
[a_1, a_2, b_1, b_2, y_0, t_e, steps] = sx2[0,0,:7]
print(a_1.item(),a_2.item(),b_1.item(),b_2.item(),y_0.item(),t_e.item(),steps.item())

def ground_truth(t,argss):
    [a_1, a_2, b_1, b_2, y_0, t_e, steps] = argss
    def integrand(s, alpha_1, alpha_2, beta_1, beta_2):
        exponent = (alpha_1 / 2) * s**2 + (alpha_2 + beta_2) * s
        return beta_1 * np.exp(exponent)

    # 计算积分
    integral_result, error = quad(integrand, 0, t_e, args=(a_1, a_2, b_1, b_2), limit=steps * 3)
    # 计算 y(t)
    y_t = np.exp(-((a_1 / 2) * t**2 + a_2 * t))(y_0 + integral_result)
    return y_t

task_sampler = get_task_sampler("ode_ivp_case2",n_dims=n_dim,batch_size=bs)
task_sampler_args = {}
task = task_sampler(**task_sampler_args)

print(sx2[0,0,6])
print(sx2[0,0,6].item())
ys = task.evaluate(sx2)
print(f"y shape: {ys.shape}")
print("stage2")

def dydt(t, y):
    return -(a_1*t+a_2)*y+(b_1*np.exp(b_2*t))

y_0 = y_0.item()
steps = int(steps.item())
t_e = t_e.item()
# Time span
t_span = (0, t_e)
t_eval = np.linspace(0, t_e, steps)
# print(t_eval)

# Solve the IVP
sol = solve_ivp(dydt, t_span, [y_0], t_eval=t_eval)
print(len(sol.t), len(sol.y[0]))
print('t', sol.t)
print('y', sol.y[0])
tru_y = ys[0,0,:steps]
print("truy",tru_y)
print("truy_shape",tru_y.shape)

tru_y = tru_y.tolist()
# print("truy_new_shape",tru_y.shape)