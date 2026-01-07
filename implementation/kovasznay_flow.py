import deepxde as dde
import imageio
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


#parametros
Re = 20 # regime laminar 
nu = 1 / Re
l = 1 / (2 * nu) - np.sqrt(1 / (4 * nu**2) + 4 * np.pi**2)

#pdes
def pde(x, u):
    u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3]

    u_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_y = dde.grad.jacobian(u, x, i=0, j=1)
    u_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

    v_x = dde.grad.jacobian(u, x, i=1, j=0)
    v_y = dde.grad.jacobian(u, x, i=1, j=1)
    v_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

    p_x = dde.grad.jacobian(u, x, i=2, j=0)
    p_y = dde.grad.jacobian(u, x, i=2, j=1)

    momentum_x = u_vel * u_x + v_vel * u_y + p_x - (1 / Re) * (u_xx + u_yy)
    momentum_y = u_vel * v_x + v_vel * v_y + p_y - (1 / Re) * (v_xx + v_yy)
    continuity = u_x + v_y

    return [momentum_x, momentum_y, continuity]

#solucao analitica
def u_func(x):
    return 1 - np.exp(l * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2])

def v_func(x):
    return l / (2 * np.pi) * np.exp(l * x[:, 0:1]) * np.sin(2 * np.pi * x[:, 1:2])

def p_func(x):
    return 0.5 * (1 - np.exp(2 * l * x[:, 0:1]))

#domain
def boundary_outflow(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)

spatial_domain = dde.geometry.Rectangle(xmin=[-0.5, -0.5], xmax=[1, 1.5])

bc_u = dde.icbc.DirichletBC(spatial_domain, u_func, lambda _, on_boundary: on_boundary, component=0)
bc_v = dde.icbc.DirichletBC(spatial_domain, v_func, lambda _, on_boundary: on_boundary, component=1)
bc_p = dde.icbc.DirichletBC(spatial_domain, p_func, boundary_outflow, component=2)

data = dde.data.PDE(
    spatial_domain,
    pde,
    [bc_u, bc_v, bc_p],
    num_domain=2601,
    num_boundary=400,
    num_test=10000,
)

class ErrorTracker(dde.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.errors_u = []
        self.errors_v = []
        self.errors_p = []

    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        # registrar erro a cada 100 épocas
        if epoch % 100 != 0:
            return

        X = spatial_domain.random_points(3000)
        pred = self.model.predict(X)
        u_pred, v_pred, p_pred = pred[:, 0], pred[:, 1], pred[:, 2]

        ue = u_func(X).flatten()
        ve = v_func(X).flatten()
        pe = p_func(X).flatten()

        self.errors_u.append(dde.metrics.l2_relative_error(ue, u_pred))
        self.errors_v.append(dde.metrics.l2_relative_error(ve, v_pred))
        self.errors_p.append(dde.metrics.l2_relative_error(pe, p_pred))

#gifs
os.makedirs("frames", exist_ok=True)

def save_frame(model, epoch):
    nx, ny = 200, 200
    xg = np.linspace(-0.5, 1.0, nx)
    yg = np.linspace(-0.5, 1.5, ny)
    Xg, Yg = np.meshgrid(xg, yg)
    XY = np.column_stack([Xg.ravel(), Yg.ravel()])

    pred = model.predict(XY)
    u = pred[:, 0].reshape(ny, nx)
    v = pred[:, 1].reshape(ny, nx)

    plt.figure(figsize=(6, 5))
    plt.streamplot(Xg, Yg, u, v, density=1.2)
    plt.title(f"Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(f"frames/frame_{epoch:05d}.png")
    plt.close()

class GIFCallback(dde.callbacks.Callback):
    def __init__(self, every=200):
        super().__init__()
        self.every = every
        self.model = None   # <<< ESSENCIAL

    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        if epoch % self.every == 0:
            print(f"Salvando frame na época {epoch}")
            save_frame(self.model, epoch)



#arquitetura da rede
net = dde.nn.FNN([2] + 4 * [50] + [3], "tanh", "Glorot normal")
model = dde.Model(data, net)

tracker = ErrorTracker()
gif_cb = GIFCallback(every=50)


#train
model.compile("adam", lr=1e-3)
model.train(iterations=1000, callbacks=[tracker, gif_cb])  # Adam

model.compile("L-BFGS")
model.train(callbacks=[tracker, gif_cb])  # L-BFGS

#plots
plt.figure(figsize=(7, 4))
plt.plot(np.arange(10, 10*len(tracker.errors_u)+1, 10), tracker.errors_u, label="Erro u")
plt.plot(np.arange(10, 10*len(tracker.errors_v)+1, 10), tracker.errors_v, label="Erro v")
plt.plot(np.arange(10, 10*len(tracker.errors_p)+1, 10), tracker.errors_p, label="Erro p")
plt.yscale("log")
plt.xlabel("Época")
plt.ylabel("Erro L2 relativo")
plt.title("Curva de aprendizado")
plt.legend()
plt.tight_layout()
plt.show()


nx, ny = 400, 400
xg = np.linspace(-0.5, 1.0, nx)
yg = np.linspace(-0.5, 1.5, ny)
Xg, Yg = np.meshgrid(xg, yg)
XY = np.column_stack([Xg.ravel(), Yg.ravel()])

pred = model.predict(XY)
u_pred = pred[:, 0].reshape(ny, nx)
v_pred = pred[:, 1].reshape(ny, nx)
p_pred = pred[:, 2].reshape(ny, nx)

vel_mag = np.sqrt(u_pred**2 + v_pred**2)


plt.figure(figsize=(7, 6))
plt.title("Linhas de correntes do escoamento")
plt.streamplot(Xg, Yg, u_pred, v_pred, density=1.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 6))
plt.title("Heatmap da velocidade u")
plt.pcolormesh(Xg, Yg, u_pred, shading="auto")
plt.colorbar(label="u")
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 6))
plt.title("Heatmap da velocidade v")
plt.pcolormesh(Xg, Yg, v_pred, shading="auto")
plt.colorbar(label="v")
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 6))
plt.title("Heatmap da pressão p")
plt.pcolormesh(Xg, Yg, p_pred, shading="auto")
plt.colorbar(label="p")
plt.tight_layout()
plt.show()


def plot_cut(ycut):
    pts = np.column_stack([xg, np.full_like(xg, ycut)])
    pred_cut = model.predict(pts)
    u_p = pred_cut[:, 0]
    v_p = pred_cut[:, 1]
    p_p = pred_cut[:, 2]

    u_e = u_func(pts).flatten()
    v_e = v_func(pts).flatten()
    p_e = p_func(pts).flatten()

    plt.figure(figsize=(7, 5))
    plt.title(f"Corte horizontal y = {ycut}")

    plt.plot(xg, u_p, label="u PINN")
    plt.plot(xg, u_e, "--", label="u exato")
    plt.plot(xg, v_p, label="v PINN")
    plt.plot(xg, v_e, "--", label="v exato")
    plt.plot(xg, p_p, label="p PINN")
    plt.plot(xg, p_e, "--", label="p exato")

    plt.legend()
    plt.tight_layout()
    plt.show()

for ycut in [-0.25, 0.5, 1.25]:
    plot_cut(ycut)


n_particles = 400
n_steps = 120
dt = 0.02


np.random.seed(1)
particles = np.zeros((n_particles, 2))
particles[:, 0] = np.random.uniform(-0.5, 1.0, n_particles)
particles[:, 1] = np.random.uniform(-0.5, 1.5, n_particles)

def velocity_field(xy):
    pred = model.predict(xy)
    return pred[:, 0:2]

os.makedirs("particle_gif", exist_ok=True)


nx, ny = 40, 40   # pouco denso para não poluir
xg = np.linspace(-0.5, 1.0, nx)
yg = np.linspace(-0.5, 1.5, ny)
Xg, Yg = np.meshgrid(xg, yg)
XY = np.column_stack([Xg.ravel(), Yg.ravel()])
UV = velocity_field(XY)
Ug = UV[:, 0].reshape(ny, nx)
Vg = UV[:, 1].reshape(ny, nx)


for step in range(n_steps):
    vel = velocity_field(particles)
    particles += dt * vel

    mask = (
        (particles[:, 0] >= -0.5) & (particles[:, 0] <= 1.0) &
        (particles[:, 1] >= -0.5) & (particles[:, 1] <= 1.5)
    )
    particles = particles[mask]

    while len(particles) < n_particles:
        new_p = np.array([
            np.random.uniform(-0.5, 1.0),
            np.random.uniform(-0.5, 1.5)
        ])
        particles = np.vstack([particles, new_p])

    plt.figure(figsize=(6, 5))


    plt.quiver(
        Xg, Yg, Ug, Vg,
        color="lightblue",
        scale=40,
        width=0.002
    )

    plt.scatter(
        particles[:, 0],
        particles[:, 1],
        s=8,
        c="red",
        alpha=0.9
    )

    plt.xlim(-0.5, 1.0)
    plt.ylim(-0.5, 1.5)
    plt.title("Advecção de partículas em campo estacionário")
    plt.tight_layout()

    fname = f"particle_gif/frame_{step:04d}.png"
    plt.savefig(fname, dpi=120)
    plt.close()


frames = sorted(os.listdir("particle_gif"))
images = [imageio.imread(os.path.join("particle_gif", f)) for f in frames]

imageio.mimsave(
    "kovasznay_particles.gif",
    images,
    fps=12
)

print("GIF de partículas salvo como kovasznay_particles.gif")



