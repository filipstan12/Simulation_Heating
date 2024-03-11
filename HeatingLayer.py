import numpy as np
import matplotlib.pyplot as plt

# Parameters
cycles = 5
num_steps = cycles*40000+1      # Number of time instances to plot
num_points = 6000      # Number of points in space (adjust as needed)
diffusion_constant = 1.5*10E-7  # Diffusion constant
x_length = 0.0006     # space dimension (=1mm)
dt = 10E-5           # Time step
dx = 10E-6           # Spatial step
t_pulse = np.linspace(0, num_steps-1, cycles+1)
t_int_pulse = np.array([int(i) for i in t_pulse])
fluor_dat =[]

def add_amplification(density):
    amp = np.zeros(2*num_points)
    amp[int(num_points - num_points*0.01):int(num_points + num_points*0.01)] = density[int(num_points - num_points*0.01):int(num_points + num_points*0.01)]
    print(amp[num_points])
    density = density + amp
    return density

# Initial conditions
x = np.linspace(-x_length, x_length, 2*num_points)
density_distribution = np.ones((num_steps, 2*num_points))

density_distribution[0, 0:2*num_points] = 1.0 # Initial density distribution
#density_distribution[0, 0:2*num_points] = genfromtxt("density_distribution.csv", delimiter=",")

# Simulation
for step in range(1, num_steps):

    # Compute the second derivative (diffusion term)
    d2rho_dx2 = np.gradient(np.gradient(density_distribution[step - 1], dx), dx)
    # Update density distribution using the diffusion equation
    density_distribution[step] = density_distribution[step - 1] + diffusion_constant * dt * d2rho_dx2

    if step in t_int_pulse:
        density_distribution[step] = add_amplification(density_distribution[step])
        print("Pulse generated at", step*dt, "seconds")
        fluor = np.sum(density_distribution[step][num_points:]/10000)
        fluor_dat.append(fluor)



# Plotting the density distribution at different time instances
plt.figure(figsize=(10, 6))
for step in t_int_pulse:
    plt.plot(x[num_points:], density_distribution[step][num_points:], label=f'Time  {step*dt} s')

plt.xlabel('Position (mm)')
plt.ylabel('Density')
plt.title('Simulation of Diffusion of Particles in 1D Space (Continuous Density)')
plt.legend()
plt.show()

np.savetxt("density_distribution.csv", density_distribution[-1], delimiter=",")
np.savetxt('Fluorescence.csv', fluor_dat, delimiter=",")
print("density distribution of size", np.size(density_distribution[-1]), "was saved")