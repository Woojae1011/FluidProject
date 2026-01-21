import time
from datetime import timedelta
import os
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.animation import FuncAnimation
from matplotlib.collections import CircleCollection
import numpy as np

# Set seed for random distribution of particles
rng = np.random.default_rng(19970613)

mplstyle.use('fast')


# -----------------
# Simulation config
# -----------------
g = 9.81
sim_length = 120
max_dt = 1
dtype = 'float32'

# World bounds
x_min, x_max = -500.0, 500.0
y_min, y_max = -0.0, 500.0
limits = [[x_min, x_max], [y_min, y_max]]

# Particles
obj_num = 1000
r = 1
m = 5

# Interaction parameters
force_radius = 50
force_radius2 = force_radius**2
force_const = 100
power = 2
max_speed = 100000
min_dist = r
eps2 = 0.1

# Wall forces
wall_rep_force = 1000
wall_force_radius = y_max / 5
wall_bounce = (wall_rep_force == 0)
wall_bounce_coeff = 0


# -----------------
# Optimization config
# -----------------
# Map / update frequency
map_update_freq = 1
inter = 0

map_size_x = force_radius
map_size_y = force_radius
map_dim_x = int((x_max-x_min)/map_size_x)
map_dim_y = int((y_max-y_min)/map_size_y)

# For profiling 
# -----------------
last_frame_count = 0
last_sim_seconds = 0.0
# -----------------

# Precalculate neighbouring maps
keys = np.zeros(obj_num, dtype = 'int16')
keys_sorted, neighboring_particle_sorted, keys_unique, keys_unique_index, keys_unique_inv, keys_unique_count = [
    np.zeros((obj_num), dtype='int16') for _ in range(6)]
key_unique_pair = {}
surrounding_keys = [[] for _ in range(map_dim_x * map_dim_y)]
for i in range(map_dim_x*map_dim_y):
    """possible_keys = [
        key,
        key - 1,
        key + 1,
        key - map_dim,
        key + map_dim,
        key - map_dim - 1,
        key - map_dim + 1,
        key + map_dim - 1,
        key + map_dim + 1,
    ] """
    # vars
    k = i
    n_x = map_dim_x
    n_y = map_dim_y 

    left, right, up, down = (False, False, False, False)

    # Calculate map
    col = k % map_dim_x
    row = k // map_dim_x
    # Check if key is on edge
    left = col == 0
    right = col == map_dim_x-1 
    down = row == 0
    up = row == map_dim_y-1

    
    surrounding_keys[i].append(i) #Itself
    # If not on edge, include surrounding
    if not right: # Not on right edge
        surrounding_keys[i].append(k+1) # Key to right
    if not left: # Not on left edge
        surrounding_keys[i].append(k-1) # Key to left
    if not up: # Not on up edge
        surrounding_keys[i].append(k+map_dim_x) # Key to up
    if not down: # Not on up edge
        surrounding_keys[i].append(k-map_dim_x) # Key to down

    # Diagonals
    if not right and not up:
        surrounding_keys[i].append(k+map_dim_x+1)
    if not left and not up:
        surrounding_keys[i].append(k+map_dim_x-1)
    if not right and not down:
        surrounding_keys[i].append(k-map_dim_x+1)
    if not left and not down:
        surrounding_keys[i].append(k-map_dim_x-1)

# -----------------
# Particle state arrays
# -----------------
# 2D vectors: pos, vel, acc, force (shape: 2 x obj_num)
vec = pos, vel, acc, force = [np.zeros((2, obj_num), dtype=dtype) for _ in range(4)]
# Last position for Verlet integration

# Scalars
radius = r * np.ones(obj_num, dtype=dtype)
mass = m * np.ones(obj_num, dtype=dtype )
speed = np.zeros(obj_num, dtype=dtype)

# Neighbouring particlces
neiboring_particle = -np.ones((map_dim_x*map_dim_y, obj_num), dtype='int16')

# Color per particle (RGB) - stored as 3 x obj_num
color = np.ones((3, obj_num), dtype=dtype) * 0.5

# Wall contact and wall forces
wall_contact = np.zeros((2, obj_num), dtype='int16')
wall_force = np.zeros((2, obj_num), dtype=dtype)

rand_dis = True
center_x = (x_max + x_min) * 0.5
center_y = (y_max + y_min) * 0.5
if rand_dis:
    # Initialize random positions (centered / scaled)
    pos[0] = 0.5 * (rng.random(obj_num) * (x_max - x_min)) + 0.5 * center_x + 0.5 * x_min
    pos[1] = 0.5 * (rng.random(obj_num) * (y_max - y_min)) + 0.5 * center_y + 0.5 * y_min
else:
    # Initialize perfect square in center
    rect_dist = r*4
    dim = int(obj_num**0.5)
    for i in range(obj_num):
        pos[0, i] = i // dim
        pos[1, i] = i % dim
    pos[...] = (pos - 0.5*dim)*rect_dist
    pos[0] += center_x
    pos[1] += center_y

def calculate_map_key() -> None:
    """Update keys to the grid cell keys for all particles"""
    global keys, neiboring_particle_sorted, keys_sorted, keys_unique, keys_unique_index, keys_unique_inv, keys_unique_count, key_unique_pair
    a = np.clip(np.floor((pos[0] - x_min) / map_size_x), 0, map_dim_x - 1)
    b = np.clip(np.floor((pos[1] - y_min) / map_size_y), 0, map_dim_y - 1)

    keys = np.int16(a + b * map_dim_x)
    neiboring_particle_sorted = np.argsort(keys) # Index of particles in order of cells
    keys_sorted = keys[neiboring_particle_sorted] # Keys sorted in order [0,0,0,1,1...]
    keys_unique, keys_unique_index, keys_unique_inv, keys_unique_count = np.unique(
        keys_sorted, return_index=True, return_inverse=True, return_counts=True)
    key_unique_pair.clear()
    key_unique_pair = {key : i for i, key in enumerate(keys_unique)}

def rep_force(i: int, j: int) -> None:
    """Apply pairwise repulsive force between particle i and j (in-place update of `force`)."""
    rel_pos = pos[:, i] - pos[:, j]
    dist2 = max(rel_pos[0]**2 + rel_pos[1]**2, eps2)
    if dist2 < force_radius2:
        if power == 2:
            f = force_const * (rel_pos) / (dist2)
        elif power % 2 != 0:
            dist = max(np.linalg.norm(rel_pos), min_dist)
            f = force_const * (rel_pos) / (dist**power)
        else:
            f = force_const * (rel_pos) / (dist2**(power/2))
        force[:, i] += f
        force[:, j] -= f

def rep_force_vectorized(neiboring_particles: np.ndarray) -> None:
    """Vectorized pairwise repulsive force calculation for neighboring particles."""
    global pos, force, eps2, force_const, force_radius2, power, min_dist

    for particles in neiboring_particles:
        # Skip grids with no particles
        if particles[0] == -1:
            continue
        
        # Get valid particle indices (filter out -1 markers)
        particles_reduced = particles[particles != -1]
        n = len(particles_reduced)
        
        # Skip grids in singular particles
        if n < 2:
            continue
        
        # Get positions of all particles in this group
        # Shape: (2, n) where first row is x coords, second is y coords
        pos_group = pos[:, particles_reduced]
        
        # Create pairwise position differences using broadcasting
        # pos_group[:, :, None] has shape (2, n, 1)
        # pos_group[:, None, :] has shape (2, 1, n)
        # rel_pos has shape (2, n, n) where rel_pos[:, i, j] = pos[i] - pos[j]
        rel_pos = pos_group[:, :, None] - pos_group[:, None, :]
        
        # Calculate squared distances: shape (n, n)
        dist2 = rel_pos[0]**2 + rel_pos[1]**2 + 1
        
        # Clamp to minimum distance squared to avoid division by zero
        dist2_clamped = np.maximum(dist2, eps2)
        
        # Create mask for particles within force radius
        # Also mask out self-interactions (diagonal) and duplicate pairs
        mask = (dist2 < force_radius2)
        # Zero out diagonal (self-interactions)
        np.fill_diagonal(mask, False)
        # Keep only upper triangle to avoid double-counting pairs
        mask = np.triu(mask, k=1)
        
        # Calculate forces based on power law
        if power == 2:
            # Shape of f_magnitude: (n, n)
            f_magnitude = force_const / dist2_clamped
        elif power % 2 != 0:
            # For odd powers, need actual distance
            dist = np.sqrt(dist2_clamped)
            f_magnitude = force_const / (dist**power)
        else:
            # For even powers, use dist2
            f_magnitude = force_const / (dist2_clamped**(power/2))
        
        # Apply mask to only calculate forces for nearby particles
        f_magnitude = np.where(mask, f_magnitude, 0)
        
        # Calculate force vectors: shape (2, n, n)
        # f[:, i, j] is the force ON particle i FROM particle j
        f = rel_pos * f_magnitude[None, :, :]
        
        # Accumulate pairwise forces for each particle, enforcing Newton's third law.
        # For each interacting pair (i, j) with i < j:
        #   particle i gets +f[:, i, j]
        #   particle j gets -f[:, i, j]
        force_updates = np.zeros_like(pos_group)
        i_idx, j_idx = np.where(mask)
        if i_idx.size > 0:
            # Accumulate in x and y dimensions
            for dim in range(2):
                np.add.at(force_updates[dim], i_idx, f[dim, i_idx, j_idx])
                np.add.at(force_updates[dim], j_idx, -f[dim, i_idx, j_idx])
        
        # Update global force array
        force[:, particles_reduced] += force_updates

def sim_update(dt: float) -> None:
    """Advance simulation state by dt seconds. Modifies global particle arrays in-place."""
    global inter, force, acc, vel, pos, color, speed

    # Reset force
    force.fill(0)
    
    calculate_map_key()

    # Preping data for vectorizing force calculation
    neiboring_particle.fill(-1)
    for i, key in enumerate(keys_unique): # key = each occupied cell
        count = 0
        neiboring_cells = []

        for neiboring_particle_keys in surrounding_keys[key]:  # List of keys for neighboring cells
            if key_unique_pair.get(neiboring_particle_keys, -1) >= i: # Keep only occupied cells
                neiboring_cells.append(neiboring_particle_keys) 
        
        neiboring_cells.sort()
        neiboring_particle_index = 0
        for neiboring_particle_keys in neiboring_cells: # lists of keys for occupied neighboring cells
            # Indexing
            act_key = key_unique_pair[neiboring_particle_keys] # Actual key
            count = keys_unique_count[act_key] # Number of particles in each of the neighboring cells + itself
            start = keys_unique_index[act_key]
            # 
            neiboring_particle[i, neiboring_particle_index:neiboring_particle_index+count] = neiboring_particle_sorted[start: start+count]
            neiboring_particle_index += count

    rep_force_vectorized(neiboring_particle)

    # # Brute force testing for force
    # for particles in neiboring_particle:
    #     for particle1 in particles:
    #         if particle1 == -1:
    #             continue
    #         for particle2 in particles:
    #             if particle2 == -1:
    #                 continue
    #             if particle1 < particle2:
    #                 rep_force(particle1, particle2)

    # Gravity
    if g != 0:
        force[1, :] += -g * mass

    # left/right and bottom/top contact indicators
    wall_contact[0] = np.where(pos[0] - radius <= x_min, 1, 0) + np.where(pos[0] + radius >= x_max, 1, 0)
    wall_contact[1] = np.where(pos[1] - radius <= y_min, 1, 0) + np.where(pos[1] + radius >= y_max, 1, 0)

    # wall force indicator
    wall_force[0] = np.where(pos[0] - wall_force_radius <= x_min, 1, 0) + np.where(pos[0] + wall_force_radius >= x_max, -1, 0)
    wall_force[1] = np.where(pos[1] - wall_force_radius <= y_min, 1, 0) + np.where(pos[1] + wall_force_radius >= y_max, -1, 0)

    dists = np.vstack((x_max - pos[0], pos[0] - x_min, y_max - pos[1], pos[1] - y_min))  # shape (4, obj_num)
    min_dist_to_wall = np.min(dists, axis=0)           # shape (obj_num,)
    clamp_dist = np.maximum(min_dist_to_wall, min_dist)     # avoid tiny denom
    denom = clamp_dist**2                              # shape (obj_num,)
    force += wall_rep_force * wall_force / denom

    #Wall collision
    if wall_bounce:
        vel[...] = vel * np.where(wall_contact != 0, -wall_bounce_coeff, 1)
    else:
        vel[...] = vel * np.where(wall_contact != 0, 0, 1)

    # Euler integration
    # acc = force / mass
    # vel += acc * dt
    # vel += 2 * wall_contact * np.abs(vel)
    # vel = np.clip(vel, -max_speed, max_speed)
    # pos += vel * dt

    # velocity-Verlet integration
    pos[...] = pos + vel*dt + acc*(0.5*dt**2) #pos_new
    acc_new = force / mass
    vel[...] = vel + (acc+acc_new) * (dt*0.5) #vel_new
    
    # pos = pos_new
    # vel = vel_new
    acc[...] = acc_new

    # Maximum velocity
    np.clip(vel, -max_speed, max_speed, out=vel)

    # Keep particles inside bounds
    np.clip(pos[0, :], x_min + radius, x_max - radius, out = pos[0, :])
    np.clip(pos[1, :], y_min + radius, y_max - radius, out = pos[1, :])

    # Update dependent variables
    # speed[...] = np.linalg.norm(vel, axis=0)

    # Color on map
    color[0] = (keys % map_dim_x / map_dim_x)
    color[1] = (keys // map_dim_x /map_dim_y)

# -----------------
# Visualization setup
# -----------------
plt.rcParams['toolbar'] = 'None'

fig, (ax) = plt.subplots(figsize=(4*(x_max-x_min)/(y_max-y_min), 4), facecolor='black')
fig.tight_layout()
ax.set_facecolor('black')
ax.plot([x_min, x_max], [y_min, y_min], '-w')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

del_frames = []


def init():   
    # FuncAnimation init function: sets up the annotations and CircleCollection
    global annotation1, annotation2, plt_obj
    annotation1 = plt.annotate("FPS: {}".format(""), xy=(0.1, 0.025), xycoords='figure fraction', color='lightgrey')
    annotation2 = plt.annotate("Time: {}".format(""), xy=(0.4, 0.025), xycoords='figure fraction', color='lightgrey')

    plt_obj = CircleCollection(
        radius**2 * np.pi,
        fc=color.T,
        offsets=pos.T,
        transOffset=ax.transData,
    )
    ax.add_collection(plt_obj)
    return annotation1, annotation2


def update(frame):
    #Per-frame update for FuncAnimation
    global time_start, last_frame_time, fps, del_frames, dt, time_passed
    frame_time = time.perf_counter()
    if frame == 0:
        time_start = frame_time
        last_frame_time = frame_time
        fps = 0
    
    # End animation
    time_passed = time.perf_counter() - time_start
    if time_passed > sim_length:
        plt.close('all')
        return

    # Clamp dt to avoid instability under significant lag
    dt = min(frame_time - last_frame_time, max_dt)
    del_frames.append(frame_time - last_frame_time)

    sec = 0
    i = 0
    for i in range(len(del_frames)):
        sec += del_frames[len(del_frames) - 1 - i]
        if sec >= 1.0:
            break
    fps = i

    # print(f"FPS: {fps}, Total Frames: {len(del_frames)}")
    # print(f"Time passed: {time_passed}, Sim end: {sim_length}")

    # Advance simulation and update visualization
    sim_update(dt)
    plt_obj.set_offsets(pos.T)
    plt_obj.set_color(color.T)

    # Energy (diagnostic)
    # E = 1 / 2 * np.sum(mass * speed**2)
    # print(f"Kinetic_Energy: {E}")

    # Update annotations
    annotation1.set_text("FPS: {}".format(fps))
    annotation2.set_text("Time: {}".format(timedelta(seconds = time_passed)))

    last_frame_time = frame_time
    return plt_obj, annotation1, annotation2


def main(length: float = 10.0, ani_save: bool = False, ani_show: bool = True) -> None:
    global sim_length, last_frame_count, last_sim_seconds
    sim_length = float(length) 
    ani = FuncAnimation(fig=fig, func=update, interval=1, init_func=init, save_count=60)
    if ani_show:
        plt.show()
    if ani_save:
        ani.save(os.path.join("animation", "ParticleFluidSim.mp4"), fps=20)
    last_sim_seconds = time.perf_counter() - time_start
    last_frame_count = len(del_frames)
    time_taken = timedelta(seconds=time_passed)
    print(f"Time taken: {time_taken}")
    print(f"Iterated frames: {last_frame_count}")
    if ani_show:
        plt.close(fig)

if __name__ == "__main__":
    main(sim_length, ani_save=True)