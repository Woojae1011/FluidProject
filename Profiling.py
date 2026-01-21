# Profiling
import cProfile as cp
import ParticleFluidSim as sim
import pstats
from pstats import Stats
import shelve

# Configuration
run = 1  # Run sim?
length = 10  # in seconds

# Initialize profile counter
filename = 'profiles\\profile'
with shelve.open('profiles\\profile_count') as db:
    counter = db.get('value', 0)
    if run == 1:
        counter += 1
    db['value'] = counter
filename += str(counter)

# Run simulation
if run == 1:
    cp.run(f"sim.main(length={length})", filename)

# Display profile statistics
p = Stats(filename)
p.strip_dirs().sort_stats('time').print_stats(30)

# Display frame count comparison
try:
    frames = getattr(sim, 'last_frame_count', None)
    secs = getattr(sim, 'last_sim_seconds', None)
    
    with shelve.open('profiles\\profile_count') as db:
        last_frames = db.get('frame', None)
        last_secs = db.get('sec', None)
        
        # Update current values
        db['frame'] = frames
        db['sec'] = secs
    
    # Only compare if there was a previous run
    if last_frames is not None and counter > 1:
        print(f"\nFrames calculated in last sim: {last_frames}")
        print(f"Frames calculated in this sim: {frames}")
    else:
        print(f"\nFrames calculated: {frames}")
    
    if secs is not None:
        print(f"Sim wait time (s): {secs}")
        
except Exception as e:
    print(f"Could not read frame count from sim module: {e}")