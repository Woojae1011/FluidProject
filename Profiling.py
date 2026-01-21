#Profiling
import cProfile as cp
import ParticleFluidSim as sim
import time
#Stats
import pstats
from pstats import Stats
#Count sim num
import shelve

run = 1 # Run sim?
length = 10 #in seconds

filename = 'profiles\\profile'
with shelve.open('profile_count') as db:
    counter = db['value']
    counter += 1 if run==1 else 0
    db['value'] = counter
filename += str(counter)

# Run sim main script
if run==1:
    cp.run(f"sim.main(length = {length})", filename)


# Sort profile data
p = Stats(filename) # pstats.Stats()
# Cumulative time in function (Top 10) - Excludes time spent in other functions it calls
p.strip_dirs().sort_stats('time').print_stats(30)
# Time spent in function (Top 10) - Actual total time spent in the function
# p.strip_dirs().sort_stats("cumtime").print_stats(10)

# Load prev data file and time diff with current profile
#Time taken
# prev = counter - 1
# if prev > 0:
#     prev_filename = f'profiles\\profile{prev}'
#     try:
#         prev_stats = Stats(prev_filename)
#         print(f'Previous run #{prev} total time (seconds):', prev_stats.total_tt)
#     except Exception as e:
#         print('Could not read previous profile file:', prev_filename, '->', e)
# print(f'Current run #{counter} total time (seconds):', p.total_tt)

#Compare number of frames calculated
try:
    frames = getattr(sim, 'last_frame_count', None)
    secs = getattr(sim, 'last_sim_seconds', None)
    with shelve.open('profile_count') as db:
        db['frame'] = frames
        db['sec'] = secs
        last_frames = db['frame']
        last_secs = db['sec']
    print(f"Frames calculated in last sim: {last_frames}")
    print(f"Frames calculated in this sim: {frames}")
    if secs is not None:
        print(f"Sim wait time (s): {secs}")
except Exception as e:
    print("Could not read frame count from sim module:", e)