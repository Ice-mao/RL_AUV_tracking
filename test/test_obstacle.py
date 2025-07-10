import holoocean
import numpy as np
import time

with holoocean.make("SimpleUnderwater-Bluerov2") as env:
    print("Environment created. Starting test...")
    # Define the properties of the box to be spawned
    scale = [3, 1, 2]
    location = [10, 0, -5]
    
    # angle from 0 to 360 degrees, changing over time
    rot_angs = 30/180 * 3.1415
    
    # HoloOcean uses Roll, Pitch, Yaw in degrees for rotation
    # rotation = [np.tan(np.radians(rot_angs)), 1, 0]
    rotation = [0,0, rot_angs]  # Use this if you want to keep the original yaw angle

    env.draw_line([0,0,-5], [10, 10, -5], thickness=5.0, lifetime=0.0)
    env.spawn_prop(prop_type="box", 
                    scale=scale, 
                    location=location,
                    rotation=rotation,
                    material='gold')
    
    print(f"Spawned box at location={location}, rotation (roll, pitch, yaw)={rotation}")
    
    # Tick the environment to update
    while True:
        env.tick()
    

    print("Test finished.")
