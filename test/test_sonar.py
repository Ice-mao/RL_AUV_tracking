import holoocean
import matplotlib.pyplot as plt
from auv_env.tools import KeyBoardCmd
import numpy as np

#### GET SONAR CONFIG
scenario = "SimpleUnderwater-Hovering"
config = holoocean.packagemanager.get_scenario(scenario)
config = config['agents'][0]['sensors'][-1]["configuration"]
azi = config['Azimuth']
minR = config['RangeMin']
maxR = config['RangeMax']
binsR = config['RangeBins']
binsA = config['AzimuthBins']

#### GET PLOT READY
plt.ion()
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 5))
ax.set_theta_zero_location("N")
ax.set_thetamin(-azi / 2)
ax.set_thetamax(azi / 2)

theta = np.linspace(-azi / 2, azi / 2, binsA) * np.pi / 180
r = np.linspace(minR, maxR, binsR)
T, R = np.meshgrid(theta, r)
z = np.zeros_like(T)

plt.grid(False)
plot = ax.pcolormesh(T, R, z, cmap='gray', shading='auto', vmin=0, vmax=1)
plt.tight_layout()
fig.canvas.draw()
fig.canvas.flush_events()

#### RUN SIMULATION
kb_cmd = KeyBoardCmd(force=30)
with holoocean.make(scenario) as env:
    for i in range(100000):
        if 'q' in kb_cmd.pressed_keys:
            break
        command = kb_cmd.parse_keys()

        # send to holoocean
        env.act("auv0", command)
        state = env.tick()

        if 'ImagingSonar' in state:
            s = state['ImagingSonar']  # (512, 512)
            plot.set_array(s.ravel())

            fig.canvas.draw()
            fig.canvas.flush_events()

print("Finished Simulation!")
plt.ioff()
plt.show()
