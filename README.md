# RL_AUV_tracking

A project for my major related graduation paper.

Using reinforcement learning(RL) to train an agent to tracking the target in the unknown underwater scenario in HoloOcean.

# Quick Start

To run the simulation, first install all dependencies

- HoloOcean==1.0.0
- Stable Baseline3
- pynput
- bezier
- filterpy
- inekf
- scipy
- sb3-contrib
- seaborn
- shapely

if you want to run in my scenario,I give the scenario link below:
https://drive.google.com/drive/folders/1MdT8NMozJARde7zL5kKebBi4WULfv2KC?usp=drive_link

you should copy the folder into /home/'yourname'/.local/share/holoocean/0.5.0/worlds/

Then simply run the script
```
python SB3_learning.py --env TargetTracking1 --map TestMap_AUV --nb_envs 5 --choice 0 --render 0 
```
choice:(0:train 1:keep training 2:eval)

render:(0:false 1:true)
## Simulation Process

![simulation](config/simulation.png)

## Additional information

If you want to know more details,you should read the code.
:smile: 

Or please keep staying tuning!

## Something mentioned

Just for single target, mutitarget task needs revise the code

(revise the target0 -> target+str(rank))
