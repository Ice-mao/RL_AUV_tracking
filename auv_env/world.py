import holoocean
import numpy as np
from auv_control.estimation import InEKF
from auv_control.control import LQR
from auv_control.planning import Traj, RRT
from auv_control import State
from tools import Plotter
from auv_env.agent import AgentAuv
from auv_env.obstacle import Obstacle

class World:
    """
    build a 3d scenario in HoloOcean to use the more real engine
    """
    def __init__(self, scenario, show, verbose,num_targets, **kwargs):
        # define the entity
        self.ocean = holoocean.make(scenario_cfg=scenario, show_viewport=show, verbose=verbose)
        self.agent = None
        # init the param
        self.sampling_period = 1 / scenario["ticks_per_sec"]  # sample time
        self.num_targets = num_targets  # num of target

        # Setup environment
        self.size = np.array([72.4, 72.4, 25])
        self.bottom_corner = np.array([0, 0, -25])
        self.fix_depth = -5
        self.ocean.draw_box(self.center.tolist(), (self.size / 2).tolist(), color=[0, 0, 255], thickness=30, lifetime=0) # draw the area

        # Tick the scenario
        self.u = np.zeros(8)
        self.ocean.act("auv0", self.u)
        self.sensors = self.ocean.tick()

        # Setup obstacles
        # rule is obstacles combined will rotate from their own center
        self.obstacles = Obstacle(self.ocean, self.fix_depth)
        self.obstacles.draw_obstacle()

        self.obstacle_center_loc = np.random.beta(1.5, 1.5, (num_obstacles, 3)) * self.size + self.bottom_corner
        # Make sure there's none too close to our start or end
        for i in range(self.num_obstacles):
            while np.linalg.norm(self.obstacle_loc[i] - self.start) < 10 or \
                    np.linalg.norm(self.obstacle_loc[i] - self.end) < 10:
                self.obstacle_loc[i] = np.random.beta(2, 2, 3) * self.size + self.bottom_corner

        # Setup agent and target
        self.build_models(sampling_period=self.sampling_period, init_state=self.sensors)

    def build_models(self, sampling_period, init_state, **kwargs):
        # Build a robot
        self.agent = AgentAuv(dim=3, sampling_period=sampling_period, init_state=init_state)
        self.targets = [AgentDoubleInt2D_Nonlinear(self.target_dim,
                                                   self.sampling_period, self.limit['target'],
                                                   lambda x: self.MAP.is_collision(x),
                                                   W=self.target_true_noise_sd, A=self.targetA,
                                                   obs_check_func=lambda x: self.MAP.get_closest_obstacle(
                                                       x, fov=2 * np.pi, r_max=10e2))
                        for _ in range(self.num_targets)]
        # self.targets = [AgentDoubleInt2D_Nonlinear(self.target_dim,
        #                                            self.sampling_period, self.limit['target'],
        #                                            lambda x: self.MAP.is_collision(x),
        #                                            W=self.target_true_noise_sd, A=self.targetA,
        #                                            obs_check_func=lambda x: self.MAP.get_closest_obstacle(
        #                                                x, fov=2 * np.pi, r_max=10e2))
        #                 for _ in range(self.num_targets)]

    def step(self, action_vw):
        self.u = self.agent.update(action_vw, self.sensors)
        self.ocean.act("auv0", self.u)
        self.sensors = self.ocean.tick()

    def draw_world(self):
        """

        :return:
        """
        # Setup environment
        env.draw_box(self.center.tolist(), (self.size/2).tolist(), color=[0,0,255], thickness=30, lifetime=0)
        for i in range(self.num_obstacles):
            loc = self.obstacle_loc[i].tolist()
            loc[1] *= -1
            env.spawn_prop('sphere', loc, [0,0,0], self.obstacle_size[i], False, "white")

        for p in self.path:
            env.draw_point(p.tolist(), color=[255,0,0], thickness=20, lifetime=0)

        super().draw_traj(env, t)

    @property
    def center(self):
        return self.bottom_corner + self.size/2










































