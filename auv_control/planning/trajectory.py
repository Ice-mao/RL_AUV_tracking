import numpy as np
from auv_control import State
from .base import BasePlanner

class Traj(BasePlanner):
    def __init__(self, route, num_seconds):
        pos, rot = self._make_route(route, num_seconds)
        self.pos_func = pos
        self.rot_func = rot

    def _make_route(self, route, num_seconds):
        # Setup trajectory
        if route == "helix":
            R = 3
            tau = 2
            pos = lambda t: np.array([R*np.cos(t*tau*2*np.pi/num_seconds), R*np.sin(t*tau*2*np.pi/num_seconds), -5-0.05*t]).T
            rot = lambda t: np.array([0*t, 15-30*t/num_seconds, 170+t*tau*360/num_seconds]).T
        
        elif route == "wave":
            pos = lambda t: np.array([t/2, 0*t, -2*np.cos(t*4*2*np.pi/num_seconds)-3]).T
            rot = lambda t: np.array([0*t, -np.arctan(2*2*np.pi/num_seconds*np.sin(t*4*2*np.pi/num_seconds))*180/np.pi, 0*t]).T
        
        elif route == "square":
            q = num_seconds/4

            def pos(t):
                v = 0.5
                if t <= q:
                    return np.array([0*t, v*t, 0*t-5])
                elif q < t and t <= 2*q:
                    t -= q
                    return np.array([-v*t, 0*t + v*q, -5])
                elif 2*q < t and t <= 3*q:
                    t -= 2*q
                    return np.array([-v*q, -v*t + v*q, -5])
                elif 3*q < t:
                    t -= 3*q
                    return np.array([v*t - v*q, 0*t, -5])
            pos = np.vectorize(pos, signature='()->(n)')
            

            def rot(t):
                if t <= q:
                    return np.array([0*t, 0*t, 0*t])
                elif q < t and t <= 2*q:
                    return np.array([0*t, 0*t, 90+0*t])
                elif 2*q < t and t <= 3*q:
                    return np.array([0*t, 0*t, 180+0*t])
                elif 3*q < t:
                    return np.array([0*t, 0*t, 270+0*t])
            rot = np.vectorize(rot, signature='()->(n)')

        else:
            raise NotImplementedError(f"Route {route} not implemented")
        
        return pos, rot