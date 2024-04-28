import numpy as np
from inekf import (
    DVLSensor,
    DepthSensor,
    MeasureModel,
    InertialProcess,
)
from inekf import InEKF as InvariantEKF
from inekf import SE3, SO3, ERROR
from auv_control import State
from auv_control import scenario

# Import parameters from scenario file
sensors_params = scenario["agents"][0]["sensors"]
sensors_params = {d["sensor_type"]: d for d in sensors_params}
sensors_params["CompassSensor"] = sensors_params["OrientationSensor"]


class CompassSensor(MeasureModel[SE3[2, 6]]):
    def __init__(self, std):
        super().__init__()

        self.error = ERROR.RIGHT

        self.M = np.eye(3) * std**2

        H = np.zeros((3, 15))
        H[1, 2] = -1
        H[2, 1] = 1
        self.H = H

    def processZ(self, z, state):
        new_z = np.zeros(5)
        new_z[:3] = z
        return new_z

    def calcV(self, z, state):
        V = state.State[:3, :] @ z - np.array([1, 0, 0])
        return V


class InEKF:
    def __init__(self, x_init, y_init, error=ERROR.RIGHT):
        # set up initial state
        xi = np.zeros(15)
        xi[6] = x_init  # x_init
        xi[7] = y_init  # y_init
        xi[8] = -5  # depth

        s = np.zeros(15)
        s[0:3] = 0.1  # rotation
        s[3:6] = 1.0  # position
        s[6:9] = 0.1  # velocity
        s[9:12] = 0.000025  # bias
        s[12:15] = 0.0025
        x0 = SE3[2, 6](xi, np.diag(s))

        # set up DVL
        dvlR = SO3()
        dvlT = np.zeros(3)
        self.dvl = DVLSensor(dvlR, dvlT)
        self.dvl.setNoise(
            sensors_params["DVLSensor"]["configuration"]["VelSigma"],
            sensors_params["IMUSensor"]["configuration"]["AngVelSigma"],
        )

        # set up depth sensor
        self.depth = DepthSensor(
            sensors_params["DepthSensor"]["configuration"]["Sigma"]
        )

        # set up compass sensor
        b = np.array([1, 0, 0, 0, 0])
        noise =  np.eye(3)*sensors_params["CompassSensor"]["configuration"]["Sigma"]**2
        self.compass = MeasureModel[SE3[2, 6]](b, noise, ERROR.RIGHT)

        # gps sensor
        b = np.array([0, 0, 0, 0, 1])
        noise = np.eye(3) * sensors_params["GPSSensor"]["configuration"]["Sigma"] ** 2
        self.gps = MeasureModel[SE3[2, 6]](b, noise, ERROR.LEFT)

        # Process model
        pModel = InertialProcess()
        pModel.setGyroNoise(
            sensors_params["IMUSensor"]["configuration"]["AngVelSigma"]
        )
        pModel.setAccelNoise(
            sensors_params["IMUSensor"]["configuration"]["AccelSigma"]
        )
        pModel.setGyroBiasNoise(
            sensors_params["IMUSensor"]["configuration"]["AngVelBiasSigma"]
        )
        pModel.setAccelBiasNoise(
            sensors_params["IMUSensor"]["configuration"]["AccelBiasSigma"]
        )

        # Initialize sensors
        self.iekf = InvariantEKF(pModel, x0, error)
        self.iekf.addMeasureModel("DVL", self.dvl)
        self.iekf.addMeasureModel("Depth", self.depth)
        self.iekf.addMeasureModel("GPS", self.gps)
        self.iekf.addMeasureModel("Compass", self.compass)

        self.last_omega = np.zeros(3)

    def predict_imu(self, imu, dt):
        a = imu[0]
        omega = imu[1]
        u = np.append(omega, a)
        self.last_omega = omega
        return self.iekf.predict(u, dt)

    def update_gps(self, gps):
        return self.iekf.update("GPS", gps)

    def update_depth(self, depth):
        return self.iekf.update("Depth", depth)

    def update_dvl(self, dvl):
        dvl = np.append(dvl, self.last_omega)
        return self.iekf.update("DVL", dvl)

    def update_compass(self, compass):
        return self.iekf.update("Compass", compass)

    def tick(self, state, ts):
        if "IMUSensor" in state:
            est_state = self.predict_imu(state["IMUSensor"], ts)
        if "DVLSensor" in state:
            est_state = self.update_dvl(state["DVLSensor"])
        if "DepthSensor" in state:
            est_state = self.update_depth(state["DepthSensor"])
        if "GPSSensor" in state:
            est_state = self.update_gps(state["GPSSensor"])
        if "CompassSensor" in state:
            z = np.linalg.inv(state["CompassSensor"])[:3, 0]
            z += np.random.normal(
                0, sensors_params["CompassSensor"]["configuration"]["Sigma"], 3
            )
            est_state = self.update_compass(z)

        est_state = State(est_state, self.last_omega)

        return est_state


if __name__ == '__main__':
    obs = InEKF()
    print('1')
    print('1')