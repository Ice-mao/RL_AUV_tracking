scenario = {
    "name": "Hovering",
    "package_name": "Ocean",
    "world": "SimpleUnderwater",
    "main_agent": "auv0",
    "ticks_per_sec": 100,
    "frames_per_sec": False,
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "PoseSensor",
                    "socket": "COM"
                },
                {
                    "sensor_type": "VelocitySensor",
                    "socket": "COM"
                },
                {
                    "sensor_type": "IMUSensor",
                    "sensor_name": "IMUSensorClean",
                    "socket": "COM"
                },
                {
                    "sensor_type": "IMUSensor",
                    "socket": "COM",
                    "Hz": 100,
                    "configuration": {
                        "AccelSigma": 0.00277,
                        "AngVelSigma": 0.00123,
                        "AccelBiasSigma": 0.0,
                        "AngVelBiasSigma": 0.0,
                        "AccelBiasSigma": 0.00141,
                        "AngVelBiasSigma": 0.00388,
                        "ReturnBias": True
                    }
                },
                {
                    "sensor_type": "GPSSensor",
                    "socket": "COM",
                    "Hz": 2,
                    "configuration": {
                        "Sigma": 0.3,
                        "Depth": 3,
                        "DepthSigma": 1
                    }
                },
                {
                    "sensor_type": "OrientationSensor",
                    "sensor_name": "CompassSensor",
                    "socket": "COM",
                    "Hz": 50,
                    "configuration": {
                        "Sigma": 0.05
                    }
                },
                {
                    "sensor_type": "DVLSensor",
                    "socket": "COM",
                    "Hz": 5,
                    "configuration": {
                        "Elevation": 22.5,
                        "VelSigma": 0.02626,
                        "ReturnRange": False,
                        "MaxRange": 50,
                        "RangeSigma": 0.1
                    }
                },
                {
                    "sensor_type": "DepthSensor",
                    "socket": "COM",
                    "Hz": 50,
                    "configuration": {
                        "Sigma": 0.255
                    }
                },
                {
                    "sensor_type": "RangeFinderSensor",
                    "socket": "COM",
                    "Hz": 15,
                    "configuration": {
                        "LaserMaxDistance": 10,
                        "LaserCount": 24,
                        "LaserDebug": False
                    }
                },
                {
                    "sensor_type": "RGBCamera",
                    "sensor_name": "LeftCamera",
                    "socket": "CameraLeftSocket",
                    "Hz": 15,
                    "configuration": {
                        "CaptureWidth": 256,
                        "CaptureHeight": 256
                    }
                },
                {
                    "sensor_type": "RGBCamera",
                    "sensor_name": "RightCamera",
                    "socket": "CameraRightSocket",
                    "Hz": 15,
                    "configuration": {
                        "CaptureWidth": 256,
                        "CaptureHeight": 256
                    }
                },
                # {
                #     "sensor_type": "ImagingSonar",
                #     "socket": "SonarSocket",
                #     "Hz": 5,
                #     "configuration": {
                #         "RangeBins": 512,
                #         "AzimuthBins": 512,
                #         "RangeMin": 1,
                #         "RangeMax": 10,
                #         "InitOctreeRange": 50,
                #         "Elevation": 20,
                #         "Azimuth": 120,
                #         "AzimuthStreaks": -1,
                #         "ScaleNoise": False,
                #         "AddSigma": 0,
                #         "MultSigma": 0,
                #         "RangeSigma": 0,
                #         "MultiPath": 0
                #     }
                # }
            ],
            "control_scheme": 0,
            "location": [0.0, 0.0, -5.0],
            "rotation": [0.0, 0.0, 0.0]
        },
        {
            "agent_name": "target",
            "agent_type": "SphereAgent",
            "sensors": [
                {
                    "sensor_type": "PoseSensor",
                    "socket": "COM"
                },
                {
                    "sensor_type": "LocationSensor",
                    "socket": "COM"
                },
                {
                    "sensor_type": "VelocitySensor",
                    "socket": "COM"
                },
            ],
            "control_scheme": 1,
            "location": [2, 2, -5.0],
            "rotation": [0.0, 0.0, 0.0]
        }
    ],

    "window_width": 1280,
    "window_height": 720
}
