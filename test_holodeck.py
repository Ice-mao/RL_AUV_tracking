# import holodeck
#
# cfg = {
#     "name": "test_rgb_camera",
#     "world": "ExampleWorld",
#     "package_name": "DefaultWorlds",
#     "main_agent": "sphere0",
#     "agents": [
#         {
#             "agent_name": "sphere0",
#             "agent_type": "SphereAgent",
#             "sensors": [
#                 {
#                     "sensor_type": "RGBCamera",
#                     "socket": "CameraSocket",
#                     "configuration": {
#                         "CaptureWidth": 512,
#                         "CaptureHeight": 512
#                     }
#                 }
#             ],
#            "control_scheme": 0,
#             "location": [0, 0, 0]
#         }
#     ]
# }
#
# with holodeck.make(scenario_cfg=cfg) as env:
#     env.tick()

from holodeck import packagemanager
import holodeck
print(holodeck.installed_packages())
holodeck.install("DefaultWorlds")
# packagemanager.available_packages()
# packagemanager.install("DefaultWorlds")