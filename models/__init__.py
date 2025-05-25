# Load the benchmarks from the subfiles
from .collision_avoidance import CollisionAvoidance
from .drone4D import Drone4D
from .linearsystem import LinearSystem
from .linearsystem4D import LinearSystem4D
from .mountain_car import MountainCar
from .pendulum import Pendulum
from .planar_robot import PlanarRobot
from .triple_integrator import TripleIntegrator
from .vanderpol import Vanderpol


def get_model_fun(model_name):
    if model_name == 'LinearSystem':
        envfun = LinearSystem
    elif model_name == 'LinearSystem4D':
        envfun = LinearSystem4D
    elif model_name == 'MyPendulum':
        envfun = Pendulum
    elif model_name == 'CollisionAvoidance':
        envfun = CollisionAvoidance
    elif model_name == 'TripleIntegrator':
        envfun = TripleIntegrator
    elif model_name == 'PlanarRobot':
        envfun = PlanarRobot
    elif model_name == 'Drone4D':
        envfun = Drone4D
    elif model_name == 'MyMountainCar':
        envfun = MountainCar
    elif model_name == 'VanDerPol':
        envfun = Vanderpol
    else:
        envfun = False
        assert False, f"Unknown model name: {model_name}"

    return envfun
