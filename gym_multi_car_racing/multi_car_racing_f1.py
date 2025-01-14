import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym.envs.box2d.car_dynamics as car_dynamics
from gymnasium import spaces
from gymnasium.utils import colorize, seeding, EzPickle

# Virtual display
import pyvirtualdisplay
# Creates a virtual display for OpenAI gym
pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

# Physical display
# from pyglet.window import key

import pyglet
from pyglet import gl
from shapely.geometry import Point, Polygon

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from .formula1 import *

import cv2


dict_obs_space = False

# Limits for removing the green background (grass)
lower_green = np.array([25, 52, 72])
upper_green = np.array([102, 255, 255])


def preprocess(img, grayscale):
    img = img[:84, 6:90, :] # CarRacing-v2-specific cropping
    # img = cv2.resize(img, dsize=(84, 84)) # or you can simply use rescaling
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img

# Easiest continuous control task to learn from pixels, a top-down racing environment.
# Discrete control is reasonable in this environment as well, on/off discretization is
# fine.
#
# State consists of STATE_W x STATE_H pixels.
#
# Reward is -0.1 every frame and +1000/N for every track tile visited, where N is
# the total number of tiles visited in the track. For example, if you have finished in 732 frames,
# your reward is 1000 - 0.1*732 = 926.8 points.
#
# Game is solved when agent consistently gets 900+ points. Track generated is random every episode.
#
# Episode finishes when all tiles are visited. Car also can go outside of PLAYFIELD, that
# is far off the track, then it will get -100 and die.
#
# Some indicators shown at the bottom of the window and the state RGB buffer. From
# left to right: true speed, four ABS sensors, steering wheel position and gyroscope.
#
# To play yourself (it's rather fast for humans), type:
#
# python gym/envs/box2d/car_racing.py
#
# Remember it's powerful rear-wheel drive car, don't press accelerator and turn at the
# same time.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

STATE_W = 96   # 96 less than Atari 160x192
STATE_H = 96   # 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE       = 6.0        # Track scale (default = 6.0)
TRACK_RAD   = 900/SCALE  # Track is heavily morphed circle with this radius (default = 900)
PLAYFIELD   = 3000/SCALE # Game over boundary
FPS         = 50        # Frames per second
ZOOM        = 2.7        # Camera zoom (default = 2.7)
ZOOM_FOLLOW = True       # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21/SCALE  # Default 21
TRACK_TURN_RATE = 0.31  # Default 0.31
TRACK_WIDTH = 40/SCALE  # Default 40
BORDER = 8/SCALE  # Default 8
BORDER_MIN_COUNT = 4  # Default 4

ROAD_FRICTION = 1.0  # Default 1.0
DOMAIN_RANDOMIZE = False  # Default False
TRACK = "Belgium"

ROAD_COLOR = [0.4, 0.4, 0.4]

# Specify different car colors
CAR_COLORS = [(0.8, 0.0, 0.0), (0.0, 0.0, 0.8),
              (0.0, 0.8, 0.0), (0.0, 0.8, 0.8),
              (0.8, 0.8, 0.8), (0.0, 0.0, 0.0),
              (0.8, 0.0, 0.8), (0.8, 0.8, 0.0)]

# Distance between cars
LINE_SPACING = 5     # Starting distance between each pair of cars
LATERAL_SPACING = 3  # Starting side distance between pairs of cars

# Penalizing backwards driving
BACKWARD_THRESHOLD = np.pi/2
K_BACKWARD = 0  # Penalty weight: backwards_penalty = K_BACKWARD * angle_diff  (if angle_diff > BACKWARD_THRESHOLD)

def get_track(track_name):
    tracks = {"Australia": Australia,
              "Austria": Austria,
              "Bahrain": Bahrain,
              "Belgium": Belgium,
              "Brazil": Brazil,
              "China": China,
              "France": France,
              "Germany": Germany,
              "Hungary": Hungary,
              "Italy": Italy,
              "Malaysia": Malaysia,
              "Monaco": Monaco,
              "Mexico": Mexico,
              "Netherlands": Netherlands,
              "Portugal": Portugal,
              "Russia": Russia,
              "Singapore": Singapore,
              "Spain": Spain,
              "UK": UK,
              "USA": USA}
    global PLAYFIELD
    PLAYFIELD = tracks[track_name].bounds/SCALE # Game over boundary
    
    return tracks[track_name]


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        self._contact(contact, True)
    def EndContact(self, contact):
        self._contact(contact, False)
    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData

        if u1 and "tile" in u1:
            if "road_friction" in u1['tile'].__dict__:
                tile = u1['tile']
                index = u1['index']
                obj = u2
        if u2 and "tile" in u2:
            if "road_friction" in u2['tile'].__dict__:
                tile = u2['tile']
                index = u2['index']
                obj = u1
        if not tile:
            return

        # tile.color[0] = ROAD_COLOR[0]
        # tile.color[1] = ROAD_COLOR[1]
        # tile.color[2] = ROAD_COLOR[2]

        tile.color[:] = self.env.road_color

        # This check seems to implicitly make sure that we only look at wheels as the tiles 
        # attribute is only set for wheels in car_dynamics.py.
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            # print tile.road_friction, "ADD", len(obj.tiles)
            if not tile.road_visited[obj.car_id]:
                tile.road_visited[obj.car_id] = True
                self.env.tile_visited_count[obj.car_id] += 1

                # The reward is dampened on tiles that have been visited already.
                past_visitors = sum(tile.road_visited)-1
                reward_factor = 1 - (past_visitors / self.env.n_agents)
                self.env.reward[obj.car_id] += reward_factor * 1000.0/len(self.env.track)
        else:
            obj.tiles.remove(tile)
            # print tile.road_friction, "DEL", len(obj.tiles) -- should delete to zero when on grass (this works)


def env(track, render_mode=None, **kwargs):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(track, render_mode=internal_render_mode, **kwargs)
    #env = wrappers.ClipOutOfBoundsWrapper(env)  # For continuous action spaces 
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(track, render_mode=None, **kwargs):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(track, render_mode=render_mode, **kwargs)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "state_pixels"],
        "name": "multi_car_racing",
        "render_fps": FPS,
    }
    
    def __init__(self, track, n_agents=2, verbose=0, direction='CCW',
                 use_random_direction=False, backwards_flag=True, h_ratio=0.25,
                 use_ego_color=False, render_mode="state_pixels",
                 discrete_action_space=False, grayscale=False,
                 domain_randomize=False):
        EzPickle.__init__(self)
        self.seed()
        self.n_agents = n_agents
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0,0), contactListener=self.contactListener_keepref)
        self.viewer = [None] * n_agents
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.cars = [None] * n_agents
        self.car_order = None  # Determines starting positions of cars
        self.reward = np.zeros(n_agents)
        self.prev_reward = np.zeros(n_agents)
        self.tile_visited_count = [0]*n_agents
        self.verbose = verbose
        self.fd_tile = fixtureDef(
                shape = polygonShape(vertices=
                    [(0, 0),(1, 0),(1, -1),(0, -1)]))
        self.driving_backward = np.zeros(n_agents, dtype=bool)
        self.driving_on_grass = np.zeros(n_agents, dtype=bool)
        self.use_random_direction = use_random_direction  # Whether to select direction randomly
        self.episode_direction = direction  # Choose 'CCW' (default) or 'CW' (flipped)
        if self.use_random_direction:  # Choose direction randomly
            self.episode_direction = np.random.choice(['CW', 'CCW'])
        self.backwards_flag = backwards_flag  # Boolean for rendering backwards driving flag
        self.h_ratio = h_ratio  # Configures vertical location of car within rendered window
        self.use_ego_color = use_ego_color  # Whether to make ego car always render as the same color
        self.discrete_action_space = discrete_action_space
        self.grayscale = grayscale
        self.domain_randomize = domain_randomize  # Whether to randomize the background and grass colors
        self.f1_track = get_track(track)  # Get track from formula1.py
        print("Track:", track)
        self._init_colors()

        self.action_lb = np.tile(np.array([-1,+0,+0]), 1).astype(np.float32)
        self.action_ub = np.tile(np.array([+1,+1,+1]), 1).astype(np.float32)

        self.possible_agents = [f"car_{i}" for i in range(self.n_agents)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        
        # Shape of one frame
        self.frame_shape = (84, 84, 1) if self.grayscale else (84, 84, 3)
        high = 1
        dtype = np.uint8

        obs_space = spaces.Box(low=0, high=high, shape=self.frame_shape, dtype=dtype)
        self.observation_spaces = dict(zip(self.possible_agents,
                                           [obs_space]*self.n_agents))
        
        # Discrete action space
        if discrete_action_space:
            self.action_spaces = dict(zip(self.possible_agents, [spaces.Discrete(5)]*self.n_agents))  # do nothing, left, right, gas, brake
        else:
            self.action_spaces = dict(zip(self.possible_agents, [spaces.Box(low=self.action_lb, high=self.action_ub, dtype=np.float32)]*self.n_agents))

        self.render_mode = render_mode

    # Observation space should be defined here.
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            t.userData = t.userData['tile']
            self.world.DestroyBody(t)
        self.road = []
        for car in self.cars:
            car.destroy()
        self.cars = [None] * self.n_agents

    def _init_colors(self):
        if self.domain_randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)/255.

            self.bg_color = self.np_random.uniform(0, 210, size=3)/255.

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20 / 255.
        else:
            # default colours
            self.road_color = np.array([102, 102, 102])/255.
            self.bg_color = np.array([102, 204, 102])/255.
            self.grass_color = np.array([102, 230, 102])/255.

    def _reinit_colors(self, randomize):
        assert (
            self.domain_randomize
        ), "domain_randomize must be True to use this function."

        if randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)/255.

            self.bg_color = self.np_random.uniform(0, 210, size=3)/255.

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20 / 255.

    def _create_track(self):

        track = []
        self.road = []

        x,y = zip(*self.f1_track.xy)
        min_x, max_x = x[-1], x[-1]
        min_y, max_y = y[-1], y[-1]

        points = self.f1_track.xy
        betas = []
        for i, p in enumerate(points[:-1]):
            x1, y1 = points[i]
            x2, y2 = points[i+1]
            dx = x2 - x1
            dy = y2 - y1
            if (dx == dy == 0):
                continue

            # alpha = math.atan(dy/(dx+1e-5))
            alpha = np.arctan2(dy, dx)
            beta = math.pi/2 + alpha

            track.append((alpha, beta, x1, y1))
            betas.append(beta)

            min_x = min(x1, min_x)
            min_y = min(y1, min_y)
            max_x = max(x1, max_x)
            max_y = max(y1, max_y)

        x_offset = min_x + (max_x - min_x)/2
        y_offset = min_y + (max_y - min_y)/2
        self.x_offset = x_offset
        self.y_offset = y_offset

        betas = np.array(betas)
        abs_dbeta = abs(betas[1:] - betas[0:-1])
        mean_abs_dbeta = abs_dbeta.mean()
        std_abs_dbeta = abs_dbeta.std()
        one_dev_dbeta = mean_abs_dbeta + std_abs_dbeta/2

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > mean_abs_dbeta
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]

            alpha2, beta2, x2, y2 = track[i - 1]

            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1) - x_offset,
                y1 - TRACK_WIDTH * math.sin(beta1) - y_offset,
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1) - x_offset,
                y1 + TRACK_WIDTH * math.sin(beta1) - y_offset,
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2) - x_offset,
                y2 - TRACK_WIDTH * math.sin(beta2) - y_offset,
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2) - x_offset,
                y2 + TRACK_WIDTH * math.sin(beta2) - y_offset,
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]

            try:
                self.fd_tile.shape.vertices = vertices
            except:
                pass
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            # t.userData = t
            t.userData = {
                'tile': t,
                'index': i
            }
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = [False]*self.n_agents
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)

            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1) - x_offset,
                    y1 + side * TRACK_WIDTH * math.sin(beta1) - y_offset,
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1) - x_offset,
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1) - y_offset,
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2) - x_offset,
                    y2 + side * TRACK_WIDTH * math.sin(beta2) - y_offset,
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2) - x_offset,
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2) - y_offset,
                )
                self.road_poly.append(
                    ([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0))
                )
        self.track = track
        self.road_poly_shapely = [Polygon(self.road_poly[i][0]) for i in
                                  range(len(self.road_poly))]

        return True

    def reset(self, seed=None, options=None):
        self._destroy()
        self.reward = np.zeros(self.n_agents)
        self.prev_reward = np.zeros(self.n_agents)
        self.tile_visited_count = [0]*self.n_agents
        self.t = 0.0
        self.road_poly = []
        self.agents = self.possible_agents[:]
        self.elapsed_time = 0
        self.speed = np.zeros(self.n_agents)
        self.color = [(0, 255, 0), (0, 255, 0)]

        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]

            self._reinit_colors(randomize)

        # Reset driving backwards/on-grass states and track direction
        self.driving_backward = np.zeros(self.n_agents, dtype=bool)
        self.driving_on_grass = np.zeros(self.n_agents, dtype=bool)
        if self.use_random_direction:  # Choose direction randomly
            self.episode_direction = np.random.choice(['CW', 'CCW'])

        # Set positions of cars randomly
        ids = [i for i in range(self.n_agents)]
        shuffle_ids = np.random.choice(ids, size=self.n_agents, replace=False)
        self.car_order = {i: shuffle_ids[i] for i in range(self.n_agents)}

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print("retry to generate track (normal if there are not many of this messages)")

        beta0, x0, y0 = self.track[0][1:4]

        x0 -= self.x_offset
        y0 -= self.y_offset

        car_width = car_dynamics.SIZE * (car_dynamics.WHEEL_W * 2 \
            + (car_dynamics.WHEELPOS[1][0]-car_dynamics.WHEELPOS[1][0]))
        for car_id in range(self.n_agents):

            # Specify line and lateral separation between cars
            line_spacing = LINE_SPACING
            lateral_spacing = LATERAL_SPACING

            #index into positions using modulo and pairs
            line_number = math.floor(self.car_order[car_id] / 2)  # Starts at 0
            side = (2 * (self.car_order[car_id] % 2)) - 1  # either {-1, 1}

            # Compute angle based off of track index for car
            angle = self.track[-line_number * line_spacing][1]
            if self.episode_direction == 'CW':  # CW direction indicates reversed
                angle -= np.pi  # Flip direction is either 0 or pi

            # Compute offset angle (normal to angle of track)
            norm_theta = angle - np.pi/2

            # Display spawn locations of cars.
            # print(f"Spawning car {car_id} at {new_x:.0f}x{new_y:.0f} with "
            #       f"orientation {angle}")

            # Create car at location with given angle
            self.cars[car_id] = car_dynamics.Car(self.world, beta0, x0, y0)
            self.cars[car_id].hull.color = CAR_COLORS[car_id % len(CAR_COLORS)]

            # This will be used to identify the car that touches a particular tile.
            for wheel in self.cars[car_id].wheels:
                wheel.car_id = car_id

        obs = self.render("state_pixels")
        observations = {agent: preprocess(obs[i], self.grayscale) for i, agent in enumerate(self.agents)}

        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        """ Run environment for one timestep. 
        """
        if actions is not None:
            for car_id, car in enumerate(self.cars):
                if self.discrete_action_space:
                    car.steer(-0.6 * (actions[f"car_{car_id}"] == 1) + 0.6 * (actions[f"car_{car_id}"] == 2))
                    car.gas(0.2 * (actions[f"car_{car_id}"] == 3))
                    car.brake(0.8 * (actions[f"car_{car_id}"] == 4))
                else:
                    car.steer(-actions[f"car_{car_id}"][0])
                    car.gas(actions[f"car_{car_id}"][1])
                    car.brake(actions[f"car_{car_id}"][2])

        for car in self.cars:
            car.step(1.0/FPS)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS

        self.state = self.render("state_pixels")

        self.elapsed_time += 1

        step_reward = np.zeros(self.n_agents)
        done = False
        if actions is not None: # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER

            # NOTE(IG): Probably not relevant. Seems not to be used anywhere. Commented it out.
            # self.cars[0].fuel_spent = 0.0

            for car_id, car in enumerate(self.cars):  # Enumerate through cars

                # Get car speed
                vel = car.hull.linearVelocity
                self.speed[car_id] = np.linalg.norm(vel)
                if np.linalg.norm(vel) > 0.5:  # If fast, compute angle with v
                    car_angle = -math.atan2(vel[0], vel[1])
                else:  # If slow, compute with hull
                    car_angle = car.hull.angle

                # Map angle to [0, 2pi] interval
                car_angle = (car_angle + (2 * np.pi)) % (2 * np.pi)

                # Retrieve car position
                car_pos = np.array(car.hull.position).reshape((1, 2))
                car_pos_as_point = Point((float(car_pos[:, 0]),
                                          float(car_pos[:, 1])))

                # Predict car position in next 100 steps
                next_pos = car_pos + 10 * vel/FPS
                next_pos_as_point = Point((float(next_pos[:, 0]),
                                          float(next_pos[:, 1])))

                # Compute closest point on track to car position (l2 norm)
                distance_to_tiles = np.linalg.norm(
                    car_pos - np.array(self.track)[:, 2:], ord=2, axis=1)
                track_index = np.argmin(distance_to_tiles)

                # Check if car is driving on grass by checking inside polygons
                on_grass = not np.array([car_pos_as_point.within(polygon)
                                   for polygon in self.road_poly_shapely]).any()
                self.driving_on_grass[car_id] = on_grass

                # Find track angle of closest point
                desired_angle = self.track[track_index][1]

                # If track direction reversed, reverse desired angle
                if self.episode_direction == 'CW':  # CW direction indicates reversed
                    desired_angle += np.pi

                # Map angle to [0, 2pi] interval
                desired_angle = (desired_angle + (2 * np.pi)) % (2 * np.pi)

                # Compute smallest angle difference between desired and car
                angle_diff = abs(desired_angle - car_angle)
                if angle_diff > np.pi:
                    angle_diff = abs(angle_diff - 2 * np.pi)

                # If car is driving backward and not on grass, penalize car. The
                # backwards flag is set even if it is driving on grass.
                if angle_diff > BACKWARD_THRESHOLD:
                    self.driving_backward[car_id] = True
                    step_reward[car_id] -= K_BACKWARD * angle_diff
                else:
                    self.driving_backward[car_id] = False

                #print("SPEED:", self.speed[car_id], "ANGLE_DIFF:", angle_diff, end="\r")

            # If all tiles were visited
            if len(self.track) in self.tile_visited_count:
                done = True

            # Terminate the episode if a car leaves the field, spends too much
            # time on grass or the time limit is reached
            for car_id, car in enumerate(self.cars):
                x, y = car.hull.position
                if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                    done = True
                    step_reward[car_id] = -100
                if self.elapsed_time > self.f1_track.max_episode_steps:
                    done = True

        # Calculate step reward
        step_reward = self.reward - self.prev_reward
        self.prev_reward = self.reward.copy()

        if self.render_mode == "human":
            self.render(self.render_mode)

        print("Reward:", np.round(self.reward), "Steps:", self.elapsed_time, "         ", end="\r")

        # Convert step_reward to a dictionary
        step_reward = {car_id: step_reward[i] for i, car_id in enumerate(self.agents)}

        if dict_obs_space:
            observations = {car_id: {"observation": preprocess(self.state[i], self.grayscale),
                                    "speed": self.speed[i]/100.} for  i, car_id in enumerate(self.agents)}
        else:
            observations = {car_id: preprocess(self.state[i], self.grayscale) for i, car_id in enumerate(self.agents)}

        terminations = {car_id: done for car_id in self.agents}
        truncations = {car_id: done for car_id in self.agents}

        infos = {car_id: {} for car_id in self.agents}

        if done and self.verbose == 1:
            print(f"\nAgent {car_id} reward: {self.reward[car_id]:.1f}\n")

        # If no actions are passed
        if actions is None:
            return observations, infos
        else:
            return observations, step_reward, terminations, truncations, infos

    def render(self, mode='state_pixels'):
        assert mode in ['human', 'state_pixels', 'rgb_array']

        result = []
        for cur_car_id in range(self.n_agents):
            result.append(self._render_window(cur_car_id, mode))
        
        return np.stack(result, axis=0)

    def _render_window(self, car_id, mode):
        """ Performs the actual rendering for each car individually. 
        
        Parameters:
            car_id(int): Numerical id of car for which the corresponding window
                will be rendered.
            mode(str): Rendering mode.
        """

        if self.viewer[car_id] is None:
            from gym.envs.classic_control import rendering
            self.viewer[car_id] = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.viewer[car_id].window.set_caption(f"Car {car_id}")
            self.score_label = pyglet.text.Label('0000', font_size=36,
                x=20, y=WINDOW_H*2.5/40.00, anchor_x='left', anchor_y='center',
                color=(255,255,255,255))
            self.transform = rendering.Transform()

        if "t" not in self.__dict__: return  # reset() not called yet

        zoom = 0.1*SCALE*max(1-self.t, 0) + ZOOM*SCALE*min(self.t, 1)   # Animate zoom first second
        #NOTE (ig): Following two variables seemed unused. Commented them out.
        #zoom_state  = ZOOM*SCALE*STATE_W/WINDOW_W 
        #zoom_video  = ZOOM*SCALE*VIDEO_W/WINDOW_W
        scroll_x = self.cars[car_id].hull.position[0]
        scroll_y = self.cars[car_id].hull.position[1]
        angle = -self.cars[car_id].hull.angle
        vel = self.cars[car_id].hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)

        # Positions car in the center with regard to the window width and 1/4 height away from the bottom.
        self.transform.set_translation(
            WINDOW_W/2 - (scroll_x*zoom*math.cos(angle) - scroll_y*zoom*math.sin(angle)),
            WINDOW_H * self.h_ratio - (scroll_x*zoom*math.sin(angle) + scroll_y*zoom*math.cos(angle)) )
        self.transform.set_rotation(angle)

        # Set colors for each viewer and draw cars
        for id, car in enumerate(self.cars):
            if self.use_ego_color:  # Apply same ego car color coloring scheme
                car.hull.color = (0.0, 0.0, 0.8)  # Set all other car colors to blue
                if id == car_id:  # Ego car
                    car.hull.color = (0.8, 0.0, 0.0)  # Set ego car color to red
            car.draw(self.viewer[car_id], mode != "state_pixels")

        arr = None
        win = self.viewer[car_id].window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode=='rgb_array':
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == 'state_pixels':
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, '_nscontext'):
                pixel_scale = win.context._nscontext.view().backingScaleFactor()  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer[car_id].onetime_geoms:
           geom.render()
        self.viewer[car_id].onetime_geoms = []
        t.disable()
        self.render_indicators(car_id, WINDOW_W, WINDOW_H)

        if mode == 'human':
            win.flip()
            return self.viewer[car_id].isopen

        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def close(self):
        if None not in self.viewer:
            for viewer in self.viewer:
                viewer.close()

        self.viewer = [None] * self.n_agents

    def render_road(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(self.bg_color[0], self.bg_color[1], self.bg_color[2], 1.0)  # 0.4, 0.8, 0.4
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        gl.glColor4f(self.grass_color[0], self.grass_color[1], self.grass_color[2], 1.0)  # 0.4, 0.9, 0.4
        k = PLAYFIELD/20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k*x + k, k*y + 0, 0)
                gl.glVertex3f(k*x + 0, k*y + 0, 0)
                gl.glVertex3f(k*x + 0, k*y + k, 0)
                gl.glVertex3f(k*x + k, k*y + k, 0)
        for poly, color in self.road_poly:
            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)
        
        gl.glEnd()

    def render_indicators(self, agent_id, W, H):
        gl.glBegin(gl.GL_QUADS)
        s = W/40.0
        h = H/40.0
        gl.glColor4f(0,0,0,1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5*h, 0)
        gl.glVertex3f(0, 5*h, 0)
        gl.glVertex3f(0, 0, 0)
        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0)*s, h + h*val, 0)
            gl.glVertex3f((place+1)*s, h + h*val, 0)
            gl.glVertex3f((place+1)*s, h, 0)
            gl.glVertex3f((place+0)*s, h, 0)
        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0)*s, 4*h , 0)
            gl.glVertex3f((place+val)*s, 4*h, 0)
            gl.glVertex3f((place+val)*s, 2*h, 0)
            gl.glVertex3f((place+0)*s, 2*h, 0)
        true_speed = np.sqrt(np.square(self.cars[agent_id].hull.linearVelocity[0]) \
            + np.square(self.cars[agent_id].hull.linearVelocity[1]))
        vertical_ind(5, 0.02*true_speed, (1,1,1))
        vertical_ind(7, 0.01*self.cars[agent_id].wheels[0].omega, (0.0,0,1)) # ABS sensors
        vertical_ind(8, 0.01*self.cars[agent_id].wheels[1].omega, (0.0,0,1))
        vertical_ind(9, 0.01*self.cars[agent_id].wheels[2].omega, (0.2,0,1))
        vertical_ind(10,0.01*self.cars[agent_id].wheels[3].omega, (0.2,0,1))
        horiz_ind(20, -10.0*self.cars[agent_id].wheels[0].joint.angle, (0,1,0))
        horiz_ind(30, -0.8*self.cars[agent_id].hull.angularVelocity, (1,0,0))
        gl.glEnd()
        self.score_label.text = "%04i" % self.reward[agent_id]
        self.score_label.draw()

        # Render backwards flag if driving backward and backwards flag render is enabled
        if self.driving_backward[agent_id] and self.backwards_flag:
            pyglet.graphics.draw(3, gl.GL_TRIANGLES,
                                 ('v2i', (W-100, 30,
                                          W-75, 70,
                                          W-50, 30)),
                                 ('c3B', (0, 0, 255) * 3))
            

if __name__=="__main__":
    from pyglet.window import key
    NUM_CARS = 2  # Supports key control of two cars, but can simulate as many as needed

    discrete_action_space = False

    # Specify key controls for cars
    CAR_CONTROL_KEYS = [[key.LEFT, key.RIGHT, key.UP, key.DOWN],
                        [key.A, key.D, key.W, key.S]]

    actions = {f"car_{i}": 0 for i in range(NUM_CARS)} if discrete_action_space else {f"car_{i}": np.zeros(3) for i in range(NUM_CARS)}
    def key_press(k, mod):
        global restart, stopped, CAR_CONTROL_KEYS
        if k==0xff1b: stopped = True # Terminate on esc.
        if k==0xff0d: restart = True # Restart on Enter.

        # Iterate through cars and assign them control keys (mod num controllers)
        for i, car_id in enumerate(actions.keys()):
            if discrete_action_space:
                if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][0]: actions[car_id] = 2  # Left
                if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][1]: actions[car_id] = 1  # Right
                if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][2]: actions[car_id] = 3  # Gas
                if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][3]: actions[car_id] = 4  # Brake
            else:
                if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][0]: actions[car_id][0] = -1.0
                if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][1]: actions[car_id][0] = +1.0
                if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][2]: actions[car_id][1] = +1.0
                if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][3]: actions[car_id][2] = +0.8   # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        global CAR_CONTROL_KEYS

        # Iterate through cars and assign them control keys (mod num controllers)
        for i, car_id in enumerate(actions.keys()):
            if discrete_action_space:
                if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][0]: actions[car_id] = 0  # Do nothing
                if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][1]: actions[car_id] = 0  # Do nothing
                if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][2]: actions[car_id] = 0  # Do nothing
                if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][3]: actions[car_id] = 0  # Do nothing
            else:
                if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][0] and actions[car_id][0]==-1.0: actions[car_id][0] = 0
                if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][1] and actions[car_id][0]==+1.0: actions[car_id][0] = 0
                if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][2]: actions[car_id][1] = 0
                if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][3]: actions[car_id][2] = 0

    track = TRACK

    env = parallel_env(track, n_agents=NUM_CARS, use_random_direction=False,
                       backwards_flag=True, verbose=1, discrete_action_space=discrete_action_space,
                       domain_randomize=DOMAIN_RANDOMIZE)
    env.render("human")
    for viewer in env.viewer:
        viewer.window.on_key_press = key_press
        viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor
        env = Monitor(env, '/tmp/video-test', force=True)
    isopen = True
    stopped = False
    while isopen and not stopped:
        env.reset()        
        total_reward = {f"car_{i}": 0 for i in range(NUM_CARS)}
        steps = 0
        restart = False
        while True:
            obs, r, done, _, _ = env.step(actions)
            for car_id, _ in total_reward.items():
                total_reward[car_id] += r[car_id]
            if done["car_0"]:
                print("\nActions: " + str.join(" ", [f"Car {car_id}: "+str(action) for car_id, action in actions.items()]))
                print(f"Step {steps} Total_reward "+str(total_reward))
            steps += 1
            isopen = env.render("human").all()
            if stopped or done["car_0"] or restart or isopen == False:
                break
    env.close()