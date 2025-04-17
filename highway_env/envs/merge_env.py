from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle


class MergeEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "collision_reward": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.2,
                "reward_speed_range": [15, 38],
                "merging_speed_reward": -0.5,
                "lane_change_reward": -0.05,
            }
        )
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        reward = sum(
            self.config.get(name, 0) * reward
            for name, reward in self._rewards(action).items()
        )
        return utils.lmap(
            reward,
            [
                self.config["collision_reward"] + self.config["merging_speed_reward"],
                self.config["high_speed_reward"] + self.config["right_lane_reward"],
            ],
            [0, 1],
        )

    def _rewards(self, action: int) -> dict[str, float]:
        scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": self.vehicle.crashed,
            "right_lane_reward": self.vehicle.lane_index[2] / 1,
            "high_speed_reward": scaled_speed,
            "lane_change_reward": action in [0, 2],
            "merging_speed_reward": sum(  # Altruistic penalty
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if vehicle.lane_index == ("b", "c", 2)
                and isinstance(vehicle, ControlledVehicle)
            ),
        }

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane(
                "a",
                "b",
                StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [sum(ends[:2]), y[i]],
                    [sum(ends[:3]), y[i]],
                    line_types=line_type_merge[i],
                ),
            )
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]
                ),
            )

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane(
            [0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True
        )
        lkb = SineLane(
            ljk.position(ends[0], -amplitude),
            ljk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lbc = StraightLane(
            lkb.position(ends[1], 0),
            lkb.position(ends[1], 0) + [ends[2], 0],
            line_types=[n, c],
            forbidden=True,
        )
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 1)).position(30, 0), speed=30
        )
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        for position, speed in [(90, 29), (70, 31), (5, 31.5)]:
            lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
            position = lane.position(position + self.np_random.uniform(-5, 5), 0)
            speed += self.np_random.uniform(-1, 1)
            road.vehicles.append(other_vehicles_type(road, position, speed=speed))

        merging_v = other_vehicles_type(
            road, road.network.get_lane(("j", "k", 0)).position(110, 0), speed=20
        )
        merging_v.target_speed = 30
        road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle

class MultiMergeEnv(MergeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {'type': 'NotiAction'},
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 10,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 200],
                        "y": [-100, 200],
                        "vx": [-20, 30],
                        "vy": [-20, 30]
                    },
                    "absolute": False,
                    "order": "sorted"
                },
                "vehicle_class": "NotiIDMVehicle",
                "target_speeds": [20, 30, 40],
                "human_utterance_memory_length": 10,
                "merge_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle",
            }
        )
        return cfg

    def _info(self, obs: Observation, action: Action | None = None) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
        }
        if action is not None:
            if isinstance(action, np.ndarray):
                info.update({
                    "action": action[-1],
                    "utterance": np.array(action[:-1], dtype=np.float32),
                })
            else:
                info.update({
                    "action": action,
                    "utterance": np.zeros(3, dtype=np.float32),  # Default utterance size is 3
                })
        info["rewards"] = self._rewards(action[-1] if isinstance(action, np.ndarray) else action)
        return info

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and three sequential merging lanes.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        self.ends = [80, 80, 80, 160]  # Before, converging, merge, after
        self.one_merge_length = sum(self.ends[1:3])
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge_right = [[c, s], [n, s]]
        line_type_merge_left = [[n, s], [n, c]]
        for i in range(2):
            net.add_lane(
                "a",
                "b",
                StraightLane([0, y[i]], [sum(self.ends[:2]), y[i]], line_types=line_type[i]),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [sum(self.ends[:2]), y[i]],
                    [sum(self.ends[:3]), y[i]],
                    line_types=line_type_merge_right[i],
                ),
            )
            net.add_lane(
                "c",
                "c0",
                StraightLane(
                    [sum(self.ends[:3]), y[i]], [sum(self.ends[:3])+self.ends[0], y[i]], line_types=line_type[i]
                ),
            )
            net.add_lane(
                "c0",
                "d",
                StraightLane(
                    [sum(self.ends[:3])+self.ends[0], y[i]], [sum(self.ends), y[i]], line_types=line_type_merge_left[i]
                ),
            )
            net.add_lane(
                "d",
                "e",
                StraightLane([sum(self.ends), y[i]], [sum(self.ends)+self.ends[0], y[i]], line_types=line_type[i])
            )
            net.add_lane(
                "e",
                "f",
                StraightLane([sum(self.ends)+self.ends[0], y[i]], [sum(self.ends)+sum(self.ends[:2]), y[i]], line_types=line_type_merge_right[i])
            )
            net.add_lane(
                "f",
                "g",
                StraightLane([sum(self.ends)+sum(self.ends[:2]), y[i]], [sum(self.ends)+sum(self.ends[:3]), y[i]], line_types=line_type[i])
            )
            net.add_lane(
                "g",
                "h",
                StraightLane([sum(self.ends)+sum(self.ends[:3]), y[i]], [sum(self.ends)*2+self.ends[-1], y[i]], line_types=line_type[i])
            )

        # First right merging lane (merges first)
        amplitude = 3.25
        ljk = StraightLane(
            [0, 6.5 + 4 + 4], [self.ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True
        )
        lkb = SineLane(
            ljk.position(self.ends[0], -amplitude),
            ljk.position(sum(self.ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * self.ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lbc = StraightLane(
            lkb.position(self.ends[1], 0),
            lkb.position(self.ends[1], 0) + [self.ends[2], 0],  # Shorter merge section
            line_types=[n, c],
            forbidden=True,
        )
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)

        # Left merging lane (merges second)
        lmn = StraightLane(
            # [self.one_merge_length, -6.5 - 4 ], [self.one_merge_length+self.ends[0], -6.5 - 4 ], line_types=[c, c], forbidden=True
            [0, -6.5 - 4], [self.one_merge_length+self.ends[0], -6.5 - 4], line_types=[c, c], forbidden=True
            
        )
        lnb = SineLane(
            lmn.position(self.one_merge_length+self.ends[0], amplitude),
            lmn.position(self.one_merge_length+sum(self.ends[:2]), amplitude),
            amplitude,
            2 * np.pi / (2 * self.ends[1]),
            -np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lde = StraightLane(
            lnb.position(self.ends[1], 0),  # Start position
            lnb.position(self.ends[1], 0) + [self.ends[2], 0],  # End position with merge section
            line_types=[c, s],
            forbidden=True,
        )
        net.add_lane("m", "n", lmn)
        net.add_lane("n", "c", lnb)
        net.add_lane("c", "d", lde)

        # Second right merging lane (merges last)
        lpq = StraightLane(
            [self.one_merge_length, 6.5 + 4 + 4], [self.one_merge_length*2+self.ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True
        )
        lqb = SineLane(
            lpq.position(self.one_merge_length+self.ends[0], -amplitude),
            lpq.position(self.one_merge_length+sum(self.ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * self.ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lef = StraightLane(
            lqb.position(self.ends[1], 0),
            lqb.position(self.ends[1], 0) + [self.ends[2], 0],  # Longer merge section
            line_types=[n, c],
            forbidden=True,
        )
        net.add_lane("p", "q", lpq)
        net.add_lane("q", "d", lqb)
        net.add_lane("d", "e", lef)

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        road.objects.append(Obstacle(road, lbc.position(self.ends[2], 0)))  # First merge point
        road.objects.append(Obstacle(road, lde.position(self.ends[2], 0)))  # Second merge point
        road.objects.append(Obstacle(road, lef.position(self.ends[2], 0)))  # Third merge point
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lanes, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        
        # Create ego vehicle with target_speeds parameter
        ego_vehicle = self.action_type.vehicle_class(
            road, 
            road.network.get_lane(("a", "b", 1)).position(30, 0), 
            speed=30,
            target_speeds=self.config.get("target_speeds", [20, 30, 40])
        )
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        merge_vehicles_type = utils.class_from_path(self.config["merge_vehicles_type"])

        for position, speed in [(90, 29), (70, 31), (5, 31.5), (120, 29.5), (140, 30.0), (170, 27.5), (200, 27.5), (230, 30.5)]:
            lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
            position = lane.position(position + self.np_random.uniform(-5, 5), 0)
            speed += self.np_random.uniform(-1, 1)
            road.vehicles.append(other_vehicles_type(road, position, speed=speed))

        # First right merging vehicle (merges first)
        merging_v_right1 = merge_vehicles_type(
            road, road.network.get_lane(("j", "k", 0)).position(self.ends[0] - 20, 0), speed=15
        )
        merging_v_right1.target_speed = 30
        road.vehicles.append(merging_v_right1)

        # Left merging vehicle (merges second)
        merging_v_left = merge_vehicles_type(
            road, road.network.get_lane(("m", "n", 0)).position(self.ends[0] - 20, 0), speed=10
        )
        merging_v_left.target_speed = 30
        road.vehicles.append(merging_v_left)

        # Second right merging vehicle (merges last)
        merging_v_right2 = merge_vehicles_type(
            road, road.network.get_lane(("p", "q", 0)).position(self.ends[0] - 20, 0), speed=0
        )
        merging_v_right2.target_speed = 30
        road.vehicles.append(merging_v_right2)

        self.vehicle = ego_vehicle

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when all merge points have been passed."""
        total_length = sum(self.ends) + self.one_merge_length * 2  # Total road length including all merge sections
        return self.vehicle.crashed or bool(self.vehicle.position[0] > total_length)

    
    def step(self, joint_action):
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError(
                "The road and vehicle must be initialized in the environment implementation"
            )
        
        action = joint_action[-2]
        if action == -1:
            action = None
        overwrite_flag = joint_action[-1]
        self.time += 1 / self.config["policy_frequency"]
        action = None
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, joint_action[:-1])
        info["overwrite_flag"] = overwrite_flag
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info