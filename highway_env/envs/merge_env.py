from __future__ import annotations

import numpy as np
import pygame

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.vehicle.behavior import MergeIDMVehicle, NotiIDMVehicle

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
        self.merge_lane_indices = [('b', 'c', 1), ('c0', 'd', 0), ('d', 'e', 0), ('e', 'f', 1), ('f', 'g', 1)]
        super().__init__(*args, **kwargs)

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {'type': 'NotiAction'},
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 8,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-10, 750],
                        "y": [-20, 20],
                        "vx": [-5, 40],
                        "vy": [-5, 40]
                    },
                    "absolute": False,
                    "order": "sorted"
                },
                "vehicle_class": "NotiIDMVehicle",
                "target_speeds": [5, 10, 15, 20, 25, 30, 35, 40],
                "human_utterance_memory_length": 10,
                "merge_vehicles_type": "highway_env.vehicle.behavior.MergeIDMVehicle",
                "max_episode_steps": 1000,
                "reward_speed_range": [5, 40],
                "high_speed_reward": 1.0,
                "collision_reward": -2.0,
                "progress_reward": 0.1,
                "completion_bonus": 5.0,
                "merging_speed_reward": -0.3,
                "lane_change_reward": -0.05,
                "right_lane_reward": 0.2,
                "noti_penalty": -0.3,
            }
        )
        return cfg
    
    def _reset(self) -> None:
        super()._reset()
        self.noti_history = []
        self.curr_agent_action = None
        self.max_history_size = 10
        self.overwrite_flag = False

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
            a = action[3]
            noti_action = np.array(action[:3])
            overwrite_flag = action[-1] if len(action) > 4 else 0
            
            # Store notification in history
            if noti_action[0] == 0: # no notification
                noti_action = np.array([0,0,0])
            elif noti_action[0] == 1: # continue previous notification
                noti_action = np.array([1,0,0])
            else:
                # ACTIONS_ALL = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"} --> but we remove IDLE, so the index now represents: 0: "LANE_LEFT", 1: "LANE_RIGHT", 2: "FASTER", 3: "SLOWER"
                if noti_action[1] > 0:
                    noti_action[1] = noti_action[1] + 1

                # process notification length: 2 or 5
                noti_action[-1] = (noti_action[-1]*3) + 2
            
            self.noti_history.append(noti_action)
            self.curr_agent_action = a
            self.overwrite_flag = overwrite_flag

            if len(self.noti_history) > self.max_history_size:
                self.noti_history.pop(0)

            if isinstance(action, np.ndarray):
                info.update({
                    "action": a,
                    "utterance": np.array(noti_action, dtype=np.int64),
                    "overwrite_flag": overwrite_flag,
                })
            else:
                info.update({
                    "action": action,
                    "utterance": np.zeros(3, dtype=np.int64),  # Default utterance size is 3
                    "overwrite_flag": False,
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
                StraightLane([sum(self.ends), y[i]], [sum(self.ends)+self.ends[0], y[i]], line_types=line_type_merge_left[i])
            )
            net.add_lane(
                "e",
                "f",
                StraightLane([sum(self.ends)+self.ends[0], y[i]], [sum(self.ends)+sum(self.ends[:2]), y[i]], line_types=line_type_merge_right[i])
            )
            net.add_lane(
                "f",
                "g",
                StraightLane([sum(self.ends)+sum(self.ends[:2]), y[i]], [sum(self.ends)+sum(self.ends[:3]), y[i]], line_types=line_type_merge_right[i])
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
        lde0 = StraightLane(
            lde.position(self.ends[1], 0),  # Start position
            lde.position(self.ends[1], 0) + [self.ends[2], 0],  # End position with merge section
            line_types=[c, s],
            forbidden=True,
        )
        net.add_lane("m", "n", lmn)
        net.add_lane("n", "c", lnb)
        net.add_lane("c0", "d", lde)
        net.add_lane("d", "e", lde0)

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
        lef0 = StraightLane(
            lef.position(self.ends[1], 0),
            lef.position(self.ends[1], 0) + [self.ends[2], 0],  # Longer merge section
            line_types=[n, c],
            forbidden=True,
        )
        net.add_lane("p", "q", lpq)
        net.add_lane("q", "d", lqb)
        net.add_lane("e", "f", lef)
        net.add_lane("f", "g", lef0)

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        road.objects.append(Obstacle(road, lbc.position(self.ends[2], 0)))  # First merge point
        road.objects.append(Obstacle(road, lde.position(self.ends[3], 0)))  # Second merge point
        road.objects.append(Obstacle(road, lef.position(self.ends[3], 0)))  # Third merge point
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
            target_speeds=self.config.get("target_speeds", [5, 10, 15, 20, 25, 30, 35, 40])
        )
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        merge_vehicles_type = utils.class_from_path(self.config["merge_vehicles_type"])

        for position, speed in [(100, 31), (120, 29.5), (140, 30.0)]:#,  (70, 31), (5, 31.5), (170, 27.5)]:
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
            road, road.network.get_lane(("p", "q", 0)).position(0, 0), speed=1
        )
        merging_v_right2.target_speed = 16
        road.vehicles.append(merging_v_right2)

        self.vehicle = ego_vehicle

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when all merge points have been passed."""
        total_length = sum(self.ends) + self.one_merge_length * 2  # Total road length including all merge sections
        return self.vehicle.crashed or bool(self.vehicle.position[0] > total_length) or self.vehicle.speed < 1

    def _is_truncated(self) -> bool:
        """The episode is truncated when exceeding the maximum number of steps."""
        return self.time >= self.config["max_episode_steps"] - 1
    
    
    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed and avoiding collisions.
        Additional rewards are given for making progress along the road and successfully completing the environment.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        reward = sum(
            self.config.get(name, 0) * reward
            for name, reward in self._rewards(action).items()
        )
        
        # Add completion bonus if the episode is about to end due to reaching the end of the road
        if self._is_terminated() and not self.vehicle.crashed and self.vehicle.speed >= 1:
            reward += self.config["completion_bonus"]
            
        return reward

    def _rewards(self, action: int) -> dict[str, float]:
        # Scale speed reward based on the configured range
        scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        
        # Calculate progress reward based on x position
        total_length = sum(self.ends) + self.one_merge_length * 2
        progress = self.vehicle.position[0] / total_length
        
        # Calculate lane change penalty
        lane_change = action in [0, 2] if action is not None else False
        
        # Calculate right lane reward (1.0 when in right lane, 0.0 when in left lane)
        right_lane = 1.0 if self.vehicle.lane_index[2] == 1 else 0.0
        
        # Calculate merging speed penalty (reduced from original)
        merging_speed_penalty = sum(
            (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
            for vehicle in self.road.vehicles
            if vehicle.lane_index in self.merge_lane_indices
            and isinstance(vehicle, NotiIDMVehicle)
        )
        
        return {
            "collision_reward": self.vehicle.crashed,
            "high_speed_reward": scaled_speed,
            "progress_reward": progress,
            "lane_change_reward": lane_change,
            "merging_speed_reward": merging_speed_penalty,
            "right_lane_reward": right_lane*self.config["right_lane_reward"],
        }

    def _noti_reward(self, utterance: np.ndarray) -> float:
        """
        Calculate the reward for the notification.
        """
        if utterance[0] == 2:
            return self.config["noti_penalty"]
        return 0
    
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
        self.time += 1 / self.config["policy_frequency"]
        # action = None
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, joint_action)
        noti_reward = self._noti_reward(info["utterance"])
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
    
    def render_text_info(self, text_info: dict, surface: pygame.Surface) -> None:
        """Render text information on the given surface."""
        font = pygame.font.Font(None, 20)  # Smaller font size
        y_offset = 5
        x_offset = 5
        line_spacing = 20  # Reduced line spacing
        
        # Calculate the maximum text width and total height needed
        max_text_width = 0
        total_lines = 0
        for text in text_info.keys():
            if "\n" in text:
                lines = text.split("\n")
                total_lines += len(lines)
                for line in lines:
                    text_surface = font.render(line, True, (0, 0, 0))
                    max_text_width = max(max_text_width, text_surface.get_width())
            else:
                total_lines += 1
                text_surface = font.render(text, True, (0, 0, 0))
                max_text_width = max(max_text_width, text_surface.get_width())
        
        # Create background
        text_height = total_lines * line_spacing + 5
        text_width = min(max_text_width + 20, surface.get_width() - 10)
        background = pygame.Surface((text_width, text_height), pygame.SRCALPHA)
        background.fill((255, 255, 255, 180))
        surface.blit(background, (x_offset, y_offset))
        
        # Draw text
        current_y = y_offset + 2
        for text in text_info.keys():
            if "\n" in text:
                lines = text.split("\n")
                for line in lines:
                    # Choose color based on whether it's an overwritten action
                    color = (255, 0, 0) if "OVERWRITTEN" in line else (0, 0, 0)
                    text_surface = font.render(line.strip(), True, color)
                    surface.blit(text_surface, (x_offset + 5, current_y))
                    current_y += line_spacing
            else:
                # Choose color based on whether it's an overwritten action
                color = (255, 0, 0) if "OVERWRITTEN" in text else (0, 0, 0)
                text_surface = font.render(text, True, color)
                surface.blit(text_surface, (x_offset + 5, current_y))
                current_y += line_spacing
        
        # Render rewards on the right side
        if "rewards_info" in text_info:
            rewards_text = text_info["rewards_info"]
            rewards_lines = rewards_text.split("\n")
            
            # Calculate position for rewards (right side)
            rewards_x_offset = surface.get_width() - max_text_width - 30
            rewards_y_offset = y_offset
            
            # Create background for rewards
            rewards_height = len(rewards_lines) * line_spacing + 5
            rewards_width = min(max_text_width + 20, surface.get_width() - rewards_x_offset - 10)
            rewards_background = pygame.Surface((rewards_width, rewards_height), pygame.SRCALPHA)
            rewards_background.fill((255, 255, 255, 180))
            surface.blit(rewards_background, (rewards_x_offset, rewards_y_offset))
            
            # Draw rewards text
            current_y = rewards_y_offset + 2
            for line in rewards_lines:
                text_surface = font.render(line.strip(), True, (0, 0, 0))
                surface.blit(text_surface, (rewards_x_offset + 5, current_y))
                current_y += line_spacing
    
    def get_text_info(self) -> dict:
        """Return a dictionary of text information to display."""
        # Format position and velocity string
        position_str = f"X: {self.vehicle.position[0]:.2f} Y: {self.vehicle.position[1]:.2f} VX: {self.vehicle.velocity[0]:.2f} VY: {self.vehicle.velocity[1]:.2f}"
        
        # Get the current action name and color
        action_names = {0: "LANE_LEFT", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
        action_name = action_names.get(self.curr_agent_action, "IDLE") if self.curr_agent_action is not None else "NONE"
        
        # Set color based on overwrite flag first, then action type
        if hasattr(self, 'overwrite_flag') and self.overwrite_flag:
            action_color = (255, 0, 0)  # Bright red for overwritten actions
        else:
            # Normal color scheme based on action type
            # if self.curr_agent_action in [0, 2]:  # Lane changes
            #     action_color = (0, 0, 255)  # Blue
            # elif self.curr_agent_action == 3:  # Faster
            #     action_color = (50, 200, 0)    # Green
            # elif self.curr_agent_action == 4:  # Slower
            #     action_color = (255, 165, 0)  # Orange
            # else:  # IDLE or NONE
            action_color = (255, 255, 0)    # Yellow
            
        human_str = f"Human: {action_name}"
        
        # Current notification
        noti_str = "Noti: "
        if hasattr(self, 'noti_history') and self.noti_history:
            current_noti = self.noti_history[-1]
            noti_str += f"[{current_noti[0]} {current_noti[1]} {current_noti[2]}]"
        else:
            noti_str += "[0 0 0]"

        # Notification history
        # noti_history_str = "Noti History:"
        if hasattr(self, 'noti_history') and self.noti_history:
            history = self.noti_history[-9:] if len(self.noti_history) > 9 else self.noti_history
            for i, noti in enumerate(history[::-1], 1):
                noti_str += f"\n{i}: [{noti[0]} {noti[1]} {noti[2]}]"
            # Pad with empty notifications if less than 9
            for i in range(len(history) + 1, 10):
                noti_str += f"\n{i}: [0 0 0]"
        else:
            for i in range(1, 10):
                noti_str += f"\n{i}: [0 0 0]"

        # Calculate rewards
        rewards = self._rewards(self.curr_agent_action if self.curr_agent_action is not None else None)
        noti_reward = self._noti_reward(self.noti_history[-1])
        total_reward = self._reward(self.curr_agent_action if self.curr_agent_action is not None else None) + noti_reward
        reward_str = f"Total Reward: {total_reward:.2f}"
        
        # Create detailed rewards info
        rewards_info = "Rewards:"
        for name, value in rewards.items():
            # Format the reward name for display
            display_name = name.replace("_", " ").title()
            rewards_info += f"\n{display_name}: {value:.2f}"
        rewards_info += f"\nNoti Reward: {noti_reward:.2f}"
        rewards_info += f"\nTotal: {total_reward:.2f}"
        
        return {
            position_str: (255, 255, 255),  # White
            human_str: action_color,        # Color based on action type
            noti_str: (255, 255, 255),       # White
            # noti_history_str: (255, 255, 255),  # White
            reward_str: (255, 255, 255),    # White
            "rewards_info": rewards_info    # Detailed rewards info
        }
        
    def render_notification_history(self, surface: pygame.Surface) -> None:
        """Render the notification history in a visual way."""
        if not hasattr(self, 'noti_history') or not self.noti_history:
            return
            
        # Define colors for different notification types
        colors = {
            0: (200, 200, 200),  # Gray for no notification
            1: (150, 150, 255),  # Light blue for continue previous
            2: (255, 200, 100),  # Orange for lane left
            3: (100, 255, 100),  # Green for lane right
            4: (255, 100, 100),  # Red for faster
            5: (100, 100, 255),  # Blue for slower
        }
        
        # Position and size for the history display
        x_offset = 10
        y_offset = 200  # Position below the text info
        bar_width = 20
        bar_spacing = 5
        bar_height = 30
        
        # Draw a background for the history
        history_width = len(self.noti_history) * (bar_width + bar_spacing) + bar_spacing
        background = pygame.Surface((history_width, bar_height + 10), pygame.SRCALPHA)
        background.fill((255, 255, 255, 180))
        surface.blit(background, (x_offset, y_offset))
        
        # Draw a border
        pygame.draw.rect(surface, (0, 0, 0), (x_offset, y_offset, history_width, bar_height + 10), 1)
        
        # Draw a title
        font = pygame.font.Font(None, 20)
        title = font.render("Notification History", True, (0, 0, 0))
        surface.blit(title, (x_offset + 5, y_offset - 20))
        
        # Draw each notification as a colored bar
        for i, noti in enumerate(self.noti_history):
            color_key = noti[0] if noti[0] < 2 else noti[1] + 2
            color = colors.get(color_key, (200, 200, 200))
            
            # Draw the bar
            bar_x = x_offset + i * (bar_width + bar_spacing) + bar_spacing
            pygame.draw.rect(surface, color, (bar_x, y_offset + 5, bar_width, bar_height))
            
            # Draw a border around the bar
            pygame.draw.rect(surface, (0, 0, 0), (bar_x, y_offset + 5, bar_width, bar_height), 1)
    