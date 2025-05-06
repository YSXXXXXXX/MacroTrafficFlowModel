from __future__ import annotations

from typing import Dict, List
import numpy as np


class BaseLink(object):

    _states = set()

    def __init__(self, name: str, link_type: str, lane_num: int, segment_num: int, segment_length: np.ndarray,
                 from_node_id: str, to_node_id: str):
        self.name = name
        self.link_type = link_type
        self.lane_num = lane_num
        self.segment_num = segment_num
        self.segment_length = segment_length
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.from_node_entity = None
        self.to_node_entity = None

        self.state = {state: np.zeros(segment_num, dtype=np.float32) for state in self._states}
        self.next_state = {state: np.zeros(segment_num, dtype=np.float32) for state in self._states}

    def get_entity(self, id2entity: Dict):
        self.from_node_entity = id2entity[self.from_node_id]
        self.to_node_entity = id2entity[self.to_node_id]

    def step(self):
        raise NotImplementedError("No implementation for method: step")

    def update(self):
        for key in self.next_state.keys():
            self.state[key] = self.next_state[key]

    @staticmethod
    def clip_state(state, lower_bound: float = 0., upper_bound: float = np.inf):
        for key in state.keys():
            state[key] = np.clip(state[key], lower_bound, upper_bound)

    def refresh(self, new_state: Dict[str, np.ndarray]):
        raise NotImplementedError("No implementation for method: refresh")


class BaseNode(object):
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.upstream_link: List = []  # the normal upstream link (not include origin)
        self.origin_link: List = []  # the origin link
        self.downstream_link: List = []

    def get_entity(self, id2entity):
        for i, predecessor in enumerate(self.upstream_link):
            self.upstream_link[i] = id2entity[(predecessor, self.node_id)]
        for i, predecessor in enumerate(self.origin_link):
            self.origin_link[i] = id2entity[(predecessor, self.node_id)]
        for i, successor in enumerate(self.downstream_link):
            self.downstream_link[i] = id2entity[(self.node_id, successor)]

    def get_upstream_flow(self):
        raise NotImplementedError("No implementation for method: get_upstream_flow")

    def get_upstream_speed(self):
        raise NotImplementedError("No implementation for method: get_upstream_speed")

    def get_downstream_speed(self):
        raise NotImplementedError("No implementation for method: get_downstream_speed")


class BaseOrigin(object):

    _states = set()
    _state = set()
    _action = set()

    def __init__(self, name: str, to_node_id: str):
        self.name = name
        self.to_node_id = to_node_id
        self.to_node_entity = None
        self.downstream_link = None
        self.state = {key: np.zeros(1, dtype=np.float32) for key in self._states}
        self.state.update({key: 0. for key in self._state})
        self.next_state = {key: np.zeros(1, dtype=np.float32) for key in self._states}

    def get_entity(self, id2entity: Dict):
        self.to_node_entity = id2entity[self.to_node_id]

    def refresh(self, new_state: Dict, disturbance, action=None):
        raise NotImplementedError("No implementation for method: refresh")

    def step(self):
        raise NotImplementedError("No implementation for method: step")

    def update(self):
        for key in self.next_state.keys():
            self.state[key] = self.next_state[key]

    def get_flow(self):
        raise NotImplementedError("No implementation for method: get_flow")

    def get_speed(self):
        raise NotImplementedError("No implementation for method: get_speed")

    def get_density(self):
        raise NotImplementedError("No implementation for method: get_density")

    @staticmethod
    def clip_state(state, lower_bound: float = 0., upper_bound: float = np.inf):
        for key in state.keys():
            state[key] = np.clip(state[key], lower_bound, upper_bound)


class BaseEngine(object):
    def __init__(self):
        self.segment_num = 0  # total segment number (sum of all links)
        self.links: Dict = {}
        self.origins: Dict = {}
        self.nodes: Dict = {}

    @property
    def model_step(self):
        raise NotImplementedError("No implementation for method model_step")

    def refresh(self, new_dict: Dict = None, demand: Dict = None, merge_rate: Dict = None, speed_limit: Dict = None):
        raise NotImplementedError("No implementation for method refresh")

    def step(self):
        raise NotImplementedError("No implementation for method step")

    def get_state(self):
        raise NotImplementedError("No implementation for method get_state")

    def reset(self):
        raise NotImplementedError("No implementation for method rest")

    def get_tts(self):
        raise NotImplementedError("No implementation for method get_tts")

    @staticmethod
    def init_model_parameters():
        raise NotImplementedError("No implementation for method init_model_parameters")

    @staticmethod
    def change_model_parameters(new_parameters: np.ndarray):
        raise NotImplementedError("Not implementation for method change_model_parameters")
