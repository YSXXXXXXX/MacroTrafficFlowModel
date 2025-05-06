from typing import Dict

import networkx as nx
import numpy as np
from MacroscopicSimulation.baseClass import BaseLink, BaseOrigin
from MacroscopicSimulation.METANET.constant import HyperParam as Hp


class Link(BaseLink, Hp):

    _states = {"density", "speed"}

    def __init__(self, name: str, net: nx.DiGraph, turn_rate: float, from_node_id: str, to_node_id: str):
        lane_num = net.edges[(from_node_id, to_node_id)]["lane_num"]
        segment_num = net.edges[(from_node_id, to_node_id)]["segment_num"]
        segment_length = net.edges[(from_node_id, to_node_id)]["length"]
        link_type = net.edges[(from_node_id, to_node_id)]["type"]
        speed_limit_type = net.edges[(from_node_id, to_node_id)]["speed_limit_type"]
        super().__init__(name, link_type, lane_num, segment_num, segment_length, from_node_id, to_node_id)
        self.speed_limit_type = speed_limit_type
        self.turn_rate = turn_rate

    def refresh(self, state: Dict = None):
        if state is None:
            return
        # FIXME: hard code here!
        self.state["density"] = state["density"]
        self.state["speed"] = state["ctrl_step_speed"]
        self.clip_state(self.state)

    def step(self):
        q = self.get_flow()
        rho = self.state["density"]
        v = self.state["speed"]
        # get upstream flow and speed of the link (use to update segment [0])
        q0, v0 = self.from_node_entity.get_upstream_flow_and_speed(self.turn_rate)
        # get downstream density of the link (use to update segment [-1])
        rho_n = self.to_node_entity.get_downstream_density()
        q_up = np.array([q0], dtype=np.float32)
        v_up = np.array([v0], dtype=np.float32)
        rho_down = np.array([rho_n], dtype=np.float32)
        if self.segment_num > 1:
            q_up = np.concatenate((q_up, q[:-1]))
            v_up = np.concatenate((v_up, self.state["speed"][:-1]))
            rho_down = np.concatenate((self.state["density"][1:], rho_down))
        if "ramp" not in self.link_type and self.from_node_entity.ramp_flag is not None:
            # an on ramp merge to this link
            q_ramp = self.from_node_entity.upstream_link[self.from_node_entity.ramp_flag].get_flow()[0]
        else:
            q_ramp = 0
        if self.to_node_entity.downstream_num == 1:
            # detector lane drop bottleneck when current link only has one outgoing link
            delta_lane = max(self.lane_num - self.to_node_entity.downstream_link[0].lane_num, 0)
        else:
            delta_lane = 0
        rho_next = rho + self.T * (q_up - q) / (self.lane_num * self.segment_length)
        v_expect = self.get_expect_speed(rho)
        relaxation = (self.T / self.tau) * (v_expect - v)
        convection = self.T * v / self.segment_length * (v_up - v)
        slow_to_start = rho_down < rho
        if "onramp" in self.link_type:
            eta = self.eta_ramp
        else:
            eta = slow_to_start * self.eta_low + (1 - slow_to_start) * self.eta_high  # slow start is True, use eta low
        anticipation = eta * self.T / (self.tau*self.segment_length) * (rho_down-rho) / (rho+self.kappa)
        v_next = v + relaxation + convection - anticipation
        # the merge action (upstream) will influence the first segment
        v_next[0] -= (self.delta * self.T * q_ramp * v[0]) / (self.lane_num * self.segment_length[0] * (rho[0] + self.kappa))
        # FIXME: assume that only mainline has lane drop bottleneck
        # the lane drop bottleneck (downstream) will influence the last segment
        v_next[-1] -= (self.phi * self.T * delta_lane * rho[-1] * v[-1] ** 2) / (self.lane_num * self.segment_length[-1] * self.rho_cri)
        self.next_state["density"], self.next_state["speed"] = rho_next, v_next
        self.clip_state(self.next_state)
        return self.next_state["density"], self.next_state["speed"]

    def get_expect_speed(self, rho):
        if "onramp" in self.speed_limit_type:
            return self.ramp_vf * np.exp((-1/self.am) * (rho / self.ramp_rho_cri) ** self.am)
        else:
            return self.vf * np.exp((-1/self.am) * (rho / self.rho_cri) ** self.am)

    def get_flow(self) -> np.ndarray:
        # get last step flow
        return self.state["density"] * self.state["speed"] * self.lane_num

    def get_speed(self) -> np.ndarray:
        # get last step speed
        return self.state["speed"]

    def get_density(self) -> np.ndarray:
        # get last step density
        return self.state["density"]

    def get_vehicle_num(self):
        # get last step vehicle number
        return self.state["density"] * self.segment_length * self.lane_num


class OnRampOrigin(BaseOrigin, Hp):

    _state = {"w"}
    _disturbance = {"demand"}
    _action = {"metering_rate"}

    def __init__(self, name: str, to_node_id: str):
        super().__init__(name, to_node_id)
        self.rate = 1.  # the metering rate
        self.demand = 0.  # the traffic demand

    def refresh(self, state: Dict = None, demand: float = None, rate: float = None):
        if state is not None:
            self.state["w"] = state["w"][0] + state["extra_veh"][0]
            self.clip_state(self.state)
        if demand is not None:
            self.demand = max(demand, 0.)
        if rate is not None:
            self.rate = max(rate, 0.)

    def step(self):
        q_out = self.get_flow()
        w_next = self.state["w"] + self.T * (self.demand - q_out)
        self.next_state["w"] = w_next
        self.clip_state(self.next_state)
        return self.next_state["w"]

    def get_flow(self) -> float:
        # get last step flow
        w = self.state["w"]
        # q1 = self.demand + w / self.T
        q1 = w / self.T
        rho_down = self.to_node_entity.get_downstream_density()  # the density of downstream link
        lane_num = self.to_node_entity.downstream_link[0].lane_num
        # q2 = self.ramp_vf * np.exp(-1 / self.am * (rho_down / self.ramp_rho_cri) ** self.am) * rho_down * lane_num
        q2 = lane_num * self.ramp_c
        q3 = lane_num * self.ramp_c * (self.ramp_rho_max - rho_down) / (self.ramp_rho_max - self.ramp_rho_cri)
        # there are two different formulations of ramp metering
        # q_out = self.rate * min(q1, q2, q3)
        q_out = min(q1, self.rate * q2, q3)
        return q_out

    def get_speed(self) -> float:
        # get last step speed
        # the downstream link may have more than one segment
        return self.to_node_entity.get_downstream_speed()[0]

    def get_vehicle_num(self):
        # get last step vehicle number
        return self.state["w"]


class MainlineOrigin(BaseOrigin, Hp):

    _state = {"w"}
    _disturbance = {"demand"}
    _action = {"speed_limit"}

    def __init__(self, name: str, to_node_id: str):
        super().__init__(name, to_node_id)
        self.demand = 0.  # the traffic demand
        self.speed_limit = np.inf  # the speed limit of mainline

    def refresh(self, state: Dict = None, demand: float = None, speed_limit: float = None):
        if state is not None:
            self.state["w"] = state["w"][0] + state["extra_veh"][0]
            self.clip_state(self.state)
        if demand is not None:
            self.demand = max(demand, 0.)
        if speed_limit is not None:
            self.speed_limit = max(self.speed_limit, 0.)

    def step(self):
        q_out = self.get_flow()
        w_next = self.state["w"] + self.T * (self.demand - q_out)
        self.next_state["w"] = w_next
        self.clip_state(self.next_state)
        return self.next_state["w"]

    def get_expect_speed(self, rho) -> float:
        return self.vf * np.exp((-1/self.am) * (rho / self.rho_cri) ** self.am)

    def get_flow(self) -> float:
        # get last step flow
        lane_num = self.to_node_entity.downstream_link[0].lane_num
        rho_down = self.to_node_entity.get_downstream_density()
        q_speed = self.vf * np.exp(-1/self.am * (rho_down/self.rho_cri)**self.am) * rho_down * lane_num
        # q = min(self.demand + self.state["w"] / self.T, q_limit)
        q = min(self.state["w"] / self.T, q_speed)
        return q

    def get_speed(self) -> float:
        # get last step speed
        # the downstream link may have more than one segment
        return self.to_node_entity.get_downstream_speed()[0]

    def get_vehicle_num(self):
        # get last step vehicle number
        return self.state["w"]
