from typing import Dict

import numpy as np
import networkx as nx

from MacroscopicSimulation.baseClass import BaseLink, BaseOrigin
from MacroscopicSimulation.CTM.constant import HyperParam as Hp


class Link(BaseLink, Hp):
    _states = {"density"}

    def __init__(self, name: str, net: nx.DiGraph, merge_priority: float, diverge_priority: float,
                 from_node_id: str, to_node_id: str):
        lane_num = net.edges[(from_node_id, to_node_id)]["lane_num"]
        segment_num = net.edges[(from_node_id, to_node_id)]["segment_num"]
        segment_length = net.edges[(from_node_id, to_node_id)]["length"]
        link_type = net.edges[(from_node_id, to_node_id)]["type"]
        speed_limit_type = net.edges[(from_node_id, to_node_id)]["speed_limit_type"]
        super().__init__(name, link_type, lane_num, segment_num, segment_length, from_node_id, to_node_id)
        self.from_node_entity, self.to_node_entity = None, None
        self.merge_priority = merge_priority  # the merging rate
        self.diverge_priority = diverge_priority  # the diverging rate
        self.speed_limit_type = speed_limit_type
        self.supply = np.zeros(self.segment_num, dtype=np.float32)  # the flow that can send to downstream
        self.demand = np.zeros(self.segment_num, dtype=np.float32)  # the flow that can receive from upstream

    def refresh(self, state: Dict = None):
        if state is None:
            return
        for key in self._states:
            if self.state[key].shape == state[key].shape:
                self.state[key] = state[key]
        self.clip_state(self.state)

    def step(self):
        rho = self.state["density"]
        # the inflow of segment 1
        q_in_1 = self.from_node_entity.get_upstream_outflow(diverge_priority=self.diverge_priority)  # merge rate = 1
        q_in_2_to_n = np.minimum(self.demand[:-1], self.supply[1:])  # the inflow of segment 2 to N
        q_in = np.concatenate(([q_in_1], q_in_2_to_n), dtype=np.float32)  # the inflow of segment 1 to N
        # the outflow of segment N
        q_out_n = self.to_node_entity.get_upstream_outflow(merge_priority=self.merge_priority)  # diverge rate = 1
        q_out_1_to_n_1 = q_in_2_to_n  # outflow of segment 1 to N-1
        q_out = np.concatenate((q_out_1_to_n_1, [q_out_n]), dtype=np.float32)  # the outflow of segment 1 to N
        rho_next = rho + self.T * (q_in - q_out) / (self.lane_num * self.segment_length)
        self.next_state["density"] = rho_next
        self.clip_state(self.next_state)

    def calculate_supply_demand(self):
        rho = self.state["density"]
        if "onramp" in self.link_type:
            # assume that on-ramps do not have the capacity drop
            if "onramp" in self.speed_limit_type:
                self.supply = self.ramp_wave_speed * (self.ramp_rho_max - rho)
                self.demand = rho * self.ramp_vf
            else:
                self.supply = self.ramp_wave_speed * (self.ramp_rho_max - rho)
                self.demand = rho * self.vf
            self.supply = np.clip(self.supply, 0., self.ramp_c) * self.lane_num
            self.demand = np.clip(self.demand, 0., self.ramp_c) * self.lane_num
            return
        rho_up = np.array([self.from_node_entity.get_upstream_density()], dtype=np.float32)
        rho_up = np.concatenate((rho_up, rho[:-1]))
        rho_cr_dch = self.get_rho_cr_dch()
        for i in range(self.segment_num):
            if rho[i] < rho_cr_dch[i]:
                self.supply[i] = rho_cr_dch[i] * self.vf  # reduced capacity
                self.demand[i] = rho[i] * self.vf
            elif rho[i] >= rho_cr_dch[i]:
                self.supply[i] = self.wave_speed * (self.rho_max - rho[i])
                self.demand[i] = rho_cr_dch[i] * self.vf
                if rho[i] < rho_up[i]:
                    # rho_cr_dch(i) <= rho(i) < rho(i-1)
                    self.supply[i] = self.supply[i] - (self.wave_speed - self.wave_speed2) * (rho_up[i] - rho[i])
        self.supply = np.clip(self.supply, 0., self.c) * self.lane_num  # veh/h
        self.demand = np.clip(self.demand, 0., self.c) * self.lane_num  # veh/h

    def get_capacity_reduce_factor(self):
        # do not call this method if the current link is on-ramp
        rho = self.state["density"]
        return np.clip(1.0 - self.alpha * (rho - self.rho_cri) / (self.rho_max - self.rho_cri), 0., 1.)

    def get_rho_cr_dch(self):
        # do not call this method if the current link is on-ramp
        upstream_capacity_reduce_factor = self.from_node_entity.get_upstream_capacity_reduce_factor()
        capacity_reduce_factor = np.array([upstream_capacity_reduce_factor], dtype=np.float32)
        capacity_reduce_factor = np.concatenate((capacity_reduce_factor, self.get_capacity_reduce_factor()[:-1]))
        reduced_capacity = self.c * capacity_reduce_factor
        rho_cr_dch = reduced_capacity / self.vf
        return rho_cr_dch

    def get_flow(self):
        # calculate the outflow of current cell
        q_out_n = self.to_node_entity.get_upstream_outflow(merge_priority=self.merge_priority)
        q_out_1_to_n_1 = np.minimum(self.demand[:-1], self.supply[1:])  # outflow of segment 1 to N-1
        q_out = np.concatenate((q_out_1_to_n_1, [q_out_n]), dtype=np.float32)  # the outflow of segment 1 to N
        return q_out

    def get_density(self):
        return self.state["density"]

    def get_supply(self):
        return self.supply

    def get_demand(self):
        return self.demand

    def get_vehicle_num(self):
        return self.state["density"] * self.segment_length * self.lane_num


class OnRampOrigin(BaseOrigin, Hp):

    _states = {"density", "extra_veh"}
    _disturbance = {"demand"}
    _action = {"metering_rate"}

    def __init__(self, name: str, net: nx.DiGraph, merge_priority: float, to_node_id: str, from_node_id: str):
        super().__init__(name, to_node_id)
        self.lane_num = net.edges[(from_node_id, to_node_id)]["lane_num"]
        self.segment_num = net.edges[(from_node_id, to_node_id)]["segment_num"]
        self.segment_length = net.edges[(from_node_id, to_node_id)]["length"]
        self.link_type = net.edges[(from_node_id, to_node_id)]["type"]
        self.merge_priority = merge_priority  # the merging rate
        self.rate = 1.  # the metering rate
        self.traffic_demand = 0.  # the traffic demand
        self.supply = np.zeros(1, dtype=np.float32)  # the flow that can receive from upstream
        self.demand = np.zeros(1, dtype=np.float32)  # the flow that can send to downstream

    def refresh(self, state: Dict = None, demand: float = None, rate: float = None):
        if state is not None:
            for key in self._states:
                self.state[key] = state[key]
            self.clip_state(self.state)
        if demand is not None:
            self.traffic_demand = max(demand, 0.)
        if rate is not None:
            self.rate = max(rate, 0.)

    def step(self):
        rho = self.state["density"]
        extra_veh = self.state["extra_veh"]
        # the inflow of segment 1
        q_in_1 = min(self.supply[0], extra_veh[0] / self.T)  # TODO: FIXME: W / T or something else??
        # the outflow of segment N
        q_out_n = self._get_flow_with_rate()
        if self.segment_num == 1:
            rho_next = rho + self.T * (q_in_1 - q_out_n) / (self.lane_num * self.segment_length)
        else:
            q_in_2_to_n = np.minimum(self.demand[:-1], self.supply[1:])  # the inflow of segment 2 to N
            q_in = np.concatenate(([q_in_1], q_in_2_to_n), dtype=np.float32)  # the inflow of segment 1 to N
            q_out_1_to_n_1 = q_in_2_to_n  # outflow of segment 1 to N-1
            q_out = np.concatenate((q_out_1_to_n_1, [q_out_n]), dtype=np.float32)  # the outflow of segment 1 to N
            rho_next = rho + self.T * (q_in - q_out) / (self.lane_num * self.segment_length)
        extra_veh_next = np.zeros_like(extra_veh, dtype=np.float32)
        extra_veh_next[0] = extra_veh[0] + self.T * (self.traffic_demand - q_in_1)
        self.next_state["density"] = rho_next
        self.next_state["extra_veh"] = extra_veh_next
        self.clip_state(self.next_state)

    def calculate_supply_demand(self):
        self.supply = self.ramp_wave_speed * (self.ramp_rho_max - self.state["density"])
        self.supply = np.clip(self.supply, 0., self.ramp_c) * self.lane_num
        self.demand = self.state["density"] * self.ramp_vf
        self.demand = np.clip(self.demand, 0., self.ramp_c) * self.lane_num

    def get_supply(self):
        return self.supply

    def get_demand(self):
        return self.demand

    def _get_flow_with_rate(self):
        # calculate the outflow of current cell
        q_out = self.to_node_entity.get_upstream_outflow(merge_priority=self.merge_priority)
        # q_out = self.rate * min(q_out, self.ramp_c * self.lane_num)
        q_out = min(q_out, self.rate * self.ramp_c * self.lane_num)
        return q_out

    def get_flow(self):
        q_out_n = self._get_flow_with_rate()
        q_out_1_to_n_1 = np.minimum(self.demand[:-1], self.supply[1:])  # outflow of segment 1 to N-1
        q_out = np.concatenate((q_out_1_to_n_1, [q_out_n]), dtype=np.float32)  # the outflow of segment 1 to N
        return q_out

    def get_density(self):
        return self.state["density"]

    def get_vehicle_num(self):
        return self.state["density"] * self.segment_length * self.lane_num
    
    def get_extra_vehicle_num(self):
        return self.state["extra_veh"]


class MainlineOrigin(BaseOrigin, Hp):

    _states = {"density", "extra_veh"}
    _disturbance = {"demand"}

    def __init__(self, name: str, net: nx.DiGraph, merge_priority: float, to_node_id: str, from_node_id: str):
        super().__init__(name, to_node_id)
        self.lane_num = net.edges[(from_node_id, to_node_id)]["lane_num"]
        self.segment_num = net.edges[(from_node_id, to_node_id)]["segment_num"]
        self.segment_length = net.edges[(from_node_id, to_node_id)]["length"]
        self.link_type = net.edges[(from_node_id, to_node_id)]["type"]
        self.merge_priority = merge_priority  # the merging rate
        self.traffic_demand = 0.  # the traffic demand
        self.supply = np.zeros(1, dtype=np.float32)  # the flow that can receive from upstream
        self.demand = np.zeros(1, dtype=np.float32)  # the flow that can send to downstream

    def refresh(self, state: Dict = None, demand: float = None, action=None):
        if state is not None:
            for key in self._states:
                self.state[key] = state[key]
            self.clip_state(self.state)
        if demand is not None:
            self.traffic_demand = max(demand, 0.)

    def step(self):
        # the origin of mainline does not have any upstream cell, cannot calculate the rho_cr_dch
        rho = self.state["density"]
        extra_veh = self.state["extra_veh"]
        # the inflow of segment 1
        q_in_1 = min(self.supply[0], extra_veh[0] / self.T)  # TODO: FIXME: W / T or something else??
        q_in_2_to_n = np.minimum(self.demand[:-1], self.supply[1:])  # the inflow of segment 2 to N
        q_in = np.concatenate(([q_in_1], q_in_2_to_n), dtype=np.float32)  # the inflow of segment 1 to N
        # the outflow of segment N
        q_out_n = self.to_node_entity.get_upstream_outflow(merge_priority=self.merge_priority)
        q_out_1_to_n_1 = q_in_2_to_n  # outflow of segment 1 to N-1
        q_out = np.concatenate((q_out_1_to_n_1, [q_out_n]), dtype=np.float32)  # the outflow of segment 1 to N
        rho_next = rho + self.T * (q_in - q_out) / (self.lane_num * self.segment_length)
        extra_veh_next = np.zeros_like(extra_veh, dtype=np.float32)
        extra_veh_next[0] = extra_veh[0] + self.T * (self.traffic_demand - q_in_1)
        self.next_state["density"] = rho_next
        self.next_state["extra_veh"] = extra_veh_next
        self.clip_state(self.next_state)

    def calculate_supply_demand(self):
        self.supply = self.wave_speed * (self.rho_max - self.state["density"])
        self.supply = np.clip(self.supply, 0., self.c) * self.lane_num
        self.demand = self.state["density"] * self.vf
        self.demand = np.clip(self.demand, 0., self.c) * self.lane_num

    def get_capacity_reduce_factor(self):
        # do not call this method if the current link is on-ramp
        rho = self.state["density"]
        return np.clip(1.0 - self.alpha * (rho - self.rho_cri) / (self.rho_max - self.rho_cri), 0., 1.)

    def get_flow(self):
        # calculate the outflow of current cell
        q_out_n = self.to_node_entity.get_upstream_outflow(merge_priority=self.merge_priority)
        q_out_1_to_n_1 = np.minimum(self.demand[:-1], self.supply[1:])  # outflow of segment 1 to N-1
        q_out = np.concatenate((q_out_1_to_n_1, [q_out_n]), dtype=np.float32)  # the outflow of segment 1 to N
        return q_out

    def get_density(self):
        return self.state["density"]

    def get_demand(self):
        return self.demand

    def get_supply(self):
        return self.supply

    def get_vehicle_num(self):
        return self.state["density"] * self.segment_length * self.lane_num

    def get_extra_vehicle_num(self):
        return self.state["extra_veh"]
