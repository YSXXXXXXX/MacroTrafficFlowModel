from typing import Dict, Union, Tuple
from copy import deepcopy

import numpy as np
import networkx as nx

from MacroscopicSimulation.baseClass import BaseEngine
from MacroscopicSimulation.CTM.constant import HyperParam as Hp
from MacroscopicSimulation.CTM.link import Link, OnRampOrigin, MainlineOrigin
from MacroscopicSimulation.CTM.node import Node


class Engine(BaseEngine):

    name = "CTM"

    def __init__(self, net: nx.DiGraph):
        super().__init__()
        # {node_id: node entity, (from_node_id, to_node_id): link/origin entity}
        self.id2entity: Dict[Union[str, Tuple], Union[Node, Link, OnRampOrigin, MainlineOrigin]] = {}
        self.nodes: Dict[str, Node] = {}
        self.links: Dict[str, Link] = {}
        self.origins: Dict[str, Union[OnRampOrigin, MainlineOrigin]] = {}
        self.demand: Dict[str, np.ndarray] = {}
        self.rate: Dict[str, np.ndarray] = {}
        self.demand_counter = 0  # the demand of current step is demand[demand_counter]
        self.rate_counter = 0  # the merging rate of the current step is rate[rate_counter]
        # ---------- build nodes and links ----------
        for node, attribute in net.nodes.data():
            if "origin" in attribute["type"]:
                # do not build a Node instance for "origin" or "origin/normal" node
                continue
            self.nodes[node] = Node(node, net)
            self.id2entity[node] = self.nodes[node]
        for from_node_id, to_node_id, attribute in net.edges.data():
            if from_node_id == to_node_id:
                # do not build the self-loop link, because there does not exist a store-forward model in the origin link
                continue
            merge_priority, diverge_priority = attribute["merge_rate"], attribute["diverge_rate"]
            if "origin" in net.nodes[from_node_id]["type"]:
                self.demand[attribute["link_id"]] = np.array([], dtype=np.float32)
                if "onramp" in attribute["type"]:
                    self.rate[attribute["link_id"]] = np.array([], dtype=np.float32)
                    # build on-ramp origin
                    self.origins[attribute["link_id"]] = OnRampOrigin(attribute["link_id"], net, merge_priority, to_node_id, from_node_id)
                else:
                    # build mainline origin
                    self.origins[attribute["link_id"]] = MainlineOrigin(attribute["link_id"], net, merge_priority, to_node_id, from_node_id)
                self.id2entity[(from_node_id, to_node_id)] = self.origins[attribute["link_id"]]
                self.segment_num += self.origins[attribute["link_id"]].segment_num
            else:
                self.links[attribute["link_id"]] = Link(attribute["link_id"], net, merge_priority, diverge_priority, from_node_id, to_node_id)
                self.id2entity[(from_node_id, to_node_id)] = self.links[attribute["link_id"]]
                self.segment_num += self.links[attribute["link_id"]].segment_num
        for entity in self.id2entity.values():
            entity.get_entity(self.id2entity)

        # the traffic state recoder, the link id is the name of edge
        self.density_recoder = {link_id: [] for link_id in list(self.links.keys()) + list(self.origins.keys())}
        self.speed_recoder = {link_id: [] for link_id in list(self.links.keys()) + list(self.origins.keys())}
        self.outflow_recoder = {link_id: [] for link_id in list(self.links.keys()) + list(self.origins.keys())}
        self.vehicle_num_recoder = {link_id: [] for link_id in list(self.links.keys()) + list(self.origins.keys())}
        self.extra_veh_recoder = {link_id: [] for link_id in self.origins.keys()}

    @property
    def model_step(self):
        return Hp.step_length

    def refresh(self, state: Dict = None, demand: Dict = None, merge_rate: Dict = None, speed_limit: Dict = None):
        if state is not None:
            for link in self.links.values():
                link.refresh(state[link.name])
            for origin in self.origins.values():
                origin.refresh(state[origin.name])
        if demand is not None:
            for key in self.demand.keys():
                self.demand[key] = deepcopy(demand[key])
            # set counter to 0 because getting new demand
            self.demand_counter = 0
        if merge_rate is not None:
            for key in self.rate.keys():
                self.rate[key] = deepcopy(merge_rate[key])
            # set counter to 0 because getting new merging rate
            self.rate_counter = 0

    def step(self):
        # calculate supply and demand for each link and origin
        for link in self.links.values():
            link.calculate_supply_demand()
        for origin in self.origins.values():
            origin.calculate_supply_demand()
        # calculate traffic state for each link and origin
        for origin in self.origins.values():
            if isinstance(origin, MainlineOrigin):
                origin.refresh(demand=self.get_demand(origin))
            elif isinstance(origin, OnRampOrigin):
                origin.refresh(demand=self.get_demand(origin), rate=self.get_rate(origin))
            else:
                raise TypeError("unknown origin type")
            origin.step()
        # update the counter
        self.rate_counter += 1 if self.rate_counter is not None else None
        self.demand_counter += 1 if self.demand_counter is not None else None
        for link in self.links.values():
            link.step()
        # update traffic state for each link and origin
        for origin in self.origins.values():
            origin.update()
        for link in self.links.values():
            link.update()
        for origin in self.origins.values():
            outflow, density = origin.get_flow(), origin.get_density()
            if (density > 0.).all():
                speed = outflow / (density * origin.lane_num)
            else:
                speed = np.zeros_like(outflow, dtype=np.float32)
            self.outflow_recoder[origin.name].append(outflow)
            self.density_recoder[origin.name].append(density)
            self.speed_recoder[origin.name].append(speed)
            self.vehicle_num_recoder[origin.name].append(origin.get_vehicle_num())
            self.extra_veh_recoder[origin.name].append(origin.get_extra_vehicle_num())
        for link in self.links.values():
            outflow, density = link.get_flow(), link.get_density()
            if (density > 0.).all():
                speed = outflow / (density * link.lane_num)
            else:
                speed = np.zeros_like(outflow, dtype=np.float32)
            self.outflow_recoder[link.name].append(outflow)
            self.density_recoder[link.name].append(density)
            self.speed_recoder[link.name].append(speed)
            self.vehicle_num_recoder[link.name].append(link.get_vehicle_num())

    def get_state(self):
        return {"flow": self.outflow_recoder, "density": self.density_recoder, "speed": self.speed_recoder,
                "vehicle_num": self.vehicle_num_recoder}

    def reset(self):
        # link id is the name of edge, {link_id: [np.ndarray, np.ndarray, ...],}
        self.density_recoder = {link_id: [] for link_id in list(self.links.keys()) + list(self.origins.keys())}
        self.speed_recoder = {link_id: [] for link_id in list(self.links.keys()) + list(self.origins.keys())}
        self.outflow_recoder = {link_id: [] for link_id in list(self.links.keys()) + list(self.origins.keys())}
        self.vehicle_num_recoder = {link_id: [] for link_id in list(self.links.keys()) + list(self.origins.keys())}
        self.extra_veh_recoder = {link_id: [] for link_id in self.origins.keys()}

    def get_rate(self, entity):
        length = len(self.rate[entity.name])
        if length > 0:
            rate = self.rate[entity.name][self.rate_counter if self.rate_counter < length else -1]
        else:
            rate = None
        return rate

    def get_demand(self, entity):
        # get the traffic demand
        length = len(self.demand[entity.name])
        if length > 0:
            demand = self.demand[entity.name][self.demand_counter if self.demand_counter < length else -1]
        else:
            demand = None
        return demand

    def get_tts(self):
        tts = 0.
        for veh_num_list in self.vehicle_num_recoder.values():
            tts += np.sum(veh_num_list)
        for veh_num_list in self.extra_veh_recoder.values():
            tts += np.sum(veh_num_list)
        tts *= Hp.step_length  # in second
        return tts

    @staticmethod
    def get_priority(turn_rate_type: str, merge_rate: float, diverge_rate: float):
        if "diverge" in turn_rate_type:
            diverge_priority = 1. - diverge_rate
        elif "offramp" in turn_rate_type:
            diverge_priority = diverge_rate
        else:
            diverge_priority = 1.
        if "merge" in turn_rate_type:
            merge_priority = 1. - merge_rate
        elif "onramp" in turn_rate_type:
            merge_priority = merge_rate
        else:
            merge_priority = 1.
        return merge_priority, diverge_priority

    @staticmethod
    def change_model_parameters(p):
        Hp.alpha, Hp.wave_speed, Hp.wave_speed2, Hp.ramp_wave_speed = p[0], p[1], p[2], p[3]
        Hp.vf, Hp.rho_max, Hp.ramp_vf, Hp.ramp_rho_max = p[4], p[5], p[6], p[7]
        # inferable parameters
        Hp.rho_cri = Hp.wave_speed * Hp.rho_max / (Hp.wave_speed + Hp.vf)
        Hp.c = Hp.vf * Hp.rho_cri
        Hp.ramp_rho_cri = Hp.ramp_wave_speed * Hp.ramp_rho_max / (Hp.ramp_wave_speed + Hp.ramp_vf)
        Hp.ramp_c = Hp.ramp_vf * Hp.ramp_rho_cri
