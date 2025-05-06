from typing import Dict, Union, Tuple
from copy import deepcopy

import numpy as np
import networkx as nx

from MacroscopicSimulation.baseClass import BaseEngine
from MacroscopicSimulation.METANET.constant import HyperParam as Hp
from MacroscopicSimulation.METANET.node import Node
from MacroscopicSimulation.METANET.link import Link, OnRampOrigin, MainlineOrigin


class Engine(BaseEngine):

    name = "METANET"

    def __init__(self, net: nx.DiGraph):
        super().__init__()
        # {node_id: node entity, (from_node_id, to_node_id): link/origin entity}
        self.id2entity: Dict[Union[str, Tuple], Union[Node, Link, OnRampOrigin, MainlineOrigin]] = {}
        self.nodes: Dict[str, Node] = {}
        self.links: Dict[str, Link] = {}
        self.origins: Dict[str, Union[Link, OnRampOrigin, MainlineOrigin]] = {}
        self.demand: Dict[str, np.ndarray] = {}
        self.rate: Dict[str, np.ndarray] = {}
        self.speed_limit: Dict[str, np.ndarray] = {}
        self.demand_counter = 0  # the demand of current step is demand[demand_counter]
        self.rate_counter = 0  # the merging rate of the current step is rate[rate_counter]
        self.speed_counter = 0  # the speed rate of mainline origin of the current step is speed_limit[speed_counter]
        for node, attribute in net.nodes.data():
            self.nodes[node] = Node(node, net)
            self.id2entity[node] = self.nodes[node]
        for from_node_id, to_node_id, attribute in net.edges.data():
            if attribute["origin"]:
                self.demand[attribute["link_id"]] = np.array([], dtype=np.float32)
                if "onramp" in attribute["type"]:
                    self.rate[attribute["link_id"]] = np.array([], dtype=np.float32)
                    # build on-ramp origin
                    self.origins[attribute["link_id"]] = OnRampOrigin(attribute["link_id"], to_node_id)
                else:
                    self.speed_limit[attribute["link_id"]] = np.array([], dtype=np.float32)
                    # build mainline origin
                    self.origins[attribute["link_id"]] = MainlineOrigin(attribute["link_id"], to_node_id)
                self.id2entity[(from_node_id, to_node_id)] = self.origins[attribute["link_id"]]
            else:
                turn_rate = attribute["diverge_rate"]
                self.links[attribute["link_id"]] = Link(attribute["link_id"], net, turn_rate, from_node_id, to_node_id)
                self.segment_num += self.links[attribute["link_id"]].segment_num
                self.id2entity[(from_node_id, to_node_id)] = self.links[attribute["link_id"]]
        for entity in self.id2entity.values():
            entity.get_entity(self.id2entity)

        # the traffic state recoder, the link id is the name of edge
        self.density_recoder = {link_id: [] for link_id in self.links.keys()}
        self.speed_recoder = {link_id: [] for link_id in self.links.keys()}
        self.outflow_recoder = {link_id: [] for link_id in list(self.links.keys()) + list(self.origins.keys())}
        self.vehicle_num_recoder = {link_id: [] for link_id in list(self.links.keys()) + list(self.origins.keys())}

    @property
    def model_step(self):
        return Hp.step_length

    def get_tts(self):
        tts = 0.
        for veh_num in self.vehicle_num_recoder.values():
            tts += np.sum(veh_num)
        tts *= Hp.step_length  # in second
        return tts

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
        if speed_limit is not None:
            for key in self.speed_limit.keys():
                self.speed_limit[key] = deepcopy(speed_limit[key])
            # set counter to 0 because getting new speed limit of mainline origin
            self.speed_counter = 0

    def step(self):
        for origin in self.origins.values():
            if isinstance(origin, MainlineOrigin):
                origin.refresh(demand=self.get_demand(origin), speed_limit=self.get_speed_limit(origin))
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
        for origin in self.origins.values():
            origin.update()
        for link in self.links.values():
            link.update()
        for origin in self.origins.values():
            self.outflow_recoder[origin.name].append(np.array([origin.get_flow()], dtype=np.float32))
            self.vehicle_num_recoder[origin.name].append(np.array([origin.get_vehicle_num()], dtype=np.float32))
        for link in self.links.values():
            self.outflow_recoder[link.name].append(link.get_flow())
            self.density_recoder[link.name].append(link.get_density())
            self.speed_recoder[link.name].append(link.get_speed())
            self.vehicle_num_recoder[link.name].append(link.get_vehicle_num())

    def get_state(self):
        return {"flow": self.outflow_recoder, "density": self.density_recoder, "speed": self.speed_recoder,
                "vehicle_num": self.vehicle_num_recoder}

    def reset(self):
        self.density_recoder = {link_id: [] for link_id in self.links.keys()}  # link id is the name of edge
        self.speed_recoder = {link_id: [] for link_id in self.links.keys()}  # {link_id: [np.ndarray, np.ndarray, ...],}
        self.outflow_recoder = {link_id: [] for link_id in list(self.links.keys()) + list(self.origins.keys())}
        self.vehicle_num_recoder = {link_id: [] for link_id in list(self.links.keys()) + list(self.origins.keys())}

    def get_rate(self, entity):
        length = len(self.rate[entity.name])
        if length > 0:
            rate = self.rate[entity.name][self.rate_counter if self.rate_counter < length else -1]
        else:
            rate = None
        return rate

    def get_speed_limit(self, entity):
        length = len(self.speed_limit[entity.name])
        if length > 0:
            speed_limit = self.speed_limit[entity.name][self.speed_counter if self.speed_counter < length else -1]
        else:
            speed_limit = None
        return speed_limit

    def get_demand(self, entity):
        length = len(self.demand[entity.name])
        if length > 0:
            demand = self.demand[entity.name][self.demand_counter if self.demand_counter < length else -1]
        else:
            demand = None
        return demand

    @staticmethod
    def change_model_parameters(p):
        Hp.am, Hp.tau, Hp.kappa, Hp.eta_low, Hp.eta_high, Hp.eta_ramp, Hp.delta, Hp.phi, = p[0], p[1] / 3600, p[2], p[3], p[4], p[5], p[6], p[7]
        Hp.vf, Hp.rho_cri = p[8], p[9]
        Hp.ramp_vf, Hp.ramp_rho_cri, Hp.ramp_rho_max = p[10], p[11], p[12]
        Hp.ramp_c = Hp.ramp_vf * Hp.ramp_rho_cri * np.exp(-1/Hp.am)
