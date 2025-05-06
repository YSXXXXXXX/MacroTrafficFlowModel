import numpy as np
import networkx as nx

from MacroscopicSimulation.baseClass import BaseNode
from MacroscopicSimulation.CTM.constant import HyperParam as Hp


class Node(BaseNode, Hp):
    def __init__(self, node_id: str, net: nx.DiGraph):
        super().__init__(node_id)
        # the method of origin link is same to normal link
        # so CTM does not distinguish the origin and normal link
        # assume that the maximum number of upstream link is 2 (mainline and on-ramp)
        # assume that the maximum number of downstream link is 2 (mainline and off-ramp)
        # if more than 1 upstream link, 1 - self.ramp_flag is the index of mainline
        # if only 1 upstream link, 1 - self.ramp_flag is the index of upstream link
        self.ramp_flag: int = 1
        for i, predecessor in enumerate(net.predecessors(node_id)):
            if self.node_id == predecessor:
                # exclude the self-loop
                continue
            self.upstream_link.append(predecessor)
            if "onramp" in net.edges[(predecessor, node_id)]["type"]:
                self.ramp_flag = i
        for successor in net.successors(node_id):
            if self.node_id == successor:
                # exclude the self-loop
                continue
            self.downstream_link.append(successor)
        self.upstream_num = len(self.upstream_link)
        self.downstream_num = len(self.downstream_link)
        self.destination_flag = True if self.downstream_num == 0 else False  # whether this node is destination
        if self.upstream_num == 0:
            raise RuntimeError("A node must have a in link")
        if self.upstream_num > 2:
            raise RuntimeError("The number of upstream link of a node must <= 2")
        if self.downstream_num > 2:
            raise RuntimeError("The number of downstream link of a node must <= 2")
        if self.upstream_num > 1 and self.downstream_num > 1:
            raise RuntimeError("The number of upstream and downstream are both > 1, check the network")
        if self.destination_flag and self.upstream_num > 1:
            raise RuntimeError("The destination node can only connected to one in link")

    def get_upstream_outflow(self, merge_priority: float = 1.0, diverge_priority: float = 1.0) -> float:
        # FIXME: deal with multi normal upstream
        if self.upstream_num > 1:
            # the number upstream is 2 (on-ramp and mainline), the number of downstream is 1
            ramp_turn_rate = self.upstream_link[self.ramp_flag].merge_priority
            mainline_turn_rate = self.upstream_link[1 - self.ramp_flag].merge_priority
            if ramp_turn_rate + mainline_turn_rate != 1.0:
                raise RuntimeError("The sum of mainline turn rate and ramp turn rate should be 1")
            ramp_demand = self.upstream_link[self.ramp_flag].get_demand()[-1]
            mainline_demand = self.upstream_link[1 - self.ramp_flag].get_demand()[-1]
            supply = self.downstream_link[0].get_supply()[0]
            if ramp_demand + mainline_demand <= supply:
                ramp_q_out = ramp_demand
                mainline_q_out = mainline_demand
            else:
                ramp_q_out = np.median((ramp_demand, supply - mainline_demand, ramp_turn_rate * supply))
                mainline_q_out = np.median((mainline_demand, supply - ramp_demand, mainline_turn_rate * supply))
            if merge_priority == ramp_turn_rate:
                q_out = ramp_q_out
            elif merge_priority == mainline_turn_rate:
                q_out = mainline_q_out
            else:
                q_out = ramp_q_out + mainline_q_out
        elif self.downstream_num > 1:
            # the number of upstream is 1, the number of downstream is 2 (off-ramp and mainline)
            demand = self.upstream_link[0].get_demand()[-1]
            total_q_out = min([demand] +
                              [self.supply_over_diverge(link.get_supply()[0], link.diverge_priority)
                               for link in self.downstream_link])
            q_out = diverge_priority * total_q_out
        elif self.downstream_num == 1:
            # the number of upstream is 1, the number of downstream is 1
            # (the case upstream is 2, downstream is 2 does not exist)
            demand = self.upstream_link[0].get_demand()[-1]
            supply = self.downstream_link[0].get_supply()[0]
            q_out = min(demand, supply)
        else:
            # this node is destination, number of upstream is 1
            demand = self.upstream_link[0].get_demand()[-1]
            supply = self.upstream_link[0].get_supply()[-1]
            q_out = min(demand, supply)
        return q_out

    @staticmethod
    def supply_over_diverge(supply: float, diverge: float):
        if diverge > 0.:
            return supply / diverge
        else:
            return np.inf

    def get_upstream_density(self):
        # FIXME: deal with multi normal upstream
        return self.upstream_link[1 - self.ramp_flag].get_density()[-1]

    def get_upstream_capacity_reduce_factor(self):
        # FIXME: deal with multi normal upstream
        return self.upstream_link[1 - self.ramp_flag].get_capacity_reduce_factor()[-1]
