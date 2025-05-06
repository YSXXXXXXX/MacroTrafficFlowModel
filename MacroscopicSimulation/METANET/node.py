from typing import Tuple
from warnings import warn

import networkx as nx
import numpy as np

from MacroscopicSimulation.baseClass import BaseNode
from MacroscopicSimulation.METANET.constant import HyperParam as Hp


class Node(BaseNode, Hp):

    def __init__(self, node_id: str, net: nx.DiGraph):
        super().__init__(node_id)
        self.origin_flag = False  # whether origin is contained in the upstream link
        self.ramp_flag = None
        for i, predecessor in enumerate(net.predecessors(node_id)):
            if net.edges[(predecessor, node_id)]["origin"]:
                self.origin_flag = True
                self.origin_link.append(predecessor)
            else:
                self.upstream_link.append(predecessor)
                if net.edges[(predecessor, node_id)]["type"] in "onramp":
                    self.ramp_flag = i
        for successor in net.successors(node_id):
            if self.node_id != successor:
                # exclude the self-loop
                self.downstream_link.append(successor)
        self.upstream_num = len(self.upstream_link) + len(self.origin_link)
        self.downstream_num = len(self.downstream_link)
        self.destination_flag = True if self.downstream_num == 0 else False  # whether this node is destination
        if self.upstream_num == 0:
            raise RuntimeError("A node must have a in link")
        if self.origin_flag and self.upstream_num == 1 and self.downstream_num > 1:
            raise RuntimeError("The origin can only connected to one out link")
        if self.destination_flag and self.upstream_num > 1:
            raise RuntimeError("The destination node can only connected to one in link")

    def get_upstream_flow_and_speed(self, turn_rate: float = 1.0) -> Tuple[float, float]:
        if self.upstream_num == 1:
            # the current node has only one upstream link
            if self.origin_flag is True:
                # the only one upstream link is origin
                q = turn_rate * self.origin_link[0].get_flow()
                v = self.origin_link[0].get_speed()
            else:
                q = turn_rate * self.upstream_link[0].get_flow()[-1]
                v = self.upstream_link[0].get_speed()[-1]
        else:
            # the current node has more than one upstream link
            if self.origin_flag is True:
                q_o = self.origin_link[0].get_flow()
            else:
                q_o = 0.
            q_up = np.zeros(self.upstream_num, dtype=np.float32)
            v_up = np.zeros(self.upstream_num, dtype=np.float32)
            for i, upstream_link in enumerate(self.upstream_link):
                q_up[i] = upstream_link.get_flow()[-1]
                v_up[i] = upstream_link.get_speed()[-1]
            q = turn_rate * (q_o + sum(q_up))
            if sum(q_up) <= 0.:
                warn("upstream flow is zero, node {}".format(self.node_id))
                v = 0.
            else:
                v = sum(v_up * q_up) / sum(q_up)
        return q, v

    def get_downstream_density(self) -> float:
        if self.destination_flag:
            # current node is destination
            # FIXME: here only consider the situation that destination is not congested
            rho_up = self.upstream_link[0].get_density()[-1]
            return min(rho_up, self.rho_cri)
        else:
            if self.downstream_num == 1:
                # the current node has only one downstream link
                return self.downstream_link[0].get_density()[0]
            else:
                # the current node has more than one downstream link
                rho_down = np.zeros(self.downstream_num, dtype=np.float32)
                for i, downstream_link in enumerate(self.downstream_link):
                    rho_down[i] = downstream_link.get_density()[0]
                if sum(rho_down) <= 0.:
                    warn("downstream rho is zero, node {}".format(self.node_id))
                    return 0.
                else:
                    return np.dot(rho_down, rho_down) / sum(rho_down)

    def get_downstream_speed(self):
        # only called by origin
        return self.downstream_link[0].get_speed()
