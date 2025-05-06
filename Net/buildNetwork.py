import pandas as pd
import networkx as nx


def prepare_macro_env(ctrl_step_len: int, demand_file: str, segment_file: str, link_file: str):
    demand_df = pd.read_csv(demand_file)
    segment_df = pd.read_csv(segment_file)
    link_df = pd.read_csv(link_file)
    sample_interval = demand_df.loc[1, "time"] - demand_df.loc[0, "time"]
    # -------------------- get demand --------------------
    demand = {}
    for group, sub_demand_df in demand_df.groupby("link_id"):
        demand[f"{group}/selfLoop"] = sub_demand_df["demand"].to_numpy().repeat(sample_interval // ctrl_step_len)
        demand[group] = sub_demand_df["demand"].to_numpy().repeat(sample_interval // ctrl_step_len)
    # -------------------- get road network --------------------
    net = nx.DiGraph()
    link_ids = set(link_df["id"])  # only process the edges that have detectors
    for link_id in link_ids:
        segment = segment_df[segment_df["link"] == link_id]
        link = link_df[link_df["id"] == link_id]
        link_type = segment["link_type"].to_list()[0]  # the type of current link
        turn_rate_type = segment["turn_rate_type"].to_list()[0]  # the turn rate of the current link
        diverge, merge = segment["diverge_rate"].to_list()[0], segment["merge_rate"].to_list()[0]
        speed_limit_type = segment["speed_limit_type"].to_list()[0]  # the free flow speed of current edge
        # do not have incoming edges or the incoming edges do not have detectors (virtual edge named by "V")
        from_node_id, to_node_id = link["from"].item(), link["to"].item()
        is_origin = len(link_df[link_df["to"] == from_node_id]) == 0
        # do not have outgoing edges
        is_destination = len(link_df[link_df["from"] == to_node_id]) == 0
        net.add_nodes_from((from_node_id, to_node_id))
        # add type label for the node
        net.nodes[from_node_id]["type"] = "origin" if is_origin else "normal"
        net.nodes[to_node_id]["type"] = "destination" if is_destination else "normal"

        # add edge for the network
        edge_attribute = {
            "link_id": link_id, "origin": "origin" in link_type, "type": link_type,
            "turn_rate_type": turn_rate_type, "speed_limit_type": speed_limit_type,
            "diverge_rate": diverge, "merge_rate": merge
        }
        net.add_edge(from_node_id, to_node_id, **edge_attribute)
        num_segment = len(segment)
        net.edges[(from_node_id, to_node_id)]["segment_num"] = num_segment
        net.edges[(from_node_id, to_node_id)]["lane_num"] = link["numLanes"].item()
        net.edges[(from_node_id, to_node_id)]["length"] = segment["length"].to_numpy()

        if is_origin and "origin" not in link_type:
            # use the self-loop link to represent the store-and-forward model (for METANET)
            new_edge_id = "{}/selfLoop".format(link_id)  # avoid self loop and normal edge have same edge id
            net.add_edge(from_node_id, from_node_id,
                         **{"link_id": new_edge_id, "origin": True, "type": "{}/origin".format(link_type)})

    return demand, net


if __name__ == "__main__":
    prepare_macro_env(5, "./demand.CSV", "./segments.CSV", "./links.CSV")
