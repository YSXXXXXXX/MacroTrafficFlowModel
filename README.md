# Macroscopic Traffic Flow Model: METANET and Cell Transmission Model (CTM)

This repository focuses on macroscopic traffic flow simulation for freeway networks, implementing two classical models: Cell Transmission Model (CTM) and METANET. Both models are specifically designed for freeway-level analysis and provide a foundation for deploying traffic flow control strategies such as: Ramp Metering and Variable Speed Limit.

The implementation supports various link types, including:

- Mainline segments
- On-ramps
- Off-ramps

Each link is connected via a node model, capable of handling up to *two* upstream and *two* downstream links per node.
The implementation of CTM model is based on the following paper:
>
    [1] Daganzo, C. F. (1994). The cell transmission model: A dynamic representation of highway traffic consistent with the hydrodynamic theory. Transportation research part B: methodological, 28(4), 269-287.

    [2] Han, Y., Yuan, Y., Hegyi, A., & Hoogendoorn, S. P. (2016). New extended discrete first-order model to reproduce propagation of jam waves. Transportation Research Record, 2560(1), 108-118.

The implementation of METANET model is based on the following paper:

>
    [1] Messmer, A., & Papageorgiou, M. (1990). METANET: A macroscopic simulation program for motorway networks. Traffic engineering & control, 31(9).

    [2] Hegyi, A. (2004). Model predictive control for integrating traffic control measures. Netherlands TRAIL Research School.

## File Structure

``` ruby
pyTrafficFlowModels/
│
├── Net/
│   ├── buildNetwork.py
│   ├── demand.CSV      # the traffic demand of each origin
│   ├── links.CSV       # the topology of the network
│   └── segments.CSV    # the attribute of each segment
│
├── MacroscopicSimulation/CTM/
│   ├── link.py         # the link model of CTM
│   ├── node.py         # the node model of CTM
│   ├── engine.py       # the simulation engine of CTM
│   └── constant.py     # the constant of CTM 
│
├── MacroscopicSimulation/METANET/
│   ├── link.py         # the link model of METANET
│   ├── node.py         # the node model of METANET
│   ├── engine.py       # the simulation engine of METANET
│   └── constant.py     # the constant of METANET
│
└── example.py
```

## Network Description

Use two `*.CSV` files to describe the network topology and the attribute of each link. The `links.CSV` file contains the following columns:

- `id`: the id of the link
- `from`: the from node id of the link
- `to`: the to node id of the link
- `numLanes`: the number of lanes of the link

The `segments.CSV` file contains the following columns:

- `link`: the link name that the segment belongs to
- `length`: the length of the link in km
- `link_type`: the type of the link, including: `normal`, `onramp`, `offramp`
- `diverage_rate`: the diverage ratio of the segment
- `merge_rate`: the merge ratio of the segment
- `speed_limit_type`: the speed limit type of the segment, including: `normal`, `onramp`, `offramp`

## Acknowledgements

This project is inspired by and partially based on the excellent repository [FilippoAiraldi/sym-metanet](https://github.com/FilippoAiraldi/sym-metanet), which provides Python package to model traffic networks with the METANET framework.
