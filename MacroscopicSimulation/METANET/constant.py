# This file contains the hyperparameters of METANET model

from dataclasses import dataclass


@dataclass
class HyperParam:
    # the step length of METANET model
    step_length = 5
    # the second of step length
    T = step_length / 3600
    # model parameter associated with driver desired speed
    am = 1.5190
    # model parameter associated with the convection term and anticipation term of driver desired speed
    tau = 6.4292 / 3600
    # model parameter associated with the anticipation term and merge term of driver desired speed
    kappa = 18.1316
    # model parameter associated with the anticipation term of driver desired speed
    # eta influences the speed of the back-propagation of both head and tail of the congestion
    # here we obey the slow start rule, which means that vehicles existing the high density areas
    # has the tendency to accelerate slower than for other densities
    # in the tail of the congestion (rho_i <= rho_i+1), use eta_high
    eta_low = 30.0736
    # in the head of the congestion (rho_i > rho_i+1), use eta_low
    eta_high = 63.1173
    # if the edge type is ramp, use eta_ramp
    eta_ramp = 1.1574
    # model parameter associated with the merge term of driver desired speed
    delta = 2.8317
    # model parameter associated with the lane drop term of driver desired speed
    phi = 1.4829
    # the free flow speed of mainline, in km/h
    vf = 115.9255
    # the free flow speed of on-ramp, in km/h
    ramp_vf = 54.7497
    # the critical density of mainline, in veh/km/lane
    rho_cri = 33.4909
    # the critical density of on-ramp, in veh/km/lane
    ramp_rho_cri = 33.6722
    # the max density of on-ramp, in veh/km/lane
    ramp_rho_max = 149.9179
    # the capacity of on-ramp, in veh/h/lane
    ramp_c = 954.43
