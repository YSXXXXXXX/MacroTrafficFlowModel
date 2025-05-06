# This file contains the hyperparameters of CTM model

from dataclasses import dataclass


@dataclass
class HyperParam:
    # the step length of CTM model
    step_length = 5
    # the second of step length
    T = step_length / 3600
    # the reduction factor of downstream capacity
    alpha = 0.1164
    # the speed wave of mainline congestion, in km/h
    wave_speed = 13.4878
    # the speed wave of mainline congestion when the density is larger than rho_cr_dch
    # and smaller than the density of upstream, in km/h
    wave_speed2 = 15.4436
    # the speed wave of on-ramp congestion
    ramp_wave_speed = 18.9342
    # the free flow speed of mainline, in km/h
    vf = 90.0376
    # the free flow speed of on-ramp, in km/h
    ramp_vf = 46.2052
    # the max density of mainline, in veh/km
    rho_max = 179.0175
    # the max density of on-ramp, in veh/km
    ramp_rho_max = 51.9402
    # the critical density of mainline, in veh/km/lane
    rho_cri = 23.3233
    # the capacity of mainline, in veh/h/lane
    c = 2099.97
    # the capacity of on-ramp, in veh/h/lane
    ramp_c = 697.59
