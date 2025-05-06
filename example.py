from Net.buildNetwork import prepare_macro_env
from MacroscopicSimulation.CTM.engine import Engine as CTM
from MacroscopicSimulation.METANET.engine import Engine as METANET

ctrl_step_len = 5
demand, net = prepare_macro_env(ctrl_step_len, "./Net/demand.CSV", "./Net/segments.CSV", "./Net/links.CSV")
ctm_engine = CTM(net)
metanet_engine = METANET(net)

ctm_engine.reset()
# initialize the traffic state, if needed
# ctm_engine.refresh(state=state)
ctm_engine.refresh(demand=demand)
# ---------- single step for CTM model ----------
ctm_engine.step()
ctm_engine.reset()

metanet_engine.reset()
# initialize the traffic state, if needed
# metanet_engine.refresh(state=state)
metanet_engine.refresh(demand=demand)
# ---------- single step for METANET model ----------
metanet_engine.step()
metanet_engine.reset()
