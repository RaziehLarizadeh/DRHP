import os
from pyomo.environ import *
import pandas as pd
import math

fine_tune = [True, False][0]
spliter_point = 180
taus_list = [2] # Rolling duration, ex: [2,4,6,8] or ...

df_output = pd.DataFrame(columns=['filename','RealCost','Backlog',"Inventoryraw","Inventoryfinal","K"])
writer_cost = pd.ExcelWriter(f"outputs/costs_df.xlsx", engine='xlsxwriter')

for scenario_d in [1]: # Demand scenarios, ex: [1,2,3] or ...
    for scenario_r in [1]: # Recycling rate scenarios, ex: [1,2,3] or ...
        df_cost = pd.DataFrame(index=['exact', '180ave','LSTMv1', 'Horizons'],
                               columns=taus_list)
        for tau_sce in taus_list:
            for pred_sce in ['exact', '180ave','LSTMv1']:

                if tau_sce == taus_list[0] and pred_sce == 'exact' : pass
                else:    del DRHP

                address_case = fr'Data\Case1-DemandS{scenario_d},RRS{scenario_r}\Python-RHP-DATA-Case1-DemandS{scenario_d},RRS{scenario_r},tau{tau_sce},{pred_sce}.xlsx'
                if f"Case1-DemandS{scenario_d},RRS{scenario_r}" not in os.listdir(f"Data"): os.mkdir(f"Data/Case1-DemandS{scenario_d},RRS{scenario_r}")

                address = fr'Data\parameters.xlsx'

                D_data = pd.read_excel(address,header= 0 , sheet_name = 'D')[f'Scenario_{scenario_d}']
                D = pd.DataFrame (D_data.values)
                D.columns = range(len(D.columns))

                RR_data = pd.read_excel(address, header= 0 ,sheet_name = 'RR')[f'Scenario_{scenario_r}']
                RR_data = pd.DataFrame(RR_data, index=range(0, len(RR_data), tau_sce))
                RR = pd.DataFrame (RR_data)
                RR.index = range(len(RR))
                RR.columns = range(len(RR.columns))

                Days_data = pd.read_excel(address, header= None ,sheet_name = 'Days')
                Days = pd.DataFrame(Days_data)


                if pred_sce == "exact":
                    Dprim = pd.DataFrame(D)
                    RRprim = pd.DataFrame(RR)
                elif  pred_sce == "180ave":
                    Dprim_data = pd.DataFrame(pd.read_excel(f"Synthetic_demand.xlsx"), columns=[f'Scenario {scenario_d}'])
                    Dprim= pd.DataFrame([Dprim_data.iloc[:spliter_point].mean().values[0]]*(Days.iloc[0,0])) #
                    RRprim_data = pd.DataFrame(pd.read_excel(f"Synthetic_recycling.xlsx"), columns=[f'Scenario {scenario_r}'])
                    RRprim= pd.DataFrame([RRprim_data.iloc[:spliter_point].mean().values[0]]*len(RR))
                elif  pred_sce == "LSTMv1":
                    Dprim_data = pd.read_excel(f"outputs/demand/{scenario_d}/{tau_sce}/{'Yes' if fine_tune else 'No'}_{tau_sce}_demand_S{scenario_d}.xlsx")
                    Dprim = pd.DataFrame (Dprim_data,columns=['demand'])
                    Dprim.columns = range(len(Dprim.columns))
                    RRprim_data = pd.read_excel(f"outputs/recycling/{scenario_r}/{tau_sce}/{'Yes' if fine_tune else 'No'}_{tau_sce}_recycling_S{scenario_r}.xlsx")
                    RRprim = pd.DataFrame (RRprim_data,columns=[f'recycling Rate'])
                    RRprim.index = range(len(RRprim))
                    RRprim = pd.DataFrame(RRprim,index = range(0,len(RRprim),int(tau_sce))).iloc[:len(RR)]
                    RRprim.index = range(len(RRprim))

                Horizons = pd.DataFrame([math.floor(Days.iloc[0,0]/tau_sce) - 1])

                Pcap_data = pd.read_excel(address,header= None , sheet_name = 'Pcap')
                Pcap = pd.DataFrame (Pcap_data)

                L_data = pd.read_excel(address, header= None ,sheet_name = 'L')
                L = pd.DataFrame (L_data)

                X_data = pd.read_excel(address, header= None ,sheet_name = 'X')
                X = pd.DataFrame (X_data)

                Invr_data = pd.read_excel(address, header= None ,sheet_name = 'Invr')
                Invr = pd.DataFrame (Invr_data)

                Invf_data = pd.read_excel(address, header= None ,sheet_name = 'Invf')
                Invf = pd.DataFrame (Invf_data)

                Cinvr_data = pd.read_excel(address, header= None ,sheet_name = 'Cinvr')
                Cinvr = pd.DataFrame (Cinvr_data)

                Cinvf_data = pd.read_excel(address, header= None ,sheet_name = 'Cinvf')
                Cinvf = pd.DataFrame (Cinvf_data)

                Cdo_data = pd.read_excel(address, header= None ,sheet_name = 'Cdo')
                Cdo = pd.DataFrame (Cdo_data)

                Cpro_data = pd.read_excel(address, header= None ,sheet_name = 'Cpro')
                Cpro = pd.DataFrame (Cpro_data)

                Cbl_data = pd.read_excel(address, header= None ,sheet_name = 'Cbl')
                Cbl = pd.DataFrame (Cbl_data)

                Cvp_data = pd.read_excel(address, header= None ,sheet_name = 'Cvp')
                Cvp = pd.DataFrame (Cvp_data)

                Cwp_data = pd.read_excel(address, header= None ,sheet_name = 'Cwp')
                Cwp = pd.DataFrame (Cwp_data)

                Cr_data = pd.read_excel(address, header= None ,sheet_name = 'Cr')
                Cr = pd.DataFrame (Cr_data)

                Ccv_data = pd.read_excel(address, header= None ,sheet_name = 'Ccv')
                Ccv = pd.DataFrame (Ccv_data)

                Ccw_data = pd.read_excel(address, header= None ,sheet_name = 'Ccw')
                Ccw = pd.DataFrame (Ccw_data)

                Alpha_data = pd.read_excel(address, header= None ,sheet_name = 'Alpha')
                Alpha = pd.DataFrame (Alpha_data)

                M_data = pd.read_excel(address, header= None ,sheet_name = 'M')
                M = pd.DataFrame (M_data)

                Epsilon_data = pd.read_excel(address, header= None ,sheet_name = 'Epsilon')
                Epsilon = pd.DataFrame (Epsilon_data)

                RealCost_data = pd.read_excel(address, header= None ,sheet_name = 'RealCost')
                RealCost = pd.DataFrame (RealCost_data)

                Backlog_data = pd.read_excel(address, header= None ,sheet_name = 'Backlog')
                Backlog = pd.DataFrame (Backlog_data)

                Inventoryraw_data = pd.read_excel(address, header= None ,sheet_name = 'Inventoryraw')
                Inventoryraw = pd.DataFrame (Inventoryraw_data)

                Inventoryfinal_data = pd.read_excel(address, header= None ,sheet_name = 'Inventoryfinal')
                Inventoryfinal = pd.DataFrame (Inventoryfinal_data)

                K_data = pd.read_excel(address, header= None ,sheet_name = 'K')
                K = pd.DataFrame (K_data)

                T = RangeSet(Days.loc[0][0])
                H = RangeSet(Horizons.loc[0][0])

                with pd.ExcelWriter(address_case, mode="w", engine="openpyxl") as writer:
                    pd.DataFrame([None]).to_excel(writer, sheet_name='initial', index=False, header=False)


                for h in H:

                    DRHP = ConcreteModel()

                    DRHP.k = Var (H, within =  NonNegativeReals)
                    DRHP.xprim = Var (H, within = NonNegativeReals)
                    DRHP.yprim = Var (H, within = NonNegativeReals)
                    DRHP.lprim = Var (H, within =  NonNegativeReals)
                    DRHP.vprim = Var (H, within = Binary)
                    DRHP.wprim = Var (H, within = Binary)
                    DRHP.invr = Var (T, within =  NonNegativeReals)
                    DRHP.invf = Var (T, within = NonNegativeIntegers)
                    DRHP.pq = Var (T, within = NonNegativeIntegers)
                    DRHP.sq = Var (T, within = NonNegativeIntegers)
                    DRHP.bl = Var (T, within=NonNegativeIntegers)

                    DRHP.obj = Objective (expr = (Cdo.loc[h-1] *DRHP.k[h] + Cvp.loc[h-1]*DRHP.xprim[h]
                                                    + Cwp.loc[h-1]*DRHP.yprim[h] + Cr.loc[h-1]*DRHP.lprim[h]
                                                    + Ccv.loc[h-1]*DRHP.vprim[h] + Ccw.loc[h-1]*DRHP.wprim[h]
                                                    + sum (Cinvr.loc[t-1]*DRHP.invr[t] +
                                                           Cinvf.loc[t-1]*DRHP.invf[t] +
                                                           Cpro.loc[t-1]*DRHP.pq[t]
                                                           for t in range (1+(h - 1)*tau_sce,(h - 1)*tau_sce+2*tau_sce+1))
                                                    + sum (Cbl.loc[t-1]*(D.loc[t-1]-DRHP.sq[t])
                                                           for t in range (1+(h - 1)*tau_sce,(h - 1)*tau_sce+tau_sce+1))
                                                    + sum (Cbl.loc[t-1]*(Dprim.loc[t-1]-DRHP.sq[t])
                                                           for t in range((h - 1)*tau_sce+tau_sce+1,(h - 1)*tau_sce+2*tau_sce+1))).loc[0]
                                                    , sense=minimize)
                    DRHP.c1 = ConstraintList()
                    for t in range(1+(h - 1)*tau_sce , (h - 1)*tau_sce + 2*tau_sce +1 ):
                        DRHP.c1.add(DRHP.pq[t]<=Pcap.loc[t-1][0])

                    DRHP.c2 = ConstraintList()
                    for t in range(1+(h - 1)*tau_sce , (h - 1)*tau_sce + tau_sce +1 ):
                        DRHP.c2.add(DRHP.sq[t]<=D.loc[t-1][0])

                    DRHP.c3 = ConstraintList()
                    for t in range(1+tau_sce+(h - 1)*tau_sce , (h - 1)*tau_sce + 2*tau_sce +1 ):
                        DRHP.c3.add(DRHP.sq[t]<=Dprim.loc[t-1][0])

                    DRHP.c4 = ConstraintList()
                    for t in [(h-1)*tau_sce+1]:
                        DRHP.c4.add(Invr.loc[h-1][0]+X.loc[h-1][0]+L.loc[h-1][0]+DRHP.k[h]-Alpha.loc[0][0]*DRHP.pq[t] == DRHP.invr[t])

                    DRHP.c5 = ConstraintList()
                    for t in range((h-1)*tau_sce+2,(h-1)*tau_sce+tau_sce+1):
                        DRHP.c5.add(DRHP.invr[t-1]-Alpha.loc[0][0]*DRHP.pq[t] == DRHP.invr[t])

                    DRHP.c6 = ConstraintList()
                    for t in [(h-1)*tau_sce+tau_sce+1]:
                        DRHP.c6.add(DRHP.invr[t-1]-Alpha.loc[0][0]*DRHP.pq[t]+DRHP.xprim[h]+DRHP.lprim[h] == DRHP.invr[t])

                    DRHP.c7 = ConstraintList()
                    for t in range((h-1)*tau_sce+tau_sce+2,(h-1)*tau_sce+2*tau_sce+1):
                        DRHP.c7.add(DRHP.invr[t-1]-Alpha.loc[0][0]*DRHP.pq[t] == DRHP.invr[t])

                    DRHP.c8 = ConstraintList()
                    for t in [(h-1)*tau_sce+1]:
                        DRHP.c8.add(Invf.loc[h-1][0]+DRHP.pq[t]-DRHP.sq[t] == DRHP.invf[t])

                    DRHP.c9 = ConstraintList()
                    for t in range((h-1)*tau_sce+2,(h-1)*tau_sce+2*tau_sce+1):
                        DRHP.c9.add(DRHP.invf[t-1]+DRHP.pq[t]-DRHP.sq[t] == DRHP.invf[t])

                    DRHP.c10 = ConstraintList()
                    for t in range((h-1)*tau_sce+1,(h-1)*tau_sce+tau_sce+1):
                        DRHP.c10.add(DRHP.bl[t] == D.loc[t-1][0]-DRHP.sq[t])

                    DRHP.c11 = Constraint(expr=DRHP.xprim[h] <= M.loc[0][0]*DRHP.vprim[h])

                    DRHP.c12 = Constraint(expr=DRHP.xprim[h] >= Epsilon.loc[0][0]*DRHP.vprim[h])

                    DRHP.c13 = Constraint(expr=DRHP.yprim[h] <= M.loc[0][0]*DRHP.wprim[h])

                    DRHP.c14 = Constraint(expr=DRHP.yprim[h] >= Epsilon.loc[0][0]*DRHP.wprim[h])

                    DRHP.c15 = Constraint(expr=RRprim.loc[h-1][0]*DRHP.yprim[h] == DRHP.lprim[h])

                    solver = SolverFactory('cplex')
                    solver.solve(DRHP)
                    display(DRHP)

                    L.loc[h] = value(RR.loc[h-1][0]*DRHP.yprim[h])

                    for t in range((h-1)*tau_sce+1, (h-1)*tau_sce+tau_sce+1):
                        Backlog.loc[t] = value(DRHP.bl[t])
                        Inventoryraw.loc[t] = value(DRHP.invr[t])
                        Inventoryfinal.loc[t] = value(DRHP.invf[t])



                    RealCost.loc[h] = value((Cdo.loc[h-1] *DRHP.k[h] + Cvp.loc[h-1]*DRHP.xprim[h]
                                                    + Cwp.loc[h-1]*DRHP.yprim[h] + Cr.loc[h-1]*L.loc[h]
                                                    + Ccv.loc[h-1]*DRHP.vprim[h] + Ccw.loc[h-1]*DRHP.wprim[h]
                                                    + sum (Cinvr.loc[t-1]*DRHP.invr[t] +
                                                           Cinvf.loc[t-1]*DRHP.invf[t] +
                                                           Cpro.loc[t-1]*DRHP.pq[t]
                                                           for t in range (1+(h - 1)*tau_sce,(h - 1)*tau_sce+tau_sce+1))
                                                    + sum (Cbl.loc[t-1]*(D.loc[t]-DRHP.sq[t])
                                                           for t in range (1+(h - 1)*tau_sce,(h - 1)*tau_sce+tau_sce+1))).loc[0])

                    Invr.loc[h] = value(DRHP.invr[(h-1)*tau_sce+tau_sce])
                    Invf.loc[h] = value(DRHP.invf[(h-1)*tau_sce+tau_sce])
                    X.loc[h] = value(DRHP.xprim[h])
                    K.loc[h] = value(DRHP.k[h])

                    with pd.ExcelWriter(address_case, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:

                        Invr.to_excel(writer, sheet_name='Invr', index=False, header = False )
                        Invf.to_excel(writer, sheet_name='Invf', index=False, header = False )
                        L.to_excel(writer, sheet_name='L', index=False, header = False )
                        X.to_excel(writer, sheet_name='X', index=False, header = False )
                        RealCost.to_excel(writer, sheet_name='RealCost', index=False, header = False )
                        Backlog.to_excel(writer, sheet_name='Backlog', index=False, header = False )
                        Inventoryraw.to_excel(writer, sheet_name='Inventoryraw', index=False, header = False )
                        Inventoryfinal.to_excel(writer, sheet_name='Inventoryfinal', index=False, header = False )
                        K.to_excel(writer, sheet_name='K', index=False, header = False )


                outputs_for_all_runs = {}
                filename = f"Python-RHP-DATA-Case1-DemandS{scenario_d},RRS{scenario_r},tau{tau_sce},{pred_sce}.xlsx"
                outputs_for_all_runs["filename"] = filename
                with pd.ExcelWriter(address_case, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                    mean = RealCost.iloc[1:,0].mean()
                    RealCost[1] = None
                    RealCost.iloc[0,1]=mean
                    RealCost.to_excel(writer, sheet_name='RealCost', index=False, header = False )
                    outputs_for_all_runs["RealCost"] = mean

                with pd.ExcelWriter(address_case, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                    summ = Backlog.iloc[1:,0].sum()
                    Backlog[1] = None
                    Backlog.iloc[0,1]=summ
                    Backlog.to_excel(writer, sheet_name='Backlog', index=False, header = False )
                    outputs_for_all_runs["Backlog"] = summ

                with pd.ExcelWriter(address_case, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                    summ = Inventoryraw.iloc[1:,0].sum()
                    Inventoryraw[1] = None
                    Inventoryraw.iloc[0,1]=summ
                    Inventoryraw.to_excel(writer, sheet_name='Inventoryraw', index=False, header = False )
                    outputs_for_all_runs["Inventoryraw"] = summ

                with pd.ExcelWriter(address_case, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                    summ = Inventoryfinal.iloc[1:,0].sum()
                    Inventoryfinal[1] = None
                    Inventoryfinal.iloc[0,1]=summ
                    Inventoryfinal.to_excel(writer, sheet_name='Inventoryfinal', index=False, header = False )
                    outputs_for_all_runs["Inventoryfinal"] = summ

                with pd.ExcelWriter(address_case, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                    summ = K.iloc[1:,0].sum()
                    K[1] = None
                    K.iloc[0,1]=summ
                    K.to_excel(writer, sheet_name='K', index=False, header = False )
                    outputs_for_all_runs["K"] = summ


                df_output.loc[filename] = [outputs_for_all_runs["filename"] , outputs_for_all_runs["RealCost"] , outputs_for_all_runs["Backlog"] , outputs_for_all_runs["Inventoryraw"] , outputs_for_all_runs["Inventoryfinal"] , outputs_for_all_runs["K"]  ]


                with pd.ExcelWriter(address_case, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                    Dprim.to_excel(writer, sheet_name='Dprim_py', index=False, header = False )
                with pd.ExcelWriter(address_case, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                    RRprim.to_excel(writer, sheet_name='RRprim_py', index=False, header = False )
                with pd.ExcelWriter(address_case, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                    Horizons.to_excel(writer, sheet_name='Horizons_py', index=False, header = False )
                with pd.ExcelWriter(address_case, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                    Days.to_excel(writer, sheet_name='Days_py', index=False, header = False )

                df_cost[tau_sce][pred_sce] =  outputs_for_all_runs["RealCost"]

        df_cost.to_excel(writer_cost, sheet_name=f'DS{scenario_d}_RRS{scenario_r}')


writer_cost.close()
df_output.to_excel("outputs/outputs.xlsx")