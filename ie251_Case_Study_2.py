# Case Study 2 - Group16

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

numComp = 5
numPlant = 3
numPeriod = 2
components = ["Aliminum", "CarbonFiber", "Manual", "Control", "Sensor"]
plants = ["Istanbul", "Ankara", "Izmir"]
periods = [1, 2]


laborData = [[1, 1.5, 1.5, 3, 4], [3.5, 3.5, 4.5, 4.5, 5], [3, 3.5, 4, 4.5, 5.5]]
laborAvailability = [12000, 15000, 22000]
packingData = [[4, 4, 5, 6, 6], [7, 7, 8, 9, 7], [7.5, 7.5, 8.5, 8.5, 8]]
packingAvailability = [20000, 40000, 35000]
assemblyData = [65, 60, 65]
assemblyAvailability = [5500, 5000, 6000]
minDemandComponents = [[0, 100, 200, 30, 100], [0, 100, 200, 30, 100], [0, 50, 100, 15, 100]]
maxDemandComponents = [[2000, 2000, 2000, 2000, 2000], [2000, 2000, 2000, 2000, 2000], [2000, 2000, 2000, 2000, 2000]]
minDemandRoboticKit = [0, 0, 0]
maxDemandRoboticKit = [200, 200, 200]
productionCostComponents = [[6, 19, 4, 10, 26], [5, 18, 5, 11, 24], [7, 20, 5, 12, 27]]
productionCostRoboticKits = [178, 175, 180]
sellingPriceComponents = [[10, 25, 8, 18, 40], [10, 25, 8, 18, 40], [12, 30, 10, 22, 45]]
sellingPriceRoboticKits = [290, 290, 310]
req = [13, 13, 10, 3, 3]

sPC={}
pCC={}
maxDC={}
minDC={}
pData = {}
labor = {}
for k in range(numPlant):
    for i in range(numComp):
        for t in range(numPeriod):
           sPC[(plants[k], components[i], periods[t])] = sellingPriceComponents[k][i]
           pCC[(plants[k], components[i], periods[t])] = productionCostComponents[k][i]
           maxDC[(plants[k], components[i], periods[t])] = maxDemandComponents[k][i]
           minDC[(plants[k], components[i], periods[t])] = minDemandComponents[k][i]
           pData[(plants[k], components[i], periods[t])] = packingData[k][i]
           labor[(plants[k], components[i], periods[t])] = laborData[k][i]
lA = {}
pA = {}
aD = {}
aA = {}
minDR = {}
maxDR = {}
pCR = {}
sPR = {}

for k in range(numPlant):
    for t in range(numPeriod):
        lA[(plants[k],periods[t])] = laborAvailability[k]
        pA[(plants[k],periods[t])] = packingAvailability[k]
        aD[(plants[k],periods[t])] = assemblyData[k]
        aA[(plants[k],periods[t])] = assemblyAvailability[k]
        minDR[(plants[k],periods[t])] = minDemandRoboticKit[k]
        maxDR[(plants[k],periods[t])] = maxDemandRoboticKit[k]
        pCR[(plants[k],periods[t])] = productionCostRoboticKits[k]
        sPR[(plants[k],periods[t])] = sellingPriceRoboticKits[k]

r = {} 
for i in range(numComp):
        r[(components[i])] = req[i]
        
#Construct the model
mdl = pyo.ConcreteModel('IE-Tech') 
#Define sets
mdl.I = pyo.Set(initialize=plants, doc='plants' ) 
mdl.J = pyo.Set(initialize=components, doc='components')
mdl.K = pyo.Set(initialize=periods, doc='periods')

#Define parameters
mdl.pL = pyo.Param(mdl.I, mdl.J, mdl.K, initialize= labor, doc='labor data')
mdl.pLA= pyo.Param(mdl.I, mdl.K , initialize=lA , doc='labor availability')
mdl.pPD= pyo.Param(mdl.I, mdl.J, mdl.K, initialize=pData, doc='packing Data')
mdl.pPA= pyo.Param(mdl.I, mdl.K, initialize=pA, doc='packing availabiity')
mdl.pAD = pyo.Param(mdl.I , mdl.K, initialize= aD , doc='assembly Data')
mdl.pAA =  pyo.Param(mdl.I , mdl.K, initialize= aA , doc='assembly availability')
mdl.pNDC =  pyo.Param(mdl.I, mdl.J, mdl.K, initialize= minDC , doc ='minimum demand components')
mdl.pXDC = pyo.Param(mdl.I, mdl.J, mdl.K, initialize= maxDC , doc = 'maximum demand components')
mdl.pNDR =  pyo.Param(mdl.I , mdl.K, initialize= minDR , doc ='minimum demand robotic kit')
mdl.pXDR = pyo.Param(mdl.I , mdl.K, initialize= maxDR , doc = 'maximum demand robotic kit')
mdl.pPCC =  pyo.Param(mdl.I, mdl.J, mdl.K, initialize= pCC , doc ='production Cost Components')
mdl.pPCR =  pyo.Param(mdl.I , mdl.K ,initialize= pCR, doc ='production Cost of Robotic Kit')
mdl.pSPC =  pyo.Param(mdl.I, mdl.J, mdl.K, initialize= sPC , doc ='selling price for components')
mdl.pSPR =  pyo.Param(mdl.I , mdl.K, initialize= sPR , doc = 'Selling price of robotic kit')
mdl.pR =  pyo.Param(mdl.J, initialize= r , doc = 'robotic kit manufacturing coefficients')


#Define decision variables
mdl.vX = pyo.Var(mdl.I , mdl.J, mdl.K , bounds=(0.0,None), doc= 'The amount of component j in plant i at period k' , within =pyo.NonNegativeReals)
mdl.vY = pyo.Var(mdl.I, mdl.K , bounds=(0.0,None), doc = 'Number of robotic kit produced in plant i at period k' ,  within =pyo.NonNegativeReals)
mdl.vI = pyo.Var(mdl.I , mdl.J, mdl.K , bounds=(0.0,None), doc= 'The amount of component j in plant i inventory at period k' , within =pyo.NonNegativeReals)
mdl.vIR = pyo.Var(mdl.I, mdl.K , bounds=(0.0,None), doc= 'The amount of robotic kit in plant i inventory at period k' , within =pyo.NonNegativeReals)


#Define Constraints

def eLabCons(mdl,i,k):
    return sum(mdl.pL[i,j,k]*mdl.vX[i,j,k] for j in mdl.J)<= mdl.pLA[i,k]
mdl.eLabCons = pyo.Constraint(mdl.I, mdl.K, rule=eLabCons, doc = 'Labor Time Avaliability Constraint')

def ePacCons(mdl,i,k):
    return sum(mdl.pPD[i,j,k]*mdl.vX[i,j,k] for j in mdl.J) <= mdl.pPA[i,k]
mdl.ePacCons = pyo.Constraint(mdl.I, mdl.K, rule = ePacCons, doc ='Packing Time Availability Constraint')

def AssCons(mdl,i,k):
    return (mdl.vY[i,k]*mdl.pAD[i,k]) <= mdl.pAA[i,k]
mdl.AssCons = pyo.Constraint(mdl.I, mdl.K , rule = AssCons, doc ='Robotic Kit Assembly Time Availability Constraint')

def CfCons(mdl, k):
    return sum(mdl.vX[i, 'CarbonFiber', k] for i in mdl.I) <= 4000
mdl.CfCons = pyo.Constraint(mdl.K, rule=CfCons, doc='Carbon Fiber Availability Constraint')


def eCompInv(mdl, i, j):
    return mdl.vI[i, j, 2] == 0
mdl.eCompInv = pyo.Constraint(mdl.I, mdl.J, rule=eCompInv, doc='2nd period Component Inventory Constraint')

def eRKInv(mdl, i):
    return mdl.vIR[i, 2] == 0
mdl.eRKInv = pyo.Constraint(mdl.I, rule=eRKInv, doc='2nd period Robotic Kit Inventory Constraint')

#Max-Min Demand of Robotic Kits

def eMRK(mdl,i):
    if k == 1:
        return mdl.vY[i,k]-mdl.vIR[i,k] >= mdl.pNDR[i,k]
    elif k == 2:
        return mdl.vY[i,k]+mdl.vIR[i,1]-mdl.vIR[i,2] >= mdl.pNDR[i,k]
mdl.eMRK = pyo.Constraint(mdl.I,  rule = eMRK, doc= ' Minimum Robotic Kit Demand ')

def eMaxRK(mdl,i,k):
    if k == 1:
        return mdl.vY[i,k]-mdl.vIR[i,1] <= mdl.pXDR[i,k]
    elif k == 2:
        return mdl.vY[i, k]+mdl.vIR[i,1]-mdl.vIR[i,2] <= mdl.pXDR[i, k]
mdl.eMaxRK = pyo.Constraint(mdl.I, mdl.K ,  rule = eMaxRK, doc= ' Maximum Robotic Kit Demand for period k=1')

#Max-Min Demand of Components

def eMDC1(mdl, i, j):
    return mdl.vX[i, j, 1] - mdl.vI[i, j, 1]- mdl.vY[i,1]*mdl.pR[j] >= mdl.pNDC[i, j, 1]
mdl.eMDC1 = pyo.Constraint(mdl.I, mdl.J, rule=eMDC1, doc='Minimum Demand for Components in Period 1')

def eMDC2(mdl, i, j):
    return mdl.vX[i, j, 2] + mdl.vI[i, j, 1] - mdl.vI[i, j, 2]-mdl.vY[i,2]*mdl.pR[j] >= mdl.pNDC[i, j, 2]
mdl.eMDC2 = pyo.Constraint(mdl.I, mdl.J, rule=eMDC2, doc='Minimum Demand for Components in Period 2')

def eMaxDC1(mdl, i, j):
    return mdl.vX[i, j, 1] - mdl.vI[i, j, 1]-mdl.vY[i,1]*mdl.pR[j] <= mdl.pXDC[i, j, 1]
mdl.eMaxDC1 = pyo.Constraint(mdl.I, mdl.J, rule=eMaxDC1, doc='Maximum Demand for Components in Period 1')

def eMaxDC2(mdl, i, j):
    return mdl.vX[i, j, 2] + mdl.vI[i, j, 1] - mdl.vI[i, j, 2]-mdl.vY[i,2]*mdl.pR[j] <= mdl.pXDC[i, j, 2]
mdl.eMaxDC2 = pyo.Constraint(mdl.I, mdl.J, rule=eMaxDC2, doc='Maximum Demand for Components in Period 2')

#Objective Function
def oTotal(mdl):
        return (sum(mdl.vX[i, j, k] * mdl.pSPC[i, j, k] for i in mdl.I for j in mdl.J for k in mdl.K) +sum(mdl.vY[i, k] * mdl.pSPR[i, k] for i in mdl.I for k in mdl.K) -  sum(mdl.vX[i, j, k] * (mdl.pPCC[i, j, k] * (1.12 if k == 2 else 1.0)) for i in mdl.I for j in mdl.J for k in mdl.K) -sum(mdl.vY[i, k] * (mdl.pPCR[i, k] * (1.12 if k == 2 else 1.0)) for i in mdl.I for k in mdl.K) -sum(mdl.vY[i, k] * sum(mdl.pPCC[i, j, k] * mdl.pR[j] * (1.12 if k == 2 else 1.0) for j in mdl.J) for i in mdl.I for k in mdl.K) -0.08 * sum(mdl.vI[i, j, k] * mdl.pPCC[i, j, k] for i in mdl.I for j in mdl.J for k in mdl.K) -0.08 * sum(mdl.vIR[i, k] * mdl.pPCR[i, k] for i in mdl.I for k in mdl.K))
mdl.oTotal = pyo.Objective(rule = oTotal, sense = pyo.maximize, doc = ' Maximum Profit')

mdl.write('mdl_labels.lp', io_options={'symbolic_solver_labels': True})
mdl.write('mdl_nolabels.lp', io_options={'symbolic_solver_labels': False})

mdl.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT) 
mdl.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT) 

Solver = SolverFactory('glpk')
Solver.options['ranges']= r'/Users/emrebasaran/Desktop/SA_Report.txt'
SolverResults = Solver.solve(mdl, tee=True) 
SolverResults.write()
mdl.pprint() 
mdl.vX.display() 
mdl.oTotal.display()

import pyomo_sens_analysis_v2 as pyo_SA
pyo_SA.reorganize_SA_report(file_path_SA = r'/Users/emrebasaran/Desktop/SA_Report.txt', \
                            file_path_LP_labels = r'/Users/emrebasaran/Desktop/mdl_labels.lp', \
                                file_path_LP_nolabels = r'/Users/emrebasaran/Desktop/mdl_nolabels.lp')

#export to excel: variables 
with pd.ExcelWriter(r'/Users/emrebasaran/Desktop/IETech.xlsx', engine='openpyxl') as writer:
    #vX
    variableX_data = {(i, j, k, v.name): pyo.value(v) for (i, j, k), v in mdl.vX.items()} 
    optimal_solution_print = pd.DataFrame.from_dict(variableX_data, orient="index", columns=["variable value"]) 
    optimal_solution_print.to_excel(writer, sheet_name='vX')
    #vY 
    variableY_data = {(i, k, v.name): pyo.value(v) for (i, k), v in mdl.vY.items()} 
    optimal_solution_print = pd.DataFrame.from_dict(variableY_data, orient="index", columns=["variable value"]) 
    optimal_solution_print.to_excel(writer, sheet_name='vY')
    #vI
    variableI_data = {(i, j, k, v.name): pyo.value(v) for (i, j, k), v in mdl.vI.items()} 
    optimal_solution_print = pd.DataFrame.from_dict(variableI_data, orient="index", columns=["variable value"]) 
    optimal_solution_print.to_excel(writer, sheet_name='vI')  
    #vIR
    variableIR_data = {(i, k, v.name): pyo.value(v) for (i, k), v in mdl.vIR.items()} 
    optimal_solution_print = pd.DataFrame.from_dict(variableIR_data, orient="index", columns=["variable value"]) 
    optimal_solution_print.to_excel(writer, sheet_name='vIR')

#export to excel: reduced costs 
reduced_cost_dict={str(key):mdl.rc[key] for key in mdl.rc.keys()} 
Reduced_Costs_print =pd.DataFrame.from_dict(reduced_cost_dict,orient="index", columns=["reduced cost"]) 
Reduced_Costs_print.to_excel(r'/Users/emrebasaran/Desktop/ReducedCosts.xlsx',sheet_name='ReducedCosts') 

#export to excel: shadow prices
duals_dict = {str(key): mdl.dual[key] for key in mdl.dual.keys()}
u_slack_dict = {
    **{str(con): con.uslack() for con in mdl.component_objects(pyo.Constraint) if not con.is_indexed()},
    **{k: v for con in mdl.component_objects(pyo.Constraint) if con.is_indexed() for k, v in
       {'{}[{}]'.format(str(con), key): con[key].uslack()
        for key in con.keys()}.items()}
}

l_slack_dict = {
    **{str(con): con.lslack() for con in mdl.component_objects(pyo.Constraint) if not con.is_indexed()},
    **{k: v for con in mdl.component_objects(pyo.Constraint) if con.is_indexed() for k, v in
       {'{}[{}]'.format(str(con), key): con[key].lslack()
        for key in con.keys()}.items()}
}
# Combine into a single df
Shadow_Prices_print = pd.concat([pd.Series(d, name=name) for name, d in {'duals': duals_dict, 'uslack': u_slack_dict, 'lslack': l_slack_dict}.items()], axis='columns')
Shadow_Prices_print.to_excel(r'/Users/emrebasaran/Desktop/ShadowPrices.xlsx' , sheet_name='ShadowPrices')

#  WE WANNA BONUS...
# For this reason:

# REVENUE/COST DISTRIBUTIONS:

# REVENUES:
# Revenue from Selling components
revenueFromSellingComponents = sum(mdl.vX[i, j, k].value * mdl.pSPC[i, j, k] for i in mdl.I for j in mdl.J for k in mdl.K)
revenueFromSellingComponents = int(revenueFromSellingComponents)
# Revenue from Selling Robotic Kits
revenueFromSellingRoboticKits = sum(mdl.vY[i, k].value * mdl.pSPR[i, k] for i in mdl.I for k in mdl.K)
revenueFromSellingRoboticKits = int(revenueFromSellingRoboticKits)

# Production Costs:
# Production Cost of Components
productionCostofComponents = sum(mdl.vX[i, j, k].value * (mdl.pPCC[i, j, k] * (1.12 if k == 2 else 1.0)) for i in mdl.I for j in mdl.J for k in mdl.K)
productionCostofComponents = int(productionCostofComponents)
# Production Cost of Robotic Kits
productionCostofRoboticKits = sum(mdl.vY[i, k].value * (mdl.pPCR[i, k] * (1.12 if k == 2 else 1.0)) for i in mdl.I for k in mdl.K) - sum(mdl.vY[i, k].value * sum(mdl.pPCC[i, j, k] * mdl.pR[j] * (1.12 if k == 2 else 1.0) for j in mdl.J) for i in mdl.I for k in mdl.K)
productionCostofRoboticKits = int(productionCostofRoboticKits)

# Inventory Holding Costs:
# Inventory Holding Cost of Components
inventoryCostofComponents = 0.08 * sum(mdl.vI[i, j, k].value * mdl.pPCC[i, j, k] for i in mdl.I for j in mdl.J for k in mdl.K)
inventoryCostofComponents = int(inventoryCostofComponents)
# Inventory Holding Cost of Robotic Kits
inventoryCostofRoboticKits = 0.08 * sum(mdl.vIR[i, k].value * mdl.pPCR[i, k] for i in mdl.I for k in mdl.K)
inventoryCostofRoboticKits = int(inventoryCostofRoboticKits)


totalRevenues = revenueFromSellingComponents + revenueFromSellingRoboticKits
totalProductionCost = productionCostofComponents + productionCostofRoboticKits
totalInventoryCost = inventoryCostofComponents + inventoryCostofRoboticKits
totalCosts = totalProductionCost + totalInventoryCost

# Pie Chart of Revenue-Cost Distribution
revCosts = [f"Total Revenues :{totalRevenues}", f"Total Costs: {totalCosts}"]
valuesRC = [totalRevenues, totalCosts]
labelsRC = ['Total Revenues', 'Total Costs']
pembiş = ['pink','red']
plt.figure(figsize=(7, 4))
wedges, texts, autotexts = plt.pie(valuesRC,labels=revCosts, startangle=90,autopct='%1.2f%%',textprops={'fontsize': 12},colors=pembiş)
plt.legend(wedges,labelsRC,title="Revenue-Cost Distribution",loc="lower right",bbox_to_anchor=(1, 0, 0.5, 1),fontsize=12)
plt.title('Revenue - Cost Distribution', fontsize=14)
plt.savefig('chart1.jpg', format='jpg', dpi=300)
plt.show()

# BAR CHART of COSTS
costs = [f"Total Production Cost of Components:{productionCostofComponents}", f"Total Production Cost of Robotic Kits:{productionCostofRoboticKits}", f"Total Inventory Cost of Components: {inventoryCostofComponents}", f"Total Inventory Cost of Robotic Kits: {inventoryCostofRoboticKits}"]
valuesCosts = [productionCostofComponents,productionCostofRoboticKits,inventoryCostofComponents,inventoryCostofRoboticKits]
labelsCosts = ["Total Production Cost of Components", "Total Production Cost of Robotic Kits", "Total Inventory Cost of Components", "Total Inventory Cost of Robotic Kits"]
tonSürTonTon = ['#006400', '#228B22', '#32CD32', '#98FB98'] #özel istek üzerine yaptık ama son 3 tanesi 0 :(

plt.figure(figsize=(8, 5))
plt.bar(labelsCosts, valuesCosts, color=tonSürTonTon)
plt.title('Cost Distribution', fontsize=14)
plt.ylabel('Costs', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig('chart2.jpg', format='jpg', dpi=300)
plt.show()

# BABA NEREDEN GELİYOR BU DEĞİRMENİN SUYU?
# PİE CHART of REVENUES
revenues = [f"Total Revenue from Selling Components: {revenueFromSellingComponents}", f"Total Revenue from Selling Robotic Kits: {revenueFromSellingRoboticKits}"]
valuesRev = [revenueFromSellingComponents, revenueFromSellingRoboticKits]
labelsRev = ["Total Revenue from Selling Components", "Total Revenue from Selling Robotic Kits"]
blues = ["blue", "red"]

plt.figure(figsize=(12, 4))
wedges, texts, autotexts = plt.pie(valuesRev,labels=revenues, startangle=90,autopct='%1.2f%%',textprops={'fontsize': 12},colors=pembiş)
plt.legend(wedges,labelsRev,title="Revenue Distribution",loc="upper left",bbox_to_anchor=(1, 0, 0.5, 1),fontsize=12)
plt.title('Revenue - Cost Distribution', fontsize=14)
plt.savefig('chart3.jpg', format='jpg', dpi=300)
plt.show()
# Component satışından geliyormuş xD

# ŞİMDİ ÜRETİMİN PLANT, COMPONENT ve PERİYOTLARA GÖRE DAĞILIMI VAR...

# Production according to the plants
angara = sum(mdl.vX[i,j,k].value for i in mdl.I for j in mdl.J for k in mdl.K if i == "Ankara") + sum(mdl.vY[i,k].value for i in mdl.I for k in mdl.K if i == "Ankara")
angara = int(angara)
# ULAN İZzzDANBUL SEN Mİ BÜYÜKSÜN BEN Mİ???
#                                          -ANKARA
izdanbul = sum(mdl.vX[i,j,k].value for i in mdl.I for j in mdl.J for k in mdl.K if i == "Istanbul") + sum(mdl.vY[i,k].value for i in mdl.I for k in mdl.K if i == "Istanbul")
izdanbul = int(izdanbul)
ızmir = sum(mdl.vX[i,j,k].value for i in mdl.I for j in mdl.J for k in mdl.K if i == "Izmir") + sum(mdl.vY[i,k].value for i in mdl.I for k in mdl.K if i == "Izmir")
ızmir = int(ızmir)
print("Plants : ", angara,izdanbul,ızmir)
# Pie Chart of Plant Distribution
pla = [f"Ankara :{angara}", f"İstanbul : {izdanbul}", f"İzmir : {ızmir}"]
valuesPla = [angara,izdanbul,ızmir]
labelsPla = ["Ankara","İstanbul","İzmir"]

plt.figure(figsize=(7, 4))
wedges, texts, autotexts = plt.pie(valuesPla,labels=pla, startangle=90,autopct='%1.2f%%',textprops={'fontsize': 12})
plt.legend(wedges,labelsPla,title="Plant Distribution",loc="lower right",bbox_to_anchor=(1, 0, 0.5, 1),fontsize=12)
plt.title('Distribution of Product according to the Plants', fontsize=14)
plt.savefig('chart4.jpg', format='jpg', dpi=300)
plt.show()
# İstanbul daha büyükmüş zaaa

# Production according to the components
aluminyum = sum(mdl.vX[i,j,k].value for i in mdl.I for j in mdl.J for k in mdl.K if j == "Aliminum")
aluminyum = int(aluminyum)
karbonFiber = sum(mdl.vX[i,j,k].value for i in mdl.I for j in mdl.J for k in mdl.K if j == "CarbonFiber")
karbonFiber = int(karbonFiber)
manuelModül = sum(mdl.vX[i,j,k].value for i in mdl.I for j in mdl.J for k in mdl.K if j == "Manual")
manuelModül = int(manuelModül)
advContModül = sum(mdl.vX[i,j,k].value for i in mdl.I for j in mdl.J for k in mdl.K if j == "Control")
advContModül = int(advContModül)
advSensModül = sum(mdl.vX[i,j,k].value for i in mdl.I for j in mdl.J for k in mdl.K if j == "Sensor")
advSensModül = int(advSensModül)
robocob = sum(mdl.vY[i,k].value for i in mdl.I for k in mdl.K)
robocob = int(robocob)
print("Components: ",aluminyum,karbonFiber,manuelModül,advContModül, advSensModül, robocob)

# Pie Chart of Component Distribution
comP = [f"Aluminum :{aluminyum}", f"Carbon Fiber Frame: {karbonFiber}", f"Manuel Modules: {manuelModül}",f"Adavnced Control Modules: {advContModül}", f"Advanced Sensor Modules: {advSensModül}", f"Robocob: {robocob}"]
valuesComp = [aluminyum,karbonFiber,manuelModül,advContModül, advSensModül, robocob]
labelsComp = ["Aliminum", "Carbon Fiber Frames", "Manual Modules", "Advanced Control Modules", "Advanced Sensor Modules", "Robotic Kit"]

plt.figure(figsize=(11, 4))
wedges, texts, autotexts = plt.pie(valuesComp,labels=comP, startangle=90,autopct='%1.2f%%',textprops={'fontsize': 12})
plt.legend(wedges,labelsComp,title="Component Distribution",loc="lower right",bbox_to_anchor=(1.7, 0.5, 0.5, 2),fontsize=12)
plt.title('Distribution of Product according to the Types ', fontsize=14)
plt.savefig('chart5.jpg', format='jpg', dpi=300)
plt.show()

# Production according to the period
ilkay = sum(mdl.vX[i,j,k].value for i in mdl.I for j in mdl.J for k in mdl.K if k ==1) + sum(mdl.vY[i,k].value for i in mdl.I for k in mdl.K if k ==1)
ilkay = int(ilkay)
ikinciay = sum(mdl.vX[i,j,k].value for i in mdl.I for j in mdl.J for k in mdl.K if k ==2) + sum(mdl.vY[i,k].value for i in mdl.I for k in mdl.K if k ==2)
ikinciay = int(ikinciay)
print("Period: ",ilkay,ikinciay)

# Pie Chart of Period Distribution
per = [f"First Period :{ilkay}", f"Second Period: {ikinciay}"]
valuesPer = [ilkay,ikinciay]
labelsPer = ['First Period', 'Second Period']

plt.figure(figsize=(6.6, 4))
wedges, texts, autotexts = plt.pie(valuesPer,labels=per, startangle=90,autopct='%1.2f%%',textprops={'fontsize': 12})
plt.legend(wedges,labelsPer,title="Period Distribution",loc="lower right",bbox_to_anchor=(1, 0, 0.5, 1),fontsize=12)
plt.title('Distribution of Product according to Periods', fontsize=14)
plt.savefig('chart6.jpg', format='jpg', dpi=300)
plt.show()


