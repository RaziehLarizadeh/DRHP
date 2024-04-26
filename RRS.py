#Libraries
import numpy as np
##Scenario1
### Constants
supplier_rates1 = {    'Supplier 1': {'mean': 66, 'std_dev': 3},
    'Supplier 2': {'mean': 63, 'std_dev': 3},
    'Supplier 3': {'mean': 70, 'std_dev': 2},
    'Supplier 4': {'mean': 87, 'std_dev': 1},
    'Supplier 5': {'mean': 90, 'std_dev': 10},
    'Supplier 6': {'mean': 52, 'std_dev': 4},
    'Supplier 7': {'mean': 81, 'std_dev': 2}}
num_days = 365
### Generate time series data for recycling rates with facility switches
data1 = []
current_supplier1 = np.random.choice(list(supplier_rates1.keys()))
days_with_current_supplier1 = np.random.randint(3, 8)
for day in range (1, num_days + 1):
    if days_with_current_supplier1 == 0:
        current_supplier1 = np.random.choice(list(supplier_rates.keys1()))
        days_with_current_supplier1 = np.random.randint(3, 8)
    supplier_name1 = current_supplier1
    mean_rate1 = supplier_rates1[supplier_name1]['mean']
    std_dev1 = supplier_rates1[supplier_name1]['std_dev']
    recycling_rate1 = np.random.normal(mean_rate1, std_dev1)
    recycling_rate1 = min (recycling_rate1, 98)
    data1.append ((day, round (recycling_rate1, 1)))
##Scenario2
### Constants
supplier_rates2 = {'Supplier 1': {'mean': 0.92, 'std': 0.004, 'probability': 0.7, 'next_prob': 0.9},
    'Supplier 2': {'mean': 0.85, 'std': 0.012, 'probability': 0.25, 'next_prob': 0.7},
    'Supplier 3': {'mean': 0.75, 'std': 0.013, 'probability': 0.05}}
num_days = 365
### Generate time series data recycling rates based on the selected supplier
suppliers2 = list(supplier_rates2.keys())
selected_suppliers2 = []
for day in range(num_days):
    if day == 0 or (selected_suppliers2[day - 1] == 'Supplier 1' and np.random.random() < supplier_rates2   ['Supplier 1']['next_prob']):
        current_supplier2 = 'Supplier 1'
    elif day == 0 or (selected_suppliers2[day - 1] == 'Supplier 2' and np.random.random() < supplier_rates2 ['Supplier 2']['next_prob']):
        current_supplier2 = 'Supplier 2'
    else:
        current_supplier2 = np.random.choice(suppliers2, p=[supplier_rates2[s]['probability'] for s in suppliers2])
    selected_suppliers2.append(current_supplier2)
recycling_rates2 = np.array([supplier_rates2[supplier]['mean'] for supplier in selected_suppliers2])
recycling_rates2 += np.random.normal(0, [supplier_rates2[supplier]['std'] for supplier in selected_suppliers2], num_days)
recycling_rates2 = np.clip(recycling_rates2, 0, 1)
##Scenario3
### Constants
supplier_rates3= {'Supplier 1': {'mean': 0.80, 'std': 0.004, 'probability': 0.7, 'next_prob': 0.33},
    'Supplier 2': {'mean': 0.85, 'std': 0.012, 'probability': 0.25, 'next_prob': 0.33},
    'Supplier 3': {'mean': 0.75, 'std': 0.013, 'probability': 0.05}}
num_days = 365
### Generate time series data recycling rates based on the selected supplier
suppliers3 = list(supplier_rates3.keys())
selected_suppliers3 = []
for day in range(num_days):
    if day == 0 or (selected_suppliers3[day - 1] == 'Supplier 1' and np.random.random() < supplier_rates3['Supplier 1']['next_prob']):
        current_supplier3 = 'Supplier 1'
    elif day == 0 or (selected_suppliers3[day - 1] == 'Supplier 2' and np.random.random() < supplier_rates3['Supplier 2']['next_prob']):
        current_supplier3 = 'Supplier 2'
    else:
        current_supplier3 = np.random.choice(suppliers3, p=[ supplier_rates3[s]['probability'] for s in suppliers3])
    selected_suppliers3.append(current_supplier3)
recycling_rates3 = np.array([supplier_rates3[supplier]['mean'] for supplier in selected_suppliers3])
recycling_rates3 += np.random.normal(0, [supplier_rates3[supplier]['std'] for supplier in selected_suppliers3], num_days)
recycling_rates3 = np.clip(recycling_rates3, 0, 1)
