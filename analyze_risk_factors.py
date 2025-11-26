import csv
from collections import Counter

# Load original data to examine raw distributions
parser_data = []
with open('MOCK_DATA_FINAL.csv', 'r') as f:
    reader = csv.DictReader(f)
    parser_data = list(reader)

print('Raw Data Analysis for Risk Calibration:')
print('='*50)

# Mental health conditions
mental_health = [row['mental_health_condition'] for row in parser_data]
mh_counts = Counter(mental_health)
print(f'Mental Health Conditions:')
for condition, count in mh_counts.items():
    print(f'  {condition}: {count} ({count/len(parser_data)*100:.1f}%)')

# Employment status  
employment = [row['employment_status'] for row in parser_data]
emp_counts = Counter(employment)
print(f'\nEmployment Status:')
for status, count in emp_counts.items():
    print(f'  {status}: {count} ({count/len(parser_data)*100:.1f}%)')

# Insurance coverage
insurance = [row['insurance_coverage'] for row in parser_data]
ins_counts = Counter(insurance)
print(f'\nInsurance Coverage:')
for coverage, count in ins_counts.items():
    print(f'  {coverage}: {count} ({count/len(parser_data)*100:.1f}%)')

# Healthcare access
healthcare = [row['access_to_healthcare'] for row in parser_data]
hc_counts = Counter(healthcare)
print(f'\nHealthcare Access:')
for access, count in hc_counts.items():
    print(f'  {access}: {count} ({count/len(parser_data)*100:.1f}%)')

# Debt levels
debts = [float(row['debt_amount'].replace('$', '').replace(',', '')) for row in parser_data]
print(f'\nDebt Distribution:')
print(f'  Mean: ${sum(debts)/len(debts):,.2f}')
print(f'  Max: ${max(debts):,.2f}')
print(f'  >$200K: {sum(1 for d in debts if d > 200000)} ({sum(1 for d in debts if d > 200000)/len(debts)*100:.1f}%)')
print(f'  >$150K: {sum(1 for d in debts if d > 150000)} ({sum(1 for d in debts if d > 150000)/len(debts)*100:.1f}%)')

# Stress levels
stress = [int(row['stress_level']) for row in parser_data]
print(f'\nStress Levels (1-10):')
print(f'  Mean: {sum(stress)/len(stress):.1f}')
print(f'  High stress (8-10): {sum(1 for s in stress if s >= 8)} ({sum(1 for s in stress if s >= 8)/len(stress)*100:.1f}%)')

# Resilience scores
resilience = [int(row['resilience_score']) for row in parser_data]
print(f'\nResilience Scores:')
print(f'  Mean: {sum(resilience)/len(resilience):.1f}')
print(f'  Low resilience (<30): {sum(1 for r in resilience if r < 30)} ({sum(1 for r in resilience if r < 30)/len(resilience)*100:.1f}%)')

# Find people with multiple risk factors
print(f'\nHigh-Risk Combinations:')
high_risk_people = []
for person in parser_data:
    risk_factors = 0
    factors = []
    
    # Unemployment
    if person['employment_status'] == 'Unemployed':
        risk_factors += 1
        factors.append('Unemployed')
    
    # High debt (>$150K)
    debt = float(person['debt_amount'].replace('$', '').replace(',', ''))
    if debt > 150000:
        risk_factors += 1
        factors.append(f'High Debt (${debt:,.0f})')
    
    # No insurance
    if person['insurance_coverage'] == 'false':
        risk_factors += 1
        factors.append('No Insurance')
    
    # No healthcare access
    if person['access_to_healthcare'] == 'false':
        risk_factors += 1
        factors.append('No Healthcare Access')
    
    # Mental health condition
    if person['mental_health_condition'] != 'None':
        risk_factors += 1
        factors.append(f"Mental Health: {person['mental_health_condition']}")
    
    # High stress
    if int(person['stress_level']) >= 8:
        risk_factors += 1
        factors.append(f"High Stress ({person['stress_level']})")
    
    # Low resilience
    if int(person['resilience_score']) < 30:
        risk_factors += 1
        factors.append(f"Low Resilience ({person['resilience_score']})")
    
    if risk_factors >= 4:  # 4+ risk factors = high risk
        high_risk_people.append((risk_factors, factors, person))

print(f'People with 4+ risk factors: {len(high_risk_people)}')
if high_risk_people:
    # Show top 3 most at-risk
    high_risk_people.sort(key=lambda x: x[0], reverse=True)  # Sort by risk factor count
    for i, (count, factors, person) in enumerate(high_risk_people[:3]):
        print(f'\n  Person {i+1}: {count} risk factors')
        print(f'    Age: {person["age"]}, Income: {person["income"]}')
        print(f'    Risk factors: {", ".join(factors)}')