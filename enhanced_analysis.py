import csv
import numpy as np
from collections import Counter

# Load the enhanced dataset
with open('ml_datasets/training_dataset.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

print("ENHANCED RISK/RESILIENCE SCORING ANALYSIS")
print("="*70)
print("Now includes: Age, Gender, Credit Score, Marital Status, Family Size")
print("="*70)

# Gender impact analysis
print("\n1. GENDER-BASED RISK IMPACT")
print("-"*40)
gender_risk_analysis = {}
for row in data:
    gender_code = row['gender']
    risk_score = float(row['calculated_risk_score'])
    if gender_code not in gender_risk_analysis:
        gender_risk_analysis[gender_code] = []
    gender_risk_analysis[gender_code].append(risk_score)

# Map gender codes back to names (from the original data)
gender_names = {
    '0': 'Agender', '1': 'Bigender', '2': 'Female', '3': 'Genderfluid',
    '4': 'Genderqueer', '5': 'Male', '6': 'Non-binary', '7': 'Polygender'
}

for code, risks in sorted(gender_risk_analysis.items()):
    if code in gender_names:
        avg_risk = np.mean(risks)
        count = len(risks)
        gender_name = gender_names.get(code, f'Code {code}')
        print(f"{gender_name:12} (n={count:2d}): Avg Risk {avg_risk:.1f}")

# Age impact analysis
print("\n2. AGE-BASED RISK PATTERNS")
print("-"*40)
age_groups = {
    'Under 25': [],
    '25-34': [],
    '35-49': [],
    '50-64': [],
    '65+': []
}

for row in data:
    age = int(row['age'])
    risk_score = float(row['calculated_risk_score'])
    resilience_score = float(row['calculated_resilience_score'])
    
    if age < 25:
        age_groups['Under 25'].append((risk_score, resilience_score))
    elif age < 35:
        age_groups['25-34'].append((risk_score, resilience_score))
    elif age < 50:
        age_groups['35-49'].append((risk_score, resilience_score))
    elif age < 65:
        age_groups['50-64'].append((risk_score, resilience_score))
    else:
        age_groups['65+'].append((risk_score, resilience_score))

for age_group, scores in age_groups.items():
    if scores:
        avg_risk = np.mean([s[0] for s in scores])
        avg_resilience = np.mean([s[1] for s in scores])
        count = len(scores)
        print(f"{age_group:8} (n={count:3d}): Risk {avg_risk:.1f}, Resilience {avg_resilience:.1f}")

# Credit score impact analysis
print("\n3. CREDIT SCORE IMPACT")
print("-"*40)
credit_ranges = {
    'Poor (300-599)': [],
    'Fair (600-699)': [],
    'Good (700-749)': [],
    'Excellent (750+)': []
}

for row in data:
    credit_score = int(row['credit_score'])
    risk_score = float(row['calculated_risk_score'])
    resilience_score = float(row['calculated_resilience_score'])
    
    if credit_score < 600:
        credit_ranges['Poor (300-599)'].append((risk_score, resilience_score))
    elif credit_score < 700:
        credit_ranges['Fair (600-699)'].append((risk_score, resilience_score))
    elif credit_score < 750:
        credit_ranges['Good (700-749)'].append((risk_score, resilience_score))
    else:
        credit_ranges['Excellent (750+)'].append((risk_score, resilience_score))

for credit_range, scores in credit_ranges.items():
    if scores:
        avg_risk = np.mean([s[0] for s in scores])
        avg_resilience = np.mean([s[1] for s in scores])
        count = len(scores)
        print(f"{credit_range:18} (n={count:3d}): Risk {avg_risk:.1f}, Resilience {avg_resilience:.1f}")

# Marital status analysis
print("\n4. MARITAL STATUS IMPACT")
print("-"*40)
marital_map = {0: 'Divorced', 1: 'Single', 2: 'Widowed', 3: 'Married'}
marital_analysis = {}

for row in data:
    marital_code = int(row['marital_status'])
    marital_status = marital_map.get(marital_code, 'Unknown')
    risk_score = float(row['calculated_risk_score'])
    resilience_score = float(row['calculated_resilience_score'])
    
    if marital_status not in marital_analysis:
        marital_analysis[marital_status] = []
    marital_analysis[marital_status].append((risk_score, resilience_score))

for status, scores in marital_analysis.items():
    if scores:
        avg_risk = np.mean([s[0] for s in scores])
        avg_resilience = np.mean([s[1] for s in scores])
        count = len(scores)
        print(f"{status:8} (n={count:3d}): Risk {avg_risk:.1f}, Resilience {avg_resilience:.1f}")

# Family size with marital context
print("\n5. FAMILY SIZE + MARITAL CONTEXT")
print("-"*40)
for row in data[:10]:  # Sample analysis
    family_size = int(row['family_size'])
    marital_status = marital_map.get(int(row['marital_status']), 'Unknown')
    risk_score = float(row['calculated_risk_score'])
    resilience_score = float(row['calculated_resilience_score'])
    
    print(f"Family size {family_size}, {marital_status:8}: Risk {risk_score:.0f}, Resilience {resilience_score:.0f}")

# Overall impact comparison
print("\n6. ENHANCED SCORING VALIDATION")
print("-"*40)

high_risk = [r for r in data if r['risk_category'] == 'High Risk']
medium_risk = [r for r in data if r['risk_category'] == 'Medium Risk']
low_risk = [r for r in data if r['risk_category'] == 'Low Risk']

print(f"Risk Distribution:")
print(f"  High Risk:   {len(high_risk):3d} ({len(high_risk)/len(data)*100:.1f}%)")
print(f"  Medium Risk: {len(medium_risk):3d} ({len(medium_risk)/len(data)*100:.1f}%)")
print(f"  Low Risk:    {len(low_risk):3d} ({len(low_risk)/len(data)*100:.1f}%)")

# Show examples of how new factors affect final categorization
print(f"\nEXAMPLE: How Credit Score Changes Risk Category")
print("-"*50)
# Find someone with poor credit
poor_credit = [r for r in data if int(r['credit_score']) < 500][:2]
excellent_credit = [r for r in data if int(r['credit_score']) > 750][:2]

if poor_credit:
    print("Poor Credit Examples:")
    for person in poor_credit:
        print(f"  Credit: {person['credit_score']}, Risk: {person['calculated_risk_score']}, Category: {person['risk_category']}")

if excellent_credit:
    print("Excellent Credit Examples:")
    for person in excellent_credit:
        print(f"  Credit: {person['credit_score']}, Resilience: {person['calculated_resilience_score']}, Category: {person['risk_category']}")

print(f"\nSCORING BREAKDOWN (Max Points):")
print(f"RISK FACTORS:")
print(f"  • Employment Risk: 25 pts")
print(f"  • Debt Burden: 25 pts") 
print(f"  • Credit Score Risk: 20 pts (WEIGHTED HEAVILY)")
print(f"  • Age Risk: 15 pts")
print(f"  • Gender Risk: 8 pts")
print(f"  • Marital Stress: 10 pts")
print(f"  • Family Burden: 8 pts")
print(f"  • Healthcare Access: 15 pts")
print(f"  • Mental Health/Stress: 15 pts")
print(f"  • Social/Sleep: 8 pts")

print(f"\nRESILIENCE FACTORS:")
print(f"  • Credit Score Resilience: 25 pts (WEIGHTED HEAVILY)")
print(f"  • Income/Financial: 25 pts")
print(f"  • Savings Buffer: 12 pts")
print(f"  • Employment Stability: 12 pts")
print(f"  • Age Wisdom: 8 pts")
print(f"  • Marital/Family Support: 8 pts")
print(f"  • Education: 12 pts")
print(f"  • Health/Activity: 11 pts")
print(f"  • Social Support: 8 pts")
print(f"  • Financial Goals: 5 pts")