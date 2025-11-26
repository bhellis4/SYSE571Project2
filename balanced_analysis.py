import csv
import numpy as np

# Load the training dataset to analyze the new scoring
with open('ml_datasets/training_dataset.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

print("BALANCED RISK VS RESILIENCE ANALYSIS")
print("="*60)

# Examples of how income outweighs debt and other balanced factors
print("\nEXAMPLES: How Income/Resilience Factors Balance Risk Factors")
print("-"*60)

# Find someone with high debt but high income
high_debt_high_income = [
    r for r in data 
    if float(r['debt_amount']) > 150000 and float(r['income']) > 150000
]

if high_debt_high_income:
    person = high_debt_high_income[0]
    print(f"Example 1 - High Debt BUT High Income:")
    print(f"  Debt: ${float(person['debt_amount']):,.0f}")
    print(f"  Income: ${float(person['income']):,.0f}")
    print(f"  Debt/Income Ratio: {float(person['debt_to_income_ratio']):.2f}")
    print(f"  Risk Score: {person['calculated_risk_score']} (debt adds risk)")
    print(f"  Resilience Score: {person['calculated_resilience_score']} (income adds resilience)")
    print(f"  Net Result: {person['net_risk_score']} → {person['risk_category']}")
    print()

# Find someone with low income but good savings/employment
good_savings_employment = [
    r for r in data
    if float(r['savings_amount']) > 100000 and r['employment_status'] == '2'
    and float(r['income']) < 100000
]

if good_savings_employment:
    person = good_savings_employment[0]
    print(f"Example 2 - Lower Income BUT High Savings + Stable Job:")
    print(f"  Income: ${float(person['income']):,.0f}")
    print(f"  Savings: ${float(person['savings_amount']):,.0f}")
    print(f"  Employment: Full-time")
    print(f"  Risk Score: {person['calculated_risk_score']}")
    print(f"  Resilience Score: {person['calculated_resilience_score']} (savings + employment)")
    print(f"  Net Result: {person['net_risk_score']} → {person['risk_category']}")
    print()

# Financial goals impact
has_goals = [r for r in data if r['financial_goals'] != '0']
no_goals = [r for r in data if r['financial_goals'] == '0']

print(f"Financial Goals Impact:")
print(f"  People WITH goals: {len(has_goals)} - Avg Resilience: {np.mean([float(r['calculated_resilience_score']) for r in has_goals]):.1f}")
print(f"  People WITHOUT goals: {len(no_goals)} - Avg Resilience: {np.mean([float(r['calculated_resilience_score']) for r in no_goals]):.1f}")
print()

# Show risk factors vs resilience factors breakdown
print("RISK vs RESILIENCE FACTOR BREAKDOWN")
print("-"*40)

print("RISK FACTORS (0-100 points):")
print("  • Unemployment: 25 pts")
print("  • Very High Debt (>$200K): 25 pts") 
print("  • High Debt (>$150K): 20 pts")
print("  • No Insurance: 12 pts")
print("  • No Healthcare: 8 pts")
print("  • Mental Health Issues: 10 pts")
print("  • High Stress: 10 pts")
print("  • Poor Sleep: 5 pts")
print("  • Low Social Support: 5 pts")

print("\nRESILIENCE FACTORS (0-100 points):")
print("  • High Income (>$200K): 20 pts")
print("  • Substantial Savings (>$200K): 15 pts")
print("  • Full-time Employment: 15 pts")
print("  • PhD Education: 12 pts")
print("  • Low Debt/Income Ratio (<0.3): 10 pts")
print("  • Strong Social Network (8+): 8 pts")
print("  • Excellent Health/Sleep: 11 pts")
print("  • Financial Goals: 5 pts")
print("  • Healthcare Access: 4 pts")

print(f"\nFINAL RISK CATEGORIZATION:")
print(f"  High Risk: Net Risk > 20 (Risk heavily outweighs resilience)")
print(f"  Medium Risk: Net Risk -10 to 20 (Balanced)")  
print(f"  Low Risk: Net Risk < -10 (Resilience outweighs risk)")

# Show specific examples of balanced scoring
print(f"\nSCORING VALIDATION:")
high_risk_sample = [r for r in data if r['risk_category'] == 'High Risk'][:2]
low_risk_sample = [r for r in data if r['risk_category'] == 'Low Risk'][:2]

print(f"\nHigh Risk Examples:")
for i, person in enumerate(high_risk_sample, 1):
    print(f"  {i}. Risk: {person['calculated_risk_score']}, Resilience: {person['calculated_resilience_score']}")
    print(f"     Income: ${float(person['income']):,.0f}, Debt: ${float(person['debt_amount']):,.0f}")
    print(f"     Employment: {['Unemployed', 'Part-time', 'Full-time', 'Self-employed'][int(person['employment_status'])]}")
    print()

print(f"Low Risk Examples:")
for i, person in enumerate(low_risk_sample, 1):
    print(f"  {i}. Risk: {person['calculated_risk_score']}, Resilience: {person['calculated_resilience_score']}")
    print(f"     Income: ${float(person['income']):,.0f}, Debt: ${float(person['debt_amount']):,.0f}")
    print(f"     Savings: ${float(person['savings_amount']):,.0f}")
    print(f"     Employment: {['Unemployed', 'Part-time', 'Full-time', 'Self-employed'][int(person['employment_status'])]}")
    print()