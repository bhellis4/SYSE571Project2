import csv

# Load original data to examine all fields
with open('MOCK_DATA_FINAL.csv', 'r') as f:
    reader = csv.DictReader(f)
    original_data = list(reader)

# Load quantified data to see what we're tracking
with open('ml_datasets/training_dataset.csv', 'r') as f:
    reader = csv.DictReader(f)
    quantified_data = list(reader)

print("COMPREHENSIVE FIELD QUANTIFICATION REVIEW")
print("="*60)

# Get all original fields
original_fields = list(original_data[0].keys())
quantified_fields = list(quantified_data[0].keys())

print(f"Original CSV has {len(original_fields)} fields:")
for i, field in enumerate(original_fields, 1):
    print(f"  {i:2d}. {field}")

print(f"\nQuantified CSV has {len(quantified_fields)} fields:")
for i, field in enumerate(quantified_fields, 1):
    print(f"  {i:2d}. {field}")

print(f"\nFIELD-BY-FIELD QUANTIFICATION STATUS:")
print("-"*60)

# Analyze each original field
for field in original_fields:
    if field in quantified_fields:
        # Sample original vs quantified values
        original_sample = [row[field] for row in original_data[:5]]
        quantified_sample = [row[field] for row in quantified_data[:5]]
        
        print(f"\n{field}:")
        print(f"  ✓ QUANTIFIED")
        print(f"  Original samples: {original_sample}")
        print(f"  Quantified samples: {quantified_sample}")
        
        # Check if quantification looks correct
        if field in ['age', 'family_size', 'credit_score', 'social_support_network', 'stress_level']:
            if all(str(v).isdigit() for v in quantified_sample if v):
                print(f"  Status: ✓ Correctly preserved as numeric")
            else:
                print(f"  Status: ⚠️ May have conversion issues")
        
        elif field in ['income', 'savings_amount', 'debt_amount']:
            orig_has_currency = any('$' in str(v) for v in original_sample)
            quant_is_numeric = all(str(v).replace('.','').isdigit() for v in quantified_sample if v)
            if orig_has_currency and quant_is_numeric:
                print(f"  Status: ✓ Currency correctly converted to numeric")
            else:
                print(f"  Status: ⚠️ Currency conversion may have issues")
        
        elif field in ['insurance_coverage', 'access_to_healthcare']:
            orig_has_bool = any(v in ['true', 'false'] for v in original_sample)
            quant_is_binary = all(v in ['0', '1'] for v in quantified_sample if v)
            if orig_has_bool and quant_is_binary:
                print(f"  Status: ✓ Boolean correctly converted to 0/1")
            else:
                print(f"  Status: ⚠️ Boolean conversion may have issues")
        
        elif field in ['gender', 'education_level', 'marital_status', 'mental_health_condition', 
                      'employment_status', 'physical_activity_level', 'sleep_quality', 
                      'risk_tolerance', 'financial_goals']:
            orig_is_text = any(isinstance(v, str) and not v.isdigit() for v in original_sample)
            quant_is_numeric = all(str(v).isdigit() for v in quantified_sample if v)
            if orig_is_text and quant_is_numeric:
                print(f"  Status: ✓ Categorical correctly encoded to numeric")
            else:
                print(f"  Status: ⚠️ Categorical encoding may have issues")
    else:
        print(f"\n{field}:")
        print(f"  ❌ NOT QUANTIFIED - Missing from output!")

# Check for new fields we added
new_fields = [f for f in quantified_fields if f not in original_fields]
if new_fields:
    print(f"\nNEW FIELDS ADDED:")
    print("-"*30)
    for field in new_fields:
        print(f"  + {field}")

# Check for specific issues with our current system
print(f"\nPOTENTIAL QUANTIFICATION ISSUES:")
print("-"*40)

# Check credit_score range
credit_scores = [int(row['credit_score']) for row in quantified_data if row['credit_score']]
print(f"Credit Score Range: {min(credit_scores)} - {max(credit_scores)}")
if min(credit_scores) < 300 or max(credit_scores) > 850:
    print(f"  ⚠️ Credit scores outside typical range (300-850)")
else:
    print(f"  ✓ Credit scores in reasonable range")

# Check family size
family_sizes = [int(row['family_size']) for row in quantified_data if row['family_size']]
print(f"Family Size Range: {min(family_sizes)} - {max(family_sizes)}")
if max(family_sizes) > 15:
    print(f"  ⚠️ Some family sizes seem unusually large")
else:
    print(f"  ✓ Family sizes seem reasonable")

# Check for any obvious encoding issues
print(f"\nENCODING VALIDATION:")
print("-"*20)

# Gender encoding check
genders = set(row['gender'] for row in quantified_data)
print(f"Gender encodings: {genders}")

# Education level check  
edu_levels = set(row['education_level'] for row in quantified_data)
print(f"Education encodings: {edu_levels}")
if edu_levels == {'1', '2', '3', '4'}:
    print(f"  ✓ Education properly encoded 1-4")
else:
    print(f"  ⚠️ Education encoding may have issues")

# Employment status check
emp_status = set(row['employment_status'] for row in quantified_data)
print(f"Employment encodings: {emp_status}")
if emp_status == {'0', '1', '2', '3'}:
    print(f"  ✓ Employment properly encoded 0-3")
else:
    print(f"  ⚠️ Employment encoding may have issues")