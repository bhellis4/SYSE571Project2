import csv
import os
import random
import numpy as np
from collections import Counter
import re

class CSVParser:
    training_data_size = 1000
    testing_data_size = 1000
    verification_data_size = 1000
    production_data_size = 2000

    def __init__(self, filename):
        """
        Initialize the CSV parser with a filename.
        
        Args:
            filename (str): Path to the CSV file
        """
        self.filename = filename
        self.headers = []
        self.data = []
        
    def parse_csv(self):
        """
        Parse the CSV file and store headers and data.
        
        Returns:
            tuple: (h              print("Each dataset includes:")
        print("- All original features (quantified/encoded)")
        print("- calculated_risk_score - Independent risk factors (0-100)")
        print("- calculated_resilience_score - Independent resilience factors (0-100)")
        print("- net_risk_score - Risk minus Resilience differential")
        print("- risk_category - High/Medium/Low based on net risk")
        print("- debt_to_income_ratio - Financial stress indicator")nt("Each dataset includes:")
        print("- All original features (quantified/encoded)")
        print("- calculated_risk_score - Independent risk factors (0-100)")
        print("- calculated_resilience_score - Independent resilience factors (0-100)")
        print("- net_risk_score - Risk minus Resilience differential")
        print("- risk_category - High/Medium/Low based on net risk")
        print("- debt_to_income_ratio - Financial stress indicator")s, data) where headers is a list of column names
                   and data is a list of dictionaries representing each row
        """
        try:
            with open(self.filename, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                
                # Get headers from first row
                self.headers = next(csv_reader)
                
                # Parse each data row
                for row in csv_reader:
                    if len(row) == len(self.headers):  # Ensure row has correct number of columns
                        row_dict = {}
                        for i, value in enumerate(row):
                            row_dict[self.headers[i]] = value.strip()  # Remove whitespace
                        self.data.append(row_dict)
                
                return self.headers, self.data
                
        except FileNotFoundError:
            print(f"Error: File '{self.filename}' not found.")
            return [], []
        except Exception as e:
            print(f"Error parsing CSV: {e}")
            return [], []
    
    def get_column_values(self, column_name):
        """
        Get all values from a specific column.
        
        Args:
            column_name (str): Name of the column
            
        Returns:
            list: All values in the specified column
        """
        if not self.data:
            self.parse_csv()
            
        if column_name not in self.headers:
            print(f"Error: Column '{column_name}' not found.")
            return []
            
        return [row[column_name] for row in self.data]
    
    def get_row(self, row_index):
        """
        Get a specific row by index.
        
        Args:
            row_index (int): Index of the row (0-based)
            
        Returns:
            dict: Dictionary containing the row data
        """
        if not self.data:
            self.parse_csv()
            
        if 0 <= row_index < len(self.data):
            return self.data[row_index]
        else:
            print(f"Error: Row index {row_index} is out of range.")
            return {}
    
    def get_value(self, row_index, column_name):
        """
        Get a specific value by row index and column name.
        
        Args:
            row_index (int): Index of the row (0-based)
            column_name (str): Name of the column
            
        Returns:
            str: The value at the specified position
        """
        if not self.data:
            self.parse_csv()
            
        if 0 <= row_index < len(self.data) and column_name in self.headers:
            return self.data[row_index][column_name]
        else:
            print(f"Error: Invalid row index {row_index} or column '{column_name}'.")
            return ""
    
    def display_headers(self):
        """Display all column headers."""
        if not self.headers:
            self.parse_csv()
        print("Column Headers:")
        for i, header in enumerate(self.headers):
            print(f"  {i}: {header}")
    
    def display_sample_data(self, num_rows=5):
        """
        Display sample data from the CSV.
        
        Args:
            num_rows (int): Number of rows to display (default: 5)
        """
        if not self.data:
            self.parse_csv()
            
        print(f"\nSample Data (first {min(num_rows, len(self.data))} rows):")
        print("-" * 80)
        
        for i in range(min(num_rows, len(self.data))):
            print(f"Row {i}:")
            for header in self.headers:
                print(f"  {header}: {self.data[i][header]}")
            print()
    
    def get_stats(self):
        """Get basic statistics about the CSV file."""
        if not self.data:
            self.parse_csv()
            
        print(f"CSV Statistics:")
        print(f"  File: {self.filename}")
        print(f"  Number of columns: {len(self.headers)}")
        print(f"  Number of rows: {len(self.data)}")
        print(f"  Total data points: {len(self.headers) * len(self.data)}")

    def split_data_for_ml(self, shuffle=True, seed=42):
        """
        Split data into training, testing, verification, and production sets.
        
        Args:
            shuffle (bool): Whether to shuffle data before splitting
            seed (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing the split datasets
        """
        if not self.data:
            self.parse_csv()
        
        if shuffle:
            random.seed(seed)
            data_copy = self.data.copy()
            random.shuffle(data_copy)
        else:
            data_copy = self.data
        
        total_needed = (self.training_data_size + self.testing_data_size + 
                       self.verification_data_size + self.production_data_size)
        
        if len(data_copy) < total_needed:
            print(f"Warning: Not enough data. Need {total_needed}, have {len(data_copy)}")
            # Adjust sizes proportionally
            ratio = len(data_copy) / total_needed
            self.training_data_size = int(self.training_data_size * ratio)
            self.testing_data_size = int(self.testing_data_size * ratio)
            self.verification_data_size = int(self.verification_data_size * ratio)
            self.production_data_size = len(data_copy) - (self.training_data_size + 
                                                         self.testing_data_size + 
                                                         self.verification_data_size)
        
        # Split the data
        start_idx = 0
        splits = {}
        
        splits['training'] = data_copy[start_idx:start_idx + self.training_data_size]
        start_idx += self.training_data_size
        
        splits['testing'] = data_copy[start_idx:start_idx + self.testing_data_size]
        start_idx += self.testing_data_size
        
        splits['verification'] = data_copy[start_idx:start_idx + self.verification_data_size]
        start_idx += self.verification_data_size
        
        splits['production'] = data_copy[start_idx:start_idx + self.production_data_size]
        
        print(f"Data split successfully:")
        for split_name, split_data in splits.items():
            print(f"  {split_name.capitalize()}: {len(split_data)} rows")
        
        return splits

    def quantify_features(self, data_subset=None):
        """
        Quantify/encode categorical features for ML algorithms.
        
        Args:
            data_subset (list): Subset of data to quantify (defaults to all data)
            
        Returns:
            list: List of dictionaries with quantified features
        """
        if not self.data:
            self.parse_csv()
        
        if data_subset is None:
            data_subset = self.data
        
        quantified_data = []
        
        # Define encoding mappings
        encodings = {
            'gender': self._create_label_encoding('gender'),
            'education_level': {
                'High School Diploma': 1,
                "Bachelor's Degree": 2, 
                "Master's Degree": 3,
                'PhD': 4
            },
            'marital_status': self._create_label_encoding('marital_status'),
            'mental_health_condition': self._create_label_encoding('mental_health_condition'),
            'employment_status': {
                'Unemployed': 0,
                'Employed Part-Time': 1,
                'Employed Full-Time': 2,
                'Self-Employed': 3
            },
            'physical_activity_level': {
                'Sedentary': 0,
                'Lightly Active': 1,
                'Moderately Active': 2,
                'Very Active': 3
            },
            'sleep_quality': {
                'Poor': 0,
                'Fair': 1,
                'Good': 2,
                'Excellent': 3
            },
            'risk_tolerance': {
                'Low': 0,
                'Medium': 1,
                'High': 2
            },
            'financial_goals': self._create_label_encoding('financial_goals')
        }
        
        for row in data_subset:
            quantified_row = {}
            
            for column, value in row.items():
                if column in encodings:
                    # Apply encoding
                    quantified_row[column] = encodings[column].get(value, 0)
                elif column in ['age', 'family_size', 'credit_score', 'social_support_network', 
                               'stress_level', 'resilience_score']:
                    # Already numeric
                    quantified_row[column] = int(value) if value.isdigit() else 0
                elif column in ['income', 'savings_amount', 'debt_amount']:
                    # Extract numeric value from currency format
                    numeric_value = re.sub(r'[^\d.]', '', value)
                    quantified_row[column] = float(numeric_value) if numeric_value else 0.0
                elif column in ['insurance_coverage', 'access_to_healthcare']:
                    # Boolean values
                    quantified_row[column] = 1 if value.lower() == 'true' else 0
                else:
                    # Default: try to convert to number, otherwise 0
                    try:
                        quantified_row[column] = float(value)
                    except:
                        quantified_row[column] = 0
            
            quantified_data.append(quantified_row)
        
        return quantified_data
    
    def _create_label_encoding(self, column_name):
        """Create a label encoding mapping for a categorical column."""
        if not self.data:
            self.parse_csv()
        
        unique_values = list(set(row[column_name] for row in self.data))
        return {value: idx for idx, value in enumerate(unique_values)}
    
    def _get_gender_risk_mapping(self):
        """
        Create gender-based risk mapping based on societal discrimination and systemic disadvantages.
        Non-conforming > Female > Male in terms of societal risk factors.
        """
        # First get the gender encoding to map numeric codes to categories
        gender_encoding = self._create_label_encoding('gender')
        
        # Reverse mapping to get category from numeric code
        code_to_gender = {v: k for k, v in gender_encoding.items()}
        
        # Risk points based on societal discrimination and systemic challenges
        gender_risk = {}
        for code, gender in code_to_gender.items():
            if gender in ['Male']:
                gender_risk[code] = 0  # Baseline - societal privilege
            elif gender in ['Female']:
                gender_risk[code] = 4  # Gender pay gap, discrimination
            else:  # Non-binary, Genderfluid, Agender, Bigender, Genderqueer, Polygender
                gender_risk[code] = 8  # Significant discrimination and systemic challenges
        
        return gender_risk
    
    def calculate_risk_resilience_scores(self, data_subset=None):
        """
        Calculate SEPARATE risk and resilience scores, ignoring arbitrary original resilience_score.
        Risk factors vs Resilience factors are balanced against each other.
        Final risk category determined by Risk Score - Resilience Score differential.
        
        Args:
            data_subset (list): Subset of data to analyze
            
        Returns:
            list: Data with added independent risk and resilience scores
        """
        quantified_data = self.quantify_features(data_subset)
        
        for row in quantified_data:
            # === RISK SCORE CALCULATION (0-100) ===
            risk_score = 0
            
            # Employment Risk (0-25 points)
            if row['employment_status'] == 0:  # Unemployed
                risk_score += 25
            elif row['employment_status'] == 1:  # Part-time only
                risk_score += 12
            
            # Financial Risk - Debt burden (0-25 points) 
            if row['debt_amount'] > 200000:  # Very high debt
                risk_score += 25
            elif row['debt_amount'] > 150000:  # High debt
                risk_score += 20
            elif row['debt_amount'] > 100000:  # Moderate debt
                risk_score += 15
            elif row['debt_amount'] > 50000:  # Some debt
                risk_score += 8
            
            # Age-based Risk (0-15 points) - Younger = higher risk, older = lower risk
            age = row['age']
            if age < 25:  # Young adults - higher risk tolerance, less stability
                risk_score += 15
            elif age < 35:  # Early career - moderate risk
                risk_score += 10
            elif age < 50:  # Mid-career - established, lower risk
                risk_score += 5
            elif age < 65:  # Pre-retirement - stable, minimal risk
                risk_score += 2
            # 65+ gets 0 additional risk points - age brings wisdom and stability
            
            # Gender-based Social Risk (0-12 points) - Non-conforming > Female > Male
            # Based on societal discrimination and systemic disadvantages
            gender_map = self._get_gender_risk_mapping()
            if row['gender'] in gender_map:
                risk_score += gender_map[row['gender']]
            
            # Marital Status Risk (0-10 points) - Single=Married < Divorced < Widowed
            marital_risk = {
                0: 5,   # Divorced - financial/emotional stress
                1: 0,   # Single - baseline
                2: 8,   # Widowed - highest stress, potential financial loss
                3: 0    # Married - baseline (has support)
            }
            if row['marital_status'] in marital_risk:
                risk_score += marital_risk[row['marital_status']]
            
            # Family Size Risk with Marital Context (0-8 points)
            family_size = row['family_size']
            is_married = (row['marital_status'] == 3)  # Married = 2 potential incomes
            
            if family_size == 1:  # Living alone
                risk_score += 4  # Isolation risk
            elif family_size >= 8:  # Very large family
                if is_married:
                    risk_score += 3  # Reduced burden with dual income
                else:
                    risk_score += 8  # High burden for single parent/provider
            
            # Credit Score Risk (WEIGHTED HEAVILY - 0-20 points)
            # This is a major financial health indicator - weighted more heavily
            credit_score = row['credit_score']
            if credit_score < 400:  # Very poor credit
                risk_score += 20
            elif credit_score < 500:  # Poor credit
                risk_score += 16
            elif credit_score < 600:  # Fair credit
                risk_score += 12
            elif credit_score < 700:  # Good credit
                risk_score += 6
            elif credit_score < 750:  # Very good credit
                risk_score += 3
            # 750+ excellent credit gets 0 risk points
            
            # Healthcare Access Risk (0-15 points)
            if row['insurance_coverage'] == 0:  # No insurance
                risk_score += 9
            if row['access_to_healthcare'] == 0:  # No healthcare access
                risk_score += 6
            
            # Mental Health & Stress Risk (0-15 points)
            if row['mental_health_condition'] > 0:  # Has mental health condition
                risk_score += 8
            if row['stress_level'] >= 8:  # High stress (8-10)
                risk_score += 7
            elif row['stress_level'] >= 6:  # Moderate-high stress (6-7)
                risk_score += 4
            
            # Social & Sleep Risk (0-8 points)
            if row['social_support_network'] <= 2:  # Very low support
                risk_score += 4
            if row['sleep_quality'] == 0:  # Poor sleep
                risk_score += 4
            
            # Cap risk at 100
            risk_score = min(100, max(0, risk_score))
            
            # === RESILIENCE SCORE CALCULATION (0-100) ===
            resilience_score = 0
            
            # Credit Score Resilience (WEIGHTED HEAVILY - 0-25 points)
            # Excellent credit indicates strong financial management and stability
            credit_score = row['credit_score']
            if credit_score >= 800:  # Exceptional credit
                resilience_score += 25
            elif credit_score >= 750:  # Excellent credit
                resilience_score += 20
            elif credit_score >= 700:  # Good credit
                resilience_score += 15
            elif credit_score >= 650:  # Fair credit
                resilience_score += 10
            elif credit_score >= 600:  # Poor credit
                resilience_score += 5
            # Below 600 gets 0 resilience points
            
            # Financial Resilience (0-25 points) - Income vs Debt relationship
            debt_to_income_ratio = row['debt_amount'] / (row['income'] + 1)  # Avoid division by zero
            
            # High income provides resilience
            if row['income'] > 200000:  # High income
                resilience_score += 18
            elif row['income'] > 150000:  # Good income
                resilience_score += 14
            elif row['income'] > 100000:  # Decent income
                resilience_score += 10
            elif row['income'] > 50000:  # Moderate income
                resilience_score += 5
            
            # Debt-to-income ratio resilience bonus
            if debt_to_income_ratio < 0.2:  # Very low debt relative to income
                resilience_score += 7
            elif debt_to_income_ratio < 0.4:  # Low debt relative to income
                resilience_score += 5
            elif debt_to_income_ratio < 0.6:  # Moderate debt ratio
                resilience_score += 2
            
            # Savings provide major resilience buffer (0-12 points)
            if row['savings_amount'] > 200000:  # Substantial savings
                resilience_score += 12
            elif row['savings_amount'] > 100000:  # Good savings
                resilience_score += 10
            elif row['savings_amount'] > 50000:  # Some savings
                resilience_score += 7
            elif row['savings_amount'] > 20000:  # Emergency fund
                resilience_score += 4
            elif row['savings_amount'] > 5000:  # Small buffer
                resilience_score += 2
            
            # Employment Stability & Growth (0-12 points)
            if row['employment_status'] == 2:  # Full-time employed
                resilience_score += 12
            elif row['employment_status'] == 3:  # Self-employed (autonomy)
                resilience_score += 10
            elif row['employment_status'] == 1:  # Part-time (some stability)
                resilience_score += 5
            
            # Age-based Resilience (0-8 points) - Older = more resilience (wisdom, stability)
            age = row['age']
            if age >= 65:  # Senior - wisdom and life experience
                resilience_score += 8
            elif age >= 50:  # Mature adult - established and stable
                resilience_score += 6
            elif age >= 35:  # Mid-career - some stability
                resilience_score += 4
            elif age >= 25:  # Young adult - minimal experience bonus
                resilience_score += 2
            # Under 25 gets 0 age-based resilience
            
            # Marital/Family Support Resilience (0-8 points)
            is_married = (row['marital_status'] == 3)
            family_size = row['family_size']
            
            if is_married:  # Marriage provides support and dual income potential
                resilience_score += 5
                if family_size >= 3:  # Married with children - family support network
                    resilience_score += 3
            elif family_size >= 3:  # Single parent with family - some support but less
                resilience_score += 2
            
            # Education as resilience foundation (0-12 points)
            if row['education_level'] == 4:  # PhD - highest adaptability
                resilience_score += 12
            elif row['education_level'] == 3:  # Master's - strong skills
                resilience_score += 9
            elif row['education_level'] == 2:  # Bachelor's - good foundation
                resilience_score += 6
            elif row['education_level'] == 1:  # High school - basic foundation
                resilience_score += 3
            
            # Health & Lifestyle Resilience (0-15 points)
            if row['physical_activity_level'] == 3:  # Very active
                resilience_score += 6
            elif row['physical_activity_level'] == 2:  # Moderately active
                resilience_score += 4
            elif row['physical_activity_level'] == 1:  # Lightly active
                resilience_score += 2
            
            if row['sleep_quality'] == 3:  # Excellent sleep
                resilience_score += 5
            elif row['sleep_quality'] == 2:  # Good sleep
                resilience_score += 3
            elif row['sleep_quality'] == 1:  # Fair sleep
                resilience_score += 1
            
            # Healthcare access adds resilience
            if row['insurance_coverage'] == 1 and row['access_to_healthcare'] == 1:
                resilience_score += 4
            
            # Social Support Network (0-8 points)
            if row['social_support_network'] >= 8:  # Strong network
                resilience_score += 8
            elif row['social_support_network'] >= 6:  # Good network
                resilience_score += 6
            elif row['social_support_network'] >= 4:  # Moderate network
                resilience_score += 4
            elif row['social_support_network'] >= 2:  # Some network
                resilience_score += 2
            
            # Financial Goals indicate planning/future orientation (0-5 points)
            if row['financial_goals'] > 0:  # Has financial goals
                resilience_score += 5
            
            # Risk Tolerance can indicate confidence (0-5 points)
            if row['risk_tolerance'] == 2:  # High risk tolerance
                resilience_score += 5
            elif row['risk_tolerance'] == 1:  # Medium risk tolerance
                resilience_score += 3
            
            # Cap resilience at 100
            resilience_score = min(100, max(0, resilience_score))
            
            # === FINAL RISK CATEGORIZATION ===
            # Based on Risk Score - Resilience Score differential
            net_risk = risk_score - resilience_score
            
            # Determine risk category based on net risk
            if net_risk > 20:
                risk_category = 'High Risk'
            elif net_risk > -10:
                risk_category = 'Medium Risk'  
            else:
                risk_category = 'Low Risk'
            
            # Add all calculated scores
            row['calculated_risk_score'] = round(risk_score, 1)
            row['calculated_resilience_score'] = round(resilience_score, 1)
            row['net_risk_score'] = round(net_risk, 1)
            row['risk_category'] = risk_category
            row['risk_resilience_ratio'] = round(risk_score / (resilience_score + 1), 2)
            row['debt_to_income_ratio'] = round(debt_to_income_ratio, 2)
        
        return quantified_data
    
    def analyze_risk_patterns(self, data_subset=None):
        """
        Perform comprehensive risk and resilience analysis.
        
        Args:
            data_subset (list): Data subset to analyze
        """
        analyzed_data = self.calculate_risk_resilience_scores(data_subset)
        
        print("\n" + "="*60)
        print("RISK & RESILIENCE ANALYSIS")
        print("="*60)
        
        # Extract scores for analysis
        risk_scores = [row['calculated_risk_score'] for row in analyzed_data]
        resilience_scores = [row['calculated_resilience_score'] for row in analyzed_data]
        net_risk_scores = [row['net_risk_score'] for row in analyzed_data]
        
        print(f"\nRisk Score Statistics (0-100):")
        print(f"  Mean: {np.mean(risk_scores):.2f}")
        print(f"  Median: {np.median(risk_scores):.2f}")
        print(f"  Std Dev: {np.std(risk_scores):.2f}")
        print(f"  Min: {min(risk_scores):.2f}")
        print(f"  Max: {max(risk_scores):.2f}")
        
        print(f"\nResilience Score Statistics (0-100):")
        print(f"  Mean: {np.mean(resilience_scores):.2f}")
        print(f"  Median: {np.median(resilience_scores):.2f}")
        print(f"  Std Dev: {np.std(resilience_scores):.2f}")
        print(f"  Min: {min(resilience_scores):.2f}")
        print(f"  Max: {max(resilience_scores):.2f}")
        
        print(f"\nNet Risk Score (Risk - Resilience):")
        print(f"  Mean: {np.mean(net_risk_scores):.2f}")
        print(f"  Median: {np.median(net_risk_scores):.2f}")
        print(f"  Std Dev: {np.std(net_risk_scores):.2f}")
        print(f"  Min: {min(net_risk_scores):.2f}")
        print(f"  Max: {max(net_risk_scores):.2f}")
        
        # Risk categories based on net risk (risk - resilience)
        high_risk = [r for r in analyzed_data if r['risk_category'] == 'High Risk']
        medium_risk = [r for r in analyzed_data if r['risk_category'] == 'Medium Risk']
        low_risk = [r for r in analyzed_data if r['risk_category'] == 'Low Risk']
        
        print(f"\nRisk Distribution (Based on Net Risk Score):")
        print(f"  High Risk (Net > 20): {len(high_risk)} ({len(high_risk)/len(analyzed_data)*100:.1f}%)")
        print(f"  Medium Risk (Net -10 to 20): {len(medium_risk)} ({len(medium_risk)/len(analyzed_data)*100:.1f}%)")
        print(f"  Low Risk (Net < -10): {len(low_risk)} ({len(low_risk)/len(analyzed_data)*100:.1f}%)")
        
        # Analysis of high-risk group
        print(f"\nHigh Risk Group Analysis:")
        if high_risk:
            avg_risk = np.mean([r['calculated_risk_score'] for r in high_risk])
            avg_resilience = np.mean([r['calculated_resilience_score'] for r in high_risk])
            avg_income = np.mean([r['income'] for r in high_risk])
            avg_debt = np.mean([r['debt_amount'] for r in high_risk])
            
            print(f"  Average Risk Score: {avg_risk:.1f}")
            print(f"  Average Resilience Score: {avg_resilience:.1f}")
            print(f"  Average Income: ${avg_income:,.0f}")
            print(f"  Average Debt: ${avg_debt:,.0f}")
            
            unemployment = sum(1 for r in high_risk if r['employment_status'] == 0)
            high_debt_count = sum(1 for r in high_risk if r['debt_amount'] > 100000)
            no_savings = sum(1 for r in high_risk if r['savings_amount'] < 10000)
            
            print(f"  Unemployed: {unemployment}/{len(high_risk)} ({unemployment/len(high_risk)*100:.1f}%)")
            print(f"  High Debt (>$100K): {high_debt_count}/{len(high_risk)} ({high_debt_count/len(high_risk)*100:.1f}%)")
            print(f"  Low Savings (<$10K): {no_savings}/{len(high_risk)} ({no_savings/len(high_risk)*100:.1f}%)")
        
        # Analysis of low-risk group
        print(f"\nLow Risk Group Analysis:")
        if low_risk:
            avg_risk = np.mean([r['calculated_risk_score'] for r in low_risk])
            avg_resilience = np.mean([r['calculated_resilience_score'] for r in low_risk])
            avg_income = np.mean([r['income'] for r in low_risk])
            avg_debt = np.mean([r['debt_amount'] for r in low_risk])
            
            print(f"  Average Risk Score: {avg_risk:.1f}")
            print(f"  Average Resilience Score: {avg_resilience:.1f}")
            print(f"  Average Income: ${avg_income:,.0f}")
            print(f"  Average Debt: ${avg_debt:,.0f}")
            
            full_time = sum(1 for r in low_risk if r['employment_status'] == 2)
            high_savings = sum(1 for r in low_risk if r['savings_amount'] > 100000)
            education_high = sum(1 for r in low_risk if r['education_level'] >= 3)
            
            print(f"  Full-time Employed: {full_time}/{len(low_risk)} ({full_time/len(low_risk)*100:.1f}%)")
            print(f"  High Savings (>$100K): {high_savings}/{len(low_risk)} ({high_savings/len(low_risk)*100:.1f}%)")
            print(f"  Advanced Education: {education_high}/{len(low_risk)} ({education_high/len(low_risk)*100:.1f}%)")
        
        return analyzed_data
    
    def export_ml_dataset(self, splits, output_dir="ml_datasets"):
        """
        Export quantified and analyzed datasets for ML training.
        
        Args:
            splits (dict): Data splits from split_data_for_ml()
            output_dir (str): Directory to save the datasets
        """
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for split_name, split_data in splits.items():
            # Quantify and analyze the data
            analyzed_data = self.calculate_risk_resilience_scores(split_data)
            
            # Save as CSV
            output_file = os.path.join(output_dir, f"{split_name}_dataset.csv")
            
            if analyzed_data:
                fieldnames = analyzed_data[0].keys()
                
                with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(analyzed_data)
                
                print(f"Exported {len(analyzed_data)} rows to {output_file}")


def main():
    """ML-focused example usage of the CSV parser for risk/resilience analysis."""
    # Initialize parser with your CSV file
    parser = CSVParser('MOCK_DATA_FINAL.csv')
    
    # Parse the CSV file
    headers, data = parser.parse_csv()
    
    if headers and data:
        print("CSV parsed successfully for ML Risk/Resilience Analysis!")
        
        # Display basic statistics
        parser.get_stats()
        
        # Split data for ML workflow
        print("\n" + "="*50)
        print("SPLITTING DATA FOR ML PIPELINE")
        print("="*50)
        splits = parser.split_data_for_ml(shuffle=True, seed=42)
        
        # Analyze risk patterns on training data
        print("\nAnalyzing risk patterns on training data...")
        training_analysis = parser.analyze_risk_patterns(splits['training'])
        
        # Show some quantified examples
        print("\n" + "="*50)
        print("QUANTIFIED DATA EXAMPLES (First 3 Training Records)")
        print("="*50)
        
        for i, record in enumerate(training_analysis[:3]):
            print(f"\nRecord {i+1}:")
            print(f"  Age: {record['age']}, Employment: {['Unemployed', 'Part-time', 'Full-time', 'Self-employed'][record['employment_status']]}")
            print(f"  Income: ${record['income']:,.2f}, Debt: ${record['debt_amount']:,.2f}")
            print(f"  Debt/Income Ratio: {record['debt_to_income_ratio']}")
            print(f"  Savings: ${record['savings_amount']:,.2f}")
            print(f"  Risk Score: {record['calculated_risk_score']}")
            print(f"  Resilience Score: {record['calculated_resilience_score']}")
            print(f"  Net Risk: {record['net_risk_score']} → {record['risk_category']}")
        
        # Export datasets for ML training
        print("\n" + "="*50)
        print("EXPORTING ML DATASETS")
        print("="*50)
        parser.export_ml_dataset(splits)
        
        print("\n" + "="*50)
        print("ML PIPELINE READY!")
        print("="*50)
        print("Your datasets are now ready for machine learning:")
        print("1. training_dataset.csv - For model training")
        print("2. testing_dataset.csv - For model evaluation") 
        print("3. verification_dataset.csv - For model validation")
        print("4. production_dataset.csv - For production inference")
        print("\nEach dataset includes:")
        print("- All original features (quantified/encoded)")
        print("- calculated_risk_score - Composite risk metric")
        print("- enhanced_resilience_score - Enhanced resilience metric") 
        print("- risk_resilience_ratio - Risk to resilience ratio")
        print("\nReady for algorithms like:")
        print("- Random Forest, XGBoost for prediction")
        print("- K-means clustering for risk segmentation")
        print("- Neural networks for complex pattern detection")


if __name__ == "__main__":
    main()