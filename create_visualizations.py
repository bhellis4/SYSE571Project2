import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os

# Set up publication-quality plot style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

class RiskResilienceVisualizer:
    def __init__(self, data_file):
        self.df = pd.read_csv(data_file)
        self.output_dir = 'white_paper_graphics'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create mappings for better labels
        self.gender_map = {
            0: 'Agender', 1: 'Bigender', 2: 'Female', 3: 'Genderfluid',
            4: 'Genderqueer', 5: 'Male', 6: 'Non-binary', 7: 'Polygender'
        }
        self.marital_map = {0: 'Divorced', 1: 'Single', 2: 'Widowed', 3: 'Married'}
        self.employment_map = {0: 'Unemployed', 1: 'Part-time', 2: 'Full-time', 3: 'Self-employed'}
        
        # Apply mappings
        self.df['gender_label'] = self.df['gender'].map(self.gender_map)
        self.df['marital_label'] = self.df['marital_status'].map(self.marital_map)
        self.df['employment_label'] = self.df['employment_status'].map(self.employment_map)
        
        print(f"Loaded dataset with {len(self.df)} records")
        print(f"Graphics will be saved to: {self.output_dir}/")
    
    def plot_risk_distribution(self):
        """Figure 1: Overall Risk Distribution"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Risk category distribution
        risk_counts = self.df['risk_category'].value_counts()
        colors = ['#FF6B6B', '#FFE66D', '#4ECDC4']  # Red, Yellow, Teal
        ax1.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax1.set_title('Population Risk Distribution', fontweight='bold')
        
        # Risk score histogram
        ax2.hist(self.df['calculated_risk_score'], bins=30, alpha=0.7, color='#FF6B6B', edgecolor='black')
        ax2.set_xlabel('Risk Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Risk Score Distribution', fontweight='bold')
        ax2.axvline(self.df['calculated_risk_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["calculated_risk_score"].mean():.1f}')
        ax2.legend()
        
        # Resilience score histogram
        ax3.hist(self.df['calculated_resilience_score'], bins=30, alpha=0.7, color='#4ECDC4', edgecolor='black')
        ax3.set_xlabel('Resilience Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Resilience Score Distribution', fontweight='bold')
        ax3.axvline(self.df['calculated_resilience_score'].mean(), color='teal', linestyle='--',
                   label=f'Mean: {self.df["calculated_resilience_score"].mean():.1f}')
        ax3.legend()
        
        # Net risk distribution
        ax4.hist(self.df['net_risk_score'], bins=30, alpha=0.7, color='#FFE66D', edgecolor='black')
        ax4.set_xlabel('Net Risk Score (Risk - Resilience)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Net Risk Score Distribution', fontweight='bold')
        ax4.axvline(0, color='red', linestyle='-', alpha=0.8, label='Balance Point')
        ax4.axvline(self.df['net_risk_score'].mean(), color='orange', linestyle='--',
                   label=f'Mean: {self.df["net_risk_score"].mean():.1f}')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_risk_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/01_risk_distribution.pdf', bbox_inches='tight')
        print("✓ Generated Figure 1: Risk Distribution")
        plt.close()
    
    def plot_demographic_analysis(self):
        """Figure 2: Demographic Risk Patterns"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Age group analysis
        age_bins = [0, 25, 35, 50, 65, 100]
        age_labels = ['18-24', '25-34', '35-49', '50-64', '65+']
        self.df['age_group'] = pd.cut(self.df['age'], bins=age_bins, labels=age_labels, right=False)
        
        age_risk = self.df.groupby('age_group')[['calculated_risk_score', 'calculated_resilience_score']].mean()
        age_risk.plot(kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4'])
        ax1.set_title('Risk & Resilience by Age Group', fontweight='bold')
        ax1.set_xlabel('Age Group')
        ax1.set_ylabel('Average Score')
        ax1.legend(['Risk Score', 'Resilience Score'])
        ax1.tick_params(axis='x', rotation=45)
        
        # Gender analysis (show top categories)
        gender_risk = self.df.groupby('gender_label')[['calculated_risk_score', 'calculated_resilience_score']].mean()
        gender_counts = self.df['gender_label'].value_counts()
        major_genders = gender_counts.head(4).index  # Show top 4 for readability
        
        gender_risk_subset = gender_risk.loc[major_genders]
        gender_risk_subset.plot(kind='bar', ax=ax2, color=['#FF6B6B', '#4ECDC4'])
        ax2.set_title('Risk & Resilience by Gender\n(Top 4 Categories)', fontweight='bold')
        ax2.set_xlabel('Gender Identity')
        ax2.set_ylabel('Average Score')
        ax2.legend(['Risk Score', 'Resilience Score'])
        ax2.tick_params(axis='x', rotation=45)
        
        # Marital status analysis
        marital_risk = self.df.groupby('marital_label')[['calculated_risk_score', 'calculated_resilience_score']].mean()
        marital_risk.plot(kind='bar', ax=ax3, color=['#FF6B6B', '#4ECDC4'])
        ax3.set_title('Risk & Resilience by Marital Status', fontweight='bold')
        ax3.set_xlabel('Marital Status')
        ax3.set_ylabel('Average Score')
        ax3.legend(['Risk Score', 'Resilience Score'])
        ax3.tick_params(axis='x', rotation=45)
        
        # Employment status analysis
        emp_risk = self.df.groupby('employment_label')[['calculated_risk_score', 'calculated_resilience_score']].mean()
        emp_risk.plot(kind='bar', ax=ax4, color=['#FF6B6B', '#4ECDC4'])
        ax4.set_title('Risk & Resilience by Employment Status', fontweight='bold')
        ax4.set_xlabel('Employment Status')
        ax4.set_ylabel('Average Score')
        ax4.legend(['Risk Score', 'Resilience Score'])
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_demographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/02_demographic_analysis.pdf', bbox_inches='tight')
        print("✓ Generated Figure 2: Demographic Analysis")
        plt.close()
    
    def plot_financial_factors(self):
        """Figure 3: Financial Risk Factors"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Credit score impact
        credit_bins = [0, 600, 700, 750, 850]
        credit_labels = ['Poor\n(300-599)', 'Fair\n(600-699)', 'Good\n(700-749)', 'Excellent\n(750+)']
        self.df['credit_group'] = pd.cut(self.df['credit_score'], bins=credit_bins, labels=credit_labels)
        
        credit_analysis = self.df.groupby('credit_group')[['calculated_risk_score', 'calculated_resilience_score']].mean()
        credit_analysis.plot(kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4'])
        ax1.set_title('Risk & Resilience by Credit Score Range', fontweight='bold')
        ax1.set_xlabel('Credit Score Category')
        ax1.set_ylabel('Average Score')
        ax1.legend(['Risk Score', 'Resilience Score'])
        ax1.tick_params(axis='x', rotation=0)
        
        # Income vs Risk scatter
        ax2.scatter(self.df['income'], self.df['calculated_risk_score'], alpha=0.6, color='#FF6B6B')
        ax2.set_xlabel('Annual Income ($)')
        ax2.set_ylabel('Risk Score')
        ax2.set_title('Income vs Risk Score', fontweight='bold')
        z = np.polyfit(self.df['income'], self.df['calculated_risk_score'], 1)
        p = np.poly1d(z)
        ax2.plot(self.df['income'].sort_values(), p(self.df['income'].sort_values()), "r--", alpha=0.8)
        
        # Debt to Income Ratio impact
        debt_ratio_bins = [0, 0.3, 0.5, 0.8, float('inf')]
        debt_ratio_labels = ['Low\n(<0.3)', 'Moderate\n(0.3-0.5)', 'High\n(0.5-0.8)', 'Very High\n(>0.8)']
        self.df['debt_ratio_group'] = pd.cut(self.df['debt_to_income_ratio'], bins=debt_ratio_bins, labels=debt_ratio_labels)
        
        debt_analysis = self.df.groupby('debt_ratio_group')[['calculated_risk_score', 'calculated_resilience_score']].mean()
        debt_analysis.plot(kind='bar', ax=ax3, color=['#FF6B6B', '#4ECDC4'])
        ax3.set_title('Risk & Resilience by Debt-to-Income Ratio', fontweight='bold')
        ax3.set_xlabel('Debt-to-Income Ratio')
        ax3.set_ylabel('Average Score')
        ax3.legend(['Risk Score', 'Resilience Score'])
        ax3.tick_params(axis='x', rotation=0)
        
        # Savings impact
        savings_bins = [0, 20000, 50000, 100000, float('inf')]
        savings_labels = ['Low\n(<$20K)', 'Moderate\n($20K-$50K)', 'Good\n($50K-$100K)', 'High\n(>$100K)']
        self.df['savings_group'] = pd.cut(self.df['savings_amount'], bins=savings_bins, labels=savings_labels)
        
        savings_analysis = self.df.groupby('savings_group')[['calculated_risk_score', 'calculated_resilience_score']].mean()
        savings_analysis.plot(kind='bar', ax=ax4, color=['#FF6B6B', '#4ECDC4'])
        ax4.set_title('Risk & Resilience by Savings Amount', fontweight='bold')
        ax4.set_xlabel('Savings Level')
        ax4.set_ylabel('Average Score')
        ax4.legend(['Risk Score', 'Resilience Score'])
        ax4.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_financial_factors.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/03_financial_factors.pdf', bbox_inches='tight')
        print("✓ Generated Figure 3: Financial Factors")
        plt.close()
    
    def plot_correlation_analysis(self):
        """Figure 4: Correlation Heatmap"""
        # Select key numerical features for correlation
        correlation_features = [
            'age', 'income', 'savings_amount', 'debt_amount', 'credit_score',
            'family_size', 'social_support_network', 'stress_level',
            'calculated_risk_score', 'calculated_resilience_score', 'net_risk_score'
        ]
        
        corr_matrix = self.df[correlation_features].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Hide upper triangle
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
        ax.set_title('Feature Correlation Matrix\n(Risk & Resilience Factors)', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/04_correlation_heatmap.pdf', bbox_inches='tight')
        print("✓ Generated Figure 4: Correlation Analysis")
        plt.close()
    
    def plot_risk_category_profiles(self):
        """Figure 5: Risk Category Profiles"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        profile_features = [
            ('age', 'Age'),
            ('income', 'Income ($)'),
            ('credit_score', 'Credit Score'),
            ('debt_amount', 'Debt Amount ($)'),
            ('savings_amount', 'Savings Amount ($)'),
            ('stress_level', 'Stress Level (1-10)')
        ]
        
        for i, (feature, label) in enumerate(profile_features):
            risk_categories = ['Low Risk', 'Medium Risk', 'High Risk']
            colors = ['#4ECDC4', '#FFE66D', '#FF6B6B']
            
            for j, (category, color) in enumerate(zip(risk_categories, colors)):
                data = self.df[self.df['risk_category'] == category][feature]
                axes[i].hist(data, alpha=0.7, label=category, color=color, bins=20)
            
            axes[i].set_xlabel(label)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'{label} Distribution by Risk Category', fontweight='bold')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_risk_category_profiles.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/05_risk_category_profiles.pdf', bbox_inches='tight')
        print("✓ Generated Figure 5: Risk Category Profiles")
        plt.close()
    
    def plot_summary_statistics(self):
        """Figure 6: Key Statistics Summary"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Risk category statistics
        stats_by_category = self.df.groupby('risk_category').agg({
            'calculated_risk_score': 'mean',
            'calculated_resilience_score': 'mean',
            'income': 'mean',
            'debt_amount': 'mean',
            'credit_score': 'mean'
        }).round(1)
        
        # Population counts
        category_counts = self.df['risk_category'].value_counts()
        ax1.bar(category_counts.index, category_counts.values, color=['#4ECDC4', '#FFE66D', '#FF6B6B'])
        ax1.set_title('Population by Risk Category', fontweight='bold')
        ax1.set_ylabel('Number of Individuals')
        for i, v in enumerate(category_counts.values):
            ax1.text(i, v + 20, str(v), ha='center', fontweight='bold')
        
        # Average scores by category
        scores_data = stats_by_category[['calculated_risk_score', 'calculated_resilience_score']]
        scores_data.plot(kind='bar', ax=ax2, color=['#FF6B6B', '#4ECDC4'])
        ax2.set_title('Average Risk & Resilience Scores by Category', fontweight='bold')
        ax2.set_ylabel('Average Score')
        ax2.legend(['Risk Score', 'Resilience Score'])
        ax2.tick_params(axis='x', rotation=45)
        
        # Financial profiles by category
        financial_data = stats_by_category[['income', 'debt_amount']] / 1000  # Convert to thousands
        financial_data.plot(kind='bar', ax=ax3, color=['#2E8B57', '#DC143C'])
        ax3.set_title('Average Financial Profile by Risk Category', fontweight='bold')
        ax3.set_ylabel('Amount (Thousands $)')
        ax3.legend(['Income', 'Debt'])
        ax3.tick_params(axis='x', rotation=45)
        
        # Credit score by category
        credit_data = stats_by_category['credit_score']
        ax4.bar(credit_data.index, credit_data.values, color=['#4ECDC4', '#FFE66D', '#FF6B6B'])
        ax4.set_title('Average Credit Score by Risk Category', fontweight='bold')
        ax4.set_ylabel('Credit Score')
        for i, v in enumerate(credit_data.values):
            ax4.text(i, v + 10, str(int(v)), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/06_summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/06_summary_statistics.pdf', bbox_inches='tight')
        print("✓ Generated Figure 6: Summary Statistics")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all white paper graphics"""
        print("GENERATING WHITE PAPER VISUALIZATIONS")
        print("="*50)
        
        self.plot_risk_distribution()
        self.plot_demographic_analysis()
        self.plot_financial_factors()
        self.plot_correlation_analysis()
        self.plot_risk_category_profiles()
        self.plot_summary_statistics()
        
        print("\n" + "="*50)
        print("✅ ALL VISUALIZATIONS GENERATED!")
        print(f"📁 Output directory: {self.output_dir}/")
        print("📄 Both PNG (high-res) and PDF formats created")
        print("🎯 Ready for white paper integration!")
        
        # Generate a summary report
        self.create_summary_report()
    
    def create_summary_report(self):
        """Create a text summary of key findings"""
        report = f"""
RISK & RESILIENCE ANALYSIS - KEY FINDINGS
==========================================
Dataset: {len(self.df)} individuals analyzed

POPULATION DISTRIBUTION:
- High Risk: {len(self.df[self.df['risk_category'] == 'High Risk'])}/{len(self.df)} ({len(self.df[self.df['risk_category'] == 'High Risk'])/len(self.df)*100:.1f}%)
- Medium Risk: {len(self.df[self.df['risk_category'] == 'Medium Risk'])}/{len(self.df)} ({len(self.df[self.df['risk_category'] == 'Medium Risk'])/len(self.df)*100:.1f}%)
- Low Risk: {len(self.df[self.df['risk_category'] == 'Low Risk'])}/{len(self.df)} ({len(self.df[self.df['risk_category'] == 'Low Risk'])/len(self.df)*100:.1f}%)

AVERAGE SCORES:
- Overall Risk Score: {self.df['calculated_risk_score'].mean():.1f} ± {self.df['calculated_risk_score'].std():.1f}
- Overall Resilience Score: {self.df['calculated_resilience_score'].mean():.1f} ± {self.df['calculated_resilience_score'].std():.1f}
- Overall Net Risk: {self.df['net_risk_score'].mean():.1f} ± {self.df['net_risk_score'].std():.1f}

HIGH RISK GROUP PROFILE:
- Average Age: {self.df[self.df['risk_category'] == 'High Risk']['age'].mean():.1f}
- Average Income: ${self.df[self.df['risk_category'] == 'High Risk']['income'].mean():,.0f}
- Average Credit Score: {self.df[self.df['risk_category'] == 'High Risk']['credit_score'].mean():.0f}
- Unemployment Rate: {len(self.df[(self.df['risk_category'] == 'High Risk') & (self.df['employment_status'] == 0)])/len(self.df[self.df['risk_category'] == 'High Risk'])*100:.1f}%

LOW RISK GROUP PROFILE:
- Average Age: {self.df[self.df['risk_category'] == 'Low Risk']['age'].mean():.1f}
- Average Income: ${self.df[self.df['risk_category'] == 'Low Risk']['income'].mean():,.0f}
- Average Credit Score: {self.df[self.df['risk_category'] == 'Low Risk']['credit_score'].mean():.0f}
- Full-time Employment Rate: {len(self.df[(self.df['risk_category'] == 'Low Risk') & (self.df['employment_status'] == 2)])/len(self.df[self.df['risk_category'] == 'Low Risk'])*100:.1f}%

KEY CORRELATIONS:
- Risk vs Credit Score: r = {self.df['calculated_risk_score'].corr(self.df['credit_score']):.3f}
- Resilience vs Income: r = {self.df['calculated_resilience_score'].corr(self.df['income']):.3f}
- Risk vs Age: r = {self.df['calculated_risk_score'].corr(self.df['age']):.3f}
"""
        
        with open(f'{self.output_dir}/analysis_summary.txt', 'w') as f:
            f.write(report)
        
        print("📊 Generated analysis summary report")


# Main execution
if __name__ == "__main__":
    # Use training dataset for visualization
    visualizer = RiskResilienceVisualizer('ml_datasets/training_dataset.csv')
    visualizer.generate_all_visualizations()