#!/usr/bin/env python3
"""
Production ML Pipeline - Risk & Resilience Prediction
=====================================================
Loads trained models and runs predictions on combined production dataset (3000 samples)
Generates comprehensive analysis outputs for white paper documentation.
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class ProductionMLPipeline:
    def __init__(self):
        """Initialize production ML pipeline"""
        self.models = {}
        self.scalers = {}
        self.label_encoder = None
        self.feature_columns = None
        
        # Load production dataset
        self.production_data = pd.read_csv('ml_datasets/combined_production_dataset.csv')
        
        # Results storage
        self.predictions = {}
        self.performance_metrics = {}
        
        print("="*60)
        print("PRODUCTION ML PIPELINE - RISK & RESILIENCE PREDICTION")
        print("="*60)
        print(f"Production dataset loaded: {len(self.production_data)} samples")
        
    def load_trained_models(self):
        """Load previously trained models"""
        print("\n📁 Loading trained models...")
        
        # Load models and components
        with open('trained_models/risk_resilience_models.pkl', 'rb') as f:
            model_data = pickle.load(f)
            
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        
        print(f"✅ Loaded {len(self.models)} model types:")
        for task, task_models in self.models.items():
            print(f"   {task}: {list(task_models.keys())}")
    
    def prepare_production_features(self):
        """Prepare features for prediction"""
        print("\n🔧 Preparing features for prediction...")
        
        # Extract feature matrix
        self.X_production = self.production_data[self.feature_columns]
        
        # Get actual values for comparison
        self.y_actual_risk = self.production_data['calculated_risk_score']
        self.y_actual_resilience = self.production_data['calculated_resilience_score']
        self.y_actual_category = self.label_encoder.transform(self.production_data['risk_category'])
        
        print(f"✅ Features prepared: {self.X_production.shape}")
        print(f"   Feature columns: {len(self.feature_columns)}")
        
    def run_production_predictions(self):
        """Run predictions on production dataset"""
        print("\n🎯 Running predictions on production dataset...")
        
        # Initialize predictions storage
        self.predictions = {
            'risk_score': {},
            'resilience_score': {},  
            'risk_category': {}
        }
        
        # Risk Score Predictions
        print("\n--- Risk Score Predictions ---")
        X_scaled_risk = self.scalers['risk_score'].transform(self.X_production)
        
        for model_name, model in self.models['risk_score'].items():
            print(f"  Predicting with {model_name}...")
            X_input = X_scaled_risk if model_name == 'Neural Network' else self.X_production
            predictions = model.predict(X_input)
            self.predictions['risk_score'][model_name] = predictions
            
            # Calculate metrics
            r2 = r2_score(self.y_actual_risk, predictions)
            rmse = np.sqrt(mean_squared_error(self.y_actual_risk, predictions))
            print(f"    R²: {r2:.3f}, RMSE: {rmse:.3f}")
        
        # Resilience Score Predictions
        print("\n--- Resilience Score Predictions ---")
        X_scaled_resilience = self.scalers['resilience_score'].transform(self.X_production)
        
        for model_name, model in self.models['resilience_score'].items():
            print(f"  Predicting with {model_name}...")
            X_input = X_scaled_resilience if model_name == 'Neural Network' else self.X_production
            predictions = model.predict(X_input)
            self.predictions['resilience_score'][model_name] = predictions
            
            # Calculate metrics
            r2 = r2_score(self.y_actual_resilience, predictions)
            rmse = np.sqrt(mean_squared_error(self.y_actual_resilience, predictions))
            print(f"    R²: {r2:.3f}, RMSE: {rmse:.3f}")
            
        # Risk Category Predictions
        print("\n--- Risk Category Predictions ---")
        X_scaled_category = self.scalers['resilience_score'].transform(self.X_production)  # Use any scaler for category
        
        for model_name, model in self.models['risk_category'].items():
            print(f"  Predicting with {model_name}...")
            X_input = X_scaled_category if model_name == 'Neural Network' else self.X_production
            predictions = model.predict(X_input)
            self.predictions['risk_category'][model_name] = predictions
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_actual_category, predictions)
            print(f"    Accuracy: {accuracy:.3f}")
    
    def generate_production_analysis(self):
        """Generate comprehensive production analysis"""
        print("\n📊 Generating production analysis...")
        
        # Best model predictions (based on previous test performance)
        best_risk_pred = self.predictions['risk_score']['XGBoost']
        best_resilience_pred = self.predictions['resilience_score']['XGBoost']  
        best_category_pred = self.predictions['risk_category']['Logistic Regression']
        
        # Create analysis results
        analysis_results = {
            'dataset_size': len(self.production_data),
            'risk_score_stats': {
                'actual_mean': float(self.y_actual_risk.mean()),
                'actual_std': float(self.y_actual_risk.std()),
                'predicted_mean': float(best_risk_pred.mean()),
                'predicted_std': float(best_risk_pred.std()),
                'r2_score': float(r2_score(self.y_actual_risk, best_risk_pred)),
                'rmse': float(np.sqrt(mean_squared_error(self.y_actual_risk, best_risk_pred)))
            },
            'resilience_score_stats': {
                'actual_mean': float(self.y_actual_resilience.mean()),
                'actual_std': float(self.y_actual_resilience.std()),
                'predicted_mean': float(best_resilience_pred.mean()),
                'predicted_std': float(best_resilience_pred.std()),
                'r2_score': float(r2_score(self.y_actual_resilience, best_resilience_pred)),
                'rmse': float(np.sqrt(mean_squared_error(self.y_actual_resilience, best_resilience_pred)))
            }
        }
        
        # Category distribution analysis
        actual_categories = self.label_encoder.inverse_transform(self.y_actual_category)
        predicted_categories = self.label_encoder.inverse_transform(best_category_pred)
        
        category_analysis = {
            'actual_distribution': {cat: int(sum(actual_categories == cat)) for cat in ['Low Risk', 'Medium Risk', 'High Risk']},
            'predicted_distribution': {cat: int(sum(predicted_categories == cat)) for cat in ['Low Risk', 'Medium Risk', 'High Risk']},
            'accuracy': float(accuracy_score(self.y_actual_category, best_category_pred))
        }
        
        analysis_results['category_stats'] = category_analysis
        
        # Performance by model
        model_performance = {}
        for task in ['risk_score', 'resilience_score', 'risk_category']:
            model_performance[task] = {}
            for model_name in self.predictions[task].keys():
                if task in ['risk_score', 'resilience_score']:
                    actual = self.y_actual_risk if task == 'risk_score' else self.y_actual_resilience
                    predicted = self.predictions[task][model_name]
                    model_performance[task][model_name] = {
                        'r2_score': float(r2_score(actual, predicted)),
                        'rmse': float(np.sqrt(mean_squared_error(actual, predicted)))
                    }
                else:  # risk_category
                    predicted = self.predictions[task][model_name]
                    model_performance[task][model_name] = {
                        'accuracy': float(accuracy_score(self.y_actual_category, predicted))
                    }
        
        analysis_results['model_performance'] = model_performance
        
        self.analysis_results = analysis_results
        
        return analysis_results
    
    def create_production_visualizations(self, output_dir='production_white_paper_graphics'):
        """Create visualizations for production analysis"""
        print(f"\n📈 Creating visualizations in {output_dir}/...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for publication-quality plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Best model predictions
        best_risk_pred = self.predictions['risk_score']['XGBoost']
        best_resilience_pred = self.predictions['resilience_score']['XGBoost']
        best_category_pred = self.predictions['risk_category']['Logistic Regression']
        
        # 1. Prediction vs Actual Scatter Plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Risk Score Prediction vs Actual
        axes[0].scatter(self.y_actual_risk, best_risk_pred, alpha=0.6, s=20)
        axes[0].plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Risk Score')
        axes[0].set_ylabel('Predicted Risk Score (XGBoost)')
        axes[0].set_title(f'Risk Score Predictions\nR² = {r2_score(self.y_actual_risk, best_risk_pred):.3f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Resilience Score Prediction vs Actual
        axes[1].scatter(self.y_actual_resilience, best_resilience_pred, alpha=0.6, s=20)
        axes[1].plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect Prediction')
        axes[1].set_xlabel('Actual Resilience Score')
        axes[1].set_ylabel('Predicted Resilience Score (XGBoost)')
        axes[1].set_title(f'Resilience Score Predictions\nR² = {r2_score(self.y_actual_resilience, best_resilience_pred):.3f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/production_prediction_accuracy.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/production_prediction_accuracy.pdf', bbox_inches='tight')
        plt.close()
        
        # 2. Distribution Comparisons
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Risk Score Distributions
        axes[0,0].hist(self.y_actual_risk, bins=30, alpha=0.7, label='Actual', density=True)
        axes[0,0].hist(best_risk_pred, bins=30, alpha=0.7, label='Predicted (XGBoost)', density=True)
        axes[0,0].set_xlabel('Risk Score')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_title('Risk Score Distribution Comparison')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Resilience Score Distributions
        axes[0,1].hist(self.y_actual_resilience, bins=30, alpha=0.7, label='Actual', density=True)
        axes[0,1].hist(best_resilience_pred, bins=30, alpha=0.7, label='Predicted (XGBoost)', density=True)
        axes[0,1].set_xlabel('Resilience Score')
        axes[0,1].set_ylabel('Density')
        axes[0,1].set_title('Resilience Score Distribution Comparison')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Category Distribution Comparison
        actual_categories = self.label_encoder.inverse_transform(self.y_actual_category)
        predicted_categories = self.label_encoder.inverse_transform(best_category_pred)
        
        categories = ['Low Risk', 'Medium Risk', 'High Risk']
        actual_counts = [sum(actual_categories == cat) for cat in categories]
        pred_counts = [sum(predicted_categories == cat) for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1,0].bar(x - width/2, actual_counts, width, label='Actual', alpha=0.8)
        axes[1,0].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        axes[1,0].set_xlabel('Risk Category')
        axes[1,0].set_ylabel('Count')
        axes[1,0].set_title('Risk Category Distribution Comparison')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(categories)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_actual_category, best_category_pred)
        im = axes[1,1].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[1,1].set_title('Risk Category Confusion Matrix')
        tick_marks = np.arange(len(categories))
        axes[1,1].set_xticks(tick_marks)
        axes[1,1].set_yticks(tick_marks)
        axes[1,1].set_xticklabels(categories, rotation=45)
        axes[1,1].set_yticklabels(categories)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            axes[1,1].text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        axes[1,1].set_ylabel('Actual')
        axes[1,1].set_xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/production_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/production_distribution_analysis.pdf', bbox_inches='tight')
        plt.close()
        
        # 3. Model Performance Comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Risk Score Model Performance
        risk_models = list(self.predictions['risk_score'].keys())
        risk_r2_scores = [r2_score(self.y_actual_risk, self.predictions['risk_score'][model]) 
                         for model in risk_models]
        
        axes[0].bar(risk_models, risk_r2_scores, alpha=0.8)
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('Risk Score Prediction Performance')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Resilience Score Model Performance
        resilience_models = list(self.predictions['resilience_score'].keys())
        resilience_r2_scores = [r2_score(self.y_actual_resilience, self.predictions['resilience_score'][model]) 
                               for model in resilience_models]
        
        axes[1].bar(resilience_models, resilience_r2_scores, alpha=0.8)
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('R² Score')
        axes[1].set_title('Resilience Score Prediction Performance')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Risk Category Model Performance
        category_models = list(self.predictions['risk_category'].keys())
        category_accuracies = [accuracy_score(self.y_actual_category, self.predictions['risk_category'][model]) 
                              for model in category_models]
        
        axes[2].bar(category_models, category_accuracies, alpha=0.8)
        axes[2].set_xlabel('Model')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('Risk Category Classification Performance')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/production_model_performance.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/production_model_performance.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to {output_dir}/")
    
    def save_production_results(self, output_dir='production_white_paper_graphics'):
        """Save all production results and analysis"""
        print(f"\n💾 Saving production results to {output_dir}/...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save predictions
        predictions_df = self.production_data.copy()
        
        # Add best model predictions
        predictions_df['predicted_risk_score_xgboost'] = self.predictions['risk_score']['XGBoost']
        predictions_df['predicted_resilience_score_xgboost'] = self.predictions['resilience_score']['XGBoost']
        predicted_categories = self.label_encoder.inverse_transform(self.predictions['risk_category']['Logistic Regression'])
        predictions_df['predicted_risk_category_logistic'] = predicted_categories
        
        predictions_df.to_csv(f'{output_dir}/production_predictions.csv', index=False)
        
        # Save analysis summary
        with open(f'{output_dir}/production_analysis_summary.txt', 'w') as f:
            f.write("PRODUCTION ML ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Dataset Size: {self.analysis_results['dataset_size']} samples\n\n")
            
            f.write("RISK SCORE ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            risk_stats = self.analysis_results['risk_score_stats']
            f.write(f"Actual Mean: {risk_stats['actual_mean']:.2f} (±{risk_stats['actual_std']:.2f})\n")
            f.write(f"Predicted Mean: {risk_stats['predicted_mean']:.2f} (±{risk_stats['predicted_std']:.2f})\n")
            f.write(f"R² Score: {risk_stats['r2_score']:.3f}\n")
            f.write(f"RMSE: {risk_stats['rmse']:.3f}\n\n")
            
            f.write("RESILIENCE SCORE ANALYSIS:\n")
            f.write("-" * 25 + "\n")
            resilience_stats = self.analysis_results['resilience_score_stats']
            f.write(f"Actual Mean: {resilience_stats['actual_mean']:.2f} (±{resilience_stats['actual_std']:.2f})\n")
            f.write(f"Predicted Mean: {resilience_stats['predicted_mean']:.2f} (±{resilience_stats['predicted_std']:.2f})\n")
            f.write(f"R² Score: {resilience_stats['r2_score']:.3f}\n")
            f.write(f"RMSE: {resilience_stats['rmse']:.3f}\n\n")
            
            f.write("RISK CATEGORY ANALYSIS:\n")
            f.write("-" * 23 + "\n")
            cat_stats = self.analysis_results['category_stats']
            f.write(f"Classification Accuracy: {cat_stats['accuracy']:.3f}\n")
            f.write("Actual Distribution:\n")
            for cat, count in cat_stats['actual_distribution'].items():
                f.write(f"  {cat}: {count} ({count/self.analysis_results['dataset_size']*100:.1f}%)\n")
            f.write("Predicted Distribution:\n")
            for cat, count in cat_stats['predicted_distribution'].items():
                f.write(f"  {cat}: {count} ({count/self.analysis_results['dataset_size']*100:.1f}%)\n")
        
        print(f"✅ Production results saved to {output_dir}/")
    
    def run_production_pipeline(self):
        """Execute complete production pipeline"""
        self.load_trained_models()
        self.prepare_production_features()
        self.run_production_predictions()
        self.generate_production_analysis()
        self.create_production_visualizations()
        self.save_production_results()
        
        print("\n" + "="*60)
        print("✅ PRODUCTION ML PIPELINE COMPLETE!")
        print("🎯 Ready for production deployment analysis!")
        print("="*60)

if __name__ == "__main__":
    # Run production pipeline
    pipeline = ProductionMLPipeline()
    pipeline.run_production_pipeline()