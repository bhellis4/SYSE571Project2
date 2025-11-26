import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

class RiskResilienceML:
    def __init__(self, data_directory='ml_datasets'):
        """Initialize ML pipeline with training data"""
        self.data_dir = data_directory
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # Load datasets
        self.load_datasets()
        self.prepare_features()
        
        print(f"ML Pipeline initialized with {len(self.train_data)} training samples")
    
    def load_datasets(self):
        """Load training, testing, and validation datasets"""
        self.train_data = pd.read_csv(f'{self.data_dir}/training_dataset.csv')
        self.test_data = pd.read_csv(f'{self.data_dir}/testing_dataset.csv')
        self.validation_data = pd.read_csv(f'{self.data_dir}/verification_dataset.csv')
        self.production_data = pd.read_csv(f'{self.data_dir}/production_dataset.csv')
        
        print(f"Loaded datasets:")
        print(f"  Training: {len(self.train_data)} samples")
        print(f"  Testing: {len(self.test_data)} samples")  
        print(f"  Validation: {len(self.validation_data)} samples")
        print(f"  Production: {len(self.production_data)} samples")
    
    def prepare_features(self):
        """Prepare features for ML models"""
        # Define feature columns (excluding target variables and identifiers)
        self.feature_columns = [
            'age', 'gender', 'income', 'education_level', 'marital_status',
            'mental_health_condition', 'savings_amount', 'employment_status',
            'family_size', 'debt_amount', 'physical_activity_level', 'credit_score',
            'sleep_quality', 'risk_tolerance', 'insurance_coverage',
            'social_support_network', 'stress_level', 'access_to_healthcare',
            'financial_goals', 'debt_to_income_ratio'
        ]
        
        # Target variables
        self.risk_score_target = 'calculated_risk_score'
        self.resilience_score_target = 'calculated_resilience_score'
        self.risk_category_target = 'risk_category'
        
        # Prepare training features
        self.X_train = self.train_data[self.feature_columns]
        self.y_risk_score = self.train_data[self.risk_score_target]
        self.y_resilience_score = self.train_data[self.resilience_score_target]
        
        # Encode risk categories for classification
        self.label_encoder = LabelEncoder()
        self.y_risk_category = self.label_encoder.fit_transform(self.train_data[self.risk_category_target])
        
        print(f"Features prepared: {len(self.feature_columns)} features")
        print(f"Target variables: Risk Score (regression), Resilience Score (regression), Risk Category (classification)")
    
    def train_risk_score_models(self):
        """Train models to predict risk scores (regression)"""
        print("\n" + "="*60)
        print("TRAINING RISK SCORE PREDICTION MODELS (REGRESSION)")
        print("="*60)
        
        # Scale features for neural network
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train)
        self.scalers['risk_score'] = scaler
        
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            'Linear Regression': LinearRegression()
        }
        
        self.models['risk_score'] = {}
        self.results['risk_score'] = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled features for neural network, original for others
            X_input = X_scaled if name == 'Neural Network' else self.X_train
            
            # Train model
            model.fit(X_input, self.y_risk_score)
            self.models['risk_score'][name] = model
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_input, self.y_risk_score, cv=5, scoring='r2')
            
            # Predictions on training set
            y_pred = model.predict(X_input)
            r2 = r2_score(self.y_risk_score, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_risk_score, y_pred))
            
            self.results['risk_score'][name] = {
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'train_r2': r2,
                'train_rmse': rmse
            }
            
            print(f"  Cross-val R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            print(f"  Training R²: {r2:.3f}")
            print(f"  Training RMSE: {rmse:.3f}")
    
    def train_resilience_score_models(self):
        """Train models to predict resilience scores (regression)"""
        print("\n" + "="*60)
        print("TRAINING RESILIENCE SCORE PREDICTION MODELS (REGRESSION)")
        print("="*60)
        
        # Scale features for neural network
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train)
        self.scalers['resilience_score'] = scaler
        
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            'Linear Regression': LinearRegression()
        }
        
        self.models['resilience_score'] = {}
        self.results['resilience_score'] = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled features for neural network, original for others
            X_input = X_scaled if name == 'Neural Network' else self.X_train
            
            # Train model
            model.fit(X_input, self.y_resilience_score)
            self.models['resilience_score'][name] = model
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_input, self.y_resilience_score, cv=5, scoring='r2')
            
            # Predictions on training set
            y_pred = model.predict(X_input)
            r2 = r2_score(self.y_resilience_score, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_resilience_score, y_pred))
            
            self.results['resilience_score'][name] = {
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'train_r2': r2,
                'train_rmse': rmse
            }
            
            print(f"  Cross-val R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            print(f"  Training R²: {r2:.3f}")
            print(f"  Training RMSE: {rmse:.3f}")
    
    def train_risk_category_models(self):
        """Train models to predict risk categories (classification)"""
        print("\n" + "="*60)
        print("TRAINING RISK CATEGORY CLASSIFICATION MODELS")
        print("="*60)
        
        # Scale features for neural network
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train)
        self.scalers['risk_category'] = scaler
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(objective='multi:softprob', random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        self.models['risk_category'] = {}
        self.results['risk_category'] = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled features for neural network and logistic regression, original for others
            X_input = X_scaled if name in ['Neural Network', 'Logistic Regression'] else self.X_train
            
            # Train model
            model.fit(X_input, self.y_risk_category)
            self.models['risk_category'][name] = model
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_input, self.y_risk_category, cv=5, scoring='accuracy')
            
            # Predictions on training set
            y_pred = model.predict(X_input)
            accuracy = accuracy_score(self.y_risk_category, y_pred)
            
            self.results['risk_category'][name] = {
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'train_accuracy': accuracy
            }
            
            print(f"  Cross-val Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            print(f"  Training Accuracy: {accuracy:.3f}")
    
    def evaluate_on_test_set(self):
        """Evaluate all models on the test set"""
        print("\n" + "="*60)
        print("EVALUATING MODELS ON TEST SET")
        print("="*60)
        
        X_test = self.test_data[self.feature_columns]
        y_test_risk = self.test_data[self.risk_score_target]
        y_test_resilience = self.test_data[self.resilience_score_target]
        y_test_category = self.label_encoder.transform(self.test_data[self.risk_category_target])
        
        # Test Risk Score Models
        print("\nRISK SCORE PREDICTION - TEST SET RESULTS:")
        print("-" * 50)
        X_test_scaled_risk = self.scalers['risk_score'].transform(X_test)
        
        for name, model in self.models['risk_score'].items():
            X_input = X_test_scaled_risk if name == 'Neural Network' else X_test
            y_pred = model.predict(X_input)
            
            r2 = r2_score(y_test_risk, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_risk, y_pred))
            
            print(f"{name:18} - R²: {r2:.3f}, RMSE: {rmse:.3f}")
            self.results['risk_score'][name]['test_r2'] = r2
            self.results['risk_score'][name]['test_rmse'] = rmse
        
        # Test Resilience Score Models
        print("\nRESILIENCE SCORE PREDICTION - TEST SET RESULTS:")
        print("-" * 50)
        X_test_scaled_resilience = self.scalers['resilience_score'].transform(X_test)
        
        for name, model in self.models['resilience_score'].items():
            X_input = X_test_scaled_resilience if name == 'Neural Network' else X_test
            y_pred = model.predict(X_input)
            
            r2 = r2_score(y_test_resilience, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_resilience, y_pred))
            
            print(f"{name:18} - R²: {r2:.3f}, RMSE: {rmse:.3f}")
            self.results['resilience_score'][name]['test_r2'] = r2
            self.results['resilience_score'][name]['test_rmse'] = rmse
        
        # Test Risk Category Models
        print("\nRISK CATEGORY CLASSIFICATION - TEST SET RESULTS:")
        print("-" * 50)
        X_test_scaled_category = self.scalers['risk_category'].transform(X_test)
        
        for name, model in self.models['risk_category'].items():
            X_input = X_test_scaled_category if name in ['Neural Network', 'Logistic Regression'] else X_test
            y_pred = model.predict(X_input)
            
            accuracy = accuracy_score(y_test_category, y_pred)
            
            print(f"{name:18} - Accuracy: {accuracy:.3f}")
            self.results['risk_category'][name]['test_accuracy'] = accuracy
    
    def analyze_feature_importance(self):
        """Analyze feature importance for tree-based models"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Feature importance for Random Forest models
        tasks = ['risk_score', 'resilience_score', 'risk_category']
        
        for task in tasks:
            if task in self.models and 'Random Forest' in self.models[task]:
                print(f"\n{task.upper().replace('_', ' ')} - Random Forest Feature Importance:")
                print("-" * 50)
                
                rf_model = self.models[task]['Random Forest']
                importance = rf_model.feature_importances_
                
                # Create feature importance dataframe
                feature_importance = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                # Show top 10 features
                print(feature_importance.head(10).to_string(index=False))
                
                # Store for visualization
                if not hasattr(self, 'feature_importance'):
                    self.feature_importance = {}
                self.feature_importance[task] = feature_importance
    
    def save_models(self, output_dir='trained_models'):
        """Save trained models and results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        with open(f'{output_dir}/risk_resilience_models.pkl', 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns
            }, f)
        
        # Save results
        results_df = pd.DataFrame()
        for task, task_results in self.results.items():
            for model_name, metrics in task_results.items():
                row = {'task': task, 'model': model_name}
                row.update(metrics)
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        
        results_df.to_csv(f'{output_dir}/model_results.csv', index=False)
        
        print(f"\n✅ Models and results saved to: {output_dir}/")
    
    def create_prediction_examples(self):
        """Create examples of model predictions"""
        print("\n" + "="*60)
        print("PREDICTION EXAMPLES")
        print("="*60)
        
        # Get sample individuals from test set
        samples = self.test_data.head(5)
        
        for i, (_, sample) in enumerate(samples.iterrows()):
            print(f"\nExample {i+1}:")
            print(f"  Age: {sample['age']}, Income: ${sample['income']:,.0f}")
            print(f"  Credit Score: {sample['credit_score']}, Employment: {sample['employment_status']}")
            print(f"  Actual Risk Category: {sample['risk_category']}")
            
            # Predict with best models
            X_sample = sample[self.feature_columns].values.reshape(1, -1)
            
            # Risk category prediction (using Random Forest)
            if 'Random Forest' in self.models['risk_category']:
                pred_category_idx = self.models['risk_category']['Random Forest'].predict(X_sample)[0]
                pred_category = self.label_encoder.inverse_transform([pred_category_idx])[0]
                print(f"  Predicted Risk Category: {pred_category}")
            
            # Risk score prediction
            if 'Random Forest' in self.models['risk_score']:
                pred_risk = self.models['risk_score']['Random Forest'].predict(X_sample)[0]
                print(f"  Predicted Risk Score: {pred_risk:.1f} (Actual: {sample['calculated_risk_score']:.1f})")
            
            # Resilience score prediction
            if 'Random Forest' in self.models['resilience_score']:
                pred_resilience = self.models['resilience_score']['Random Forest'].predict(X_sample)[0]
                print(f"  Predicted Resilience Score: {pred_resilience:.1f} (Actual: {sample['calculated_resilience_score']:.1f})")
    
    def run_complete_ml_pipeline(self):
        """Execute the complete ML pipeline"""
        print("RISK & RESILIENCE ML PIPELINE")
        print("="*60)
        
        # Train all models
        self.train_risk_score_models()
        self.train_resilience_score_models()
        self.train_risk_category_models()
        
        # Evaluate on test set
        self.evaluate_on_test_set()
        
        # Analyze feature importance
        self.analyze_feature_importance()
        
        # Save models and results
        self.save_models()
        
        # Create prediction examples
        self.create_prediction_examples()
        
        print("\n" + "="*60)
        print("✅ ML PIPELINE COMPLETE!")
        print("📊 Models trained for risk/resilience prediction")
        print("🎯 Ready for production deployment!")
        print("="*60)


if __name__ == "__main__":
    # Run the complete ML pipeline
    ml_pipeline = RiskResilienceML()
    ml_pipeline.run_complete_ml_pipeline()