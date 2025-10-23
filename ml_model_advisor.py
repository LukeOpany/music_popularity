import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             mean_squared_error, mean_absolute_error, r2_score, classification_report)
import warnings
warnings.filterwarnings('ignore')

class MLModelAdvisor:
    def __init__(self):
        self.problem_type = None
        self.dataset_size = None
        self.interpretability = None
        self.recommended_models = []
        self.results = {}
        
    def ask_questions(self):
        """Interactive questionnaire to understand the ML problem"""
        print("=" * 60)
        print("ML MODEL RECOMMENDATION SYSTEM")
        print("=" * 60)
        print("\nLet's find the best models for your problem!\n")
        
        # Question 1: Problem Type
        print("Question 1: What type of problem are you solving?")
        print("  1. Classification (predicting categories/labels)")
        print("  2. Regression (predicting numbers/continuous values)")
        problem_choice = input("\nEnter 1 or 2: ").strip()
        self.problem_type = "classification" if problem_choice == "1" else "regression"
        
        # Question 2: Dataset Size
        print("\n" + "-" * 60)
        print("Question 2: How many rows of data do you have?")
        print("  1. Small (< 1,000 rows)")
        print("  2. Medium (1,000 - 10,000 rows)")
        print("  3. Large (10,000 - 100,000 rows)")
        print("  4. Very Large (> 100,000 rows)")
        size_choice = input("\nEnter 1, 2, 3, or 4: ").strip()
        size_map = {"1": "small", "2": "medium", "3": "large", "4": "very_large"}
        self.dataset_size = size_map.get(size_choice, "medium")
        
        # Question 3: Interpretability
        print("\n" + "-" * 60)
        print("Question 3: Do you need to explain/interpret the model predictions?")
        print("  1. Yes, interpretability is important")
        print("  2. No, I just care about accuracy")
        interp_choice = input("\nEnter 1 or 2: ").strip()
        self.interpretability = interp_choice == "1"
        
        print("\n" + "=" * 60)
        
    def recommend_models(self):
        """Recommend 3-5 models based on user answers"""
        models = []
        
        if self.problem_type == "classification":
            # Always start with logistic regression as baseline
            models.append(("Logistic Regression", LogisticRegression(max_iter=1000)))
            
            if self.interpretability:
                models.append(("Decision Tree", DecisionTreeClassifier(random_state=42)))
                models.append(("Naive Bayes", GaussianNB()))
            else:
                models.append(("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)))
                models.append(("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)))
                
            if self.dataset_size in ["small", "medium"]:
                models.append(("K-Nearest Neighbors", KNeighborsClassifier()))
            
            if self.dataset_size in ["medium", "large"] and not self.interpretability:
                models.append(("Support Vector Machine", SVC(random_state=42)))
                
        else:  # regression
            # Always start with linear regression as baseline
            models.append(("Linear Regression", LinearRegression()))
            
            if self.interpretability:
                models.append(("Decision Tree", DecisionTreeRegressor(random_state=42)))
            else:
                models.append(("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)))
                models.append(("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)))
                
            if self.dataset_size in ["small", "medium"]:
                models.append(("K-Nearest Neighbors", KNeighborsRegressor()))
            
            if self.dataset_size in ["medium", "large"] and not self.interpretability:
                models.append(("Support Vector Machine", SVR()))
        
        self.recommended_models = models[:5]  # Limit to 5 models
        
        print("\nBased on your answers, here are the recommended models:")
        print("-" * 60)
        for i, (name, _) in enumerate(self.recommended_models, 1):
            print(f"  {i}. {name}")
        print("-" * 60)
        
    def evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all recommended models"""
        print("\n\nEVALUATING MODELS...")
        print("=" * 60)
        
        for name, model in self.recommended_models:
            print(f"\nTraining {name}...")
            
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Evaluate based on problem type
                if self.problem_type == "classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    self.results[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    }
                    
                    print(f"  ✓ Accuracy: {accuracy:.4f}")
                    print(f"  ✓ Precision: {precision:.4f}")
                    print(f"  ✓ Recall: {recall:.4f}")
                    print(f"  ✓ F1-Score: {f1:.4f}")
                    
                else:  # regression
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    self.results[name] = {
                        'model': model,
                        'rmse': rmse,
                        'mae': mae,
                        'r2_score': r2
                    }
                    
                    print(f"  ✓ RMSE: {rmse:.4f}")
                    print(f"  ✓ MAE: {mae:.4f}")
                    print(f"  ✓ R² Score: {r2:.4f}")
                    
            except Exception as e:
                print(f"  ✗ Error training {name}: {str(e)}")
                
        print("\n" + "=" * 60)
        
    def suggest_feature_engineering(self, X_train, y_train):
        """Provide specific feature engineering recommendations"""
        print("\n\n🔧 FEATURE ENGINEERING RECOMMENDATIONS")
        print("=" * 60)
        
        n_features = X_train.shape[1]
        n_samples = X_train.shape[0]
        
        print(f"\nCurrent Dataset: {n_samples} rows, {n_features} features")
        print("\nHere are specific techniques you should try:\n")
        
        # Analyze feature characteristics
        print("1. INTERACTION FEATURES")
        print("-" * 60)
        print("   Create new features by multiplying or combining existing ones:")
        print("   • If you have 'age' and 'income': create 'age_income_ratio'")
        print("   • If you have 'height' and 'weight': create 'BMI = weight/height²'")
        print("   • Try polynomial features: X² or X³ for non-linear relationships")
        print("   Code: from sklearn.preprocessing import PolynomialFeatures")
        
        print("\n2. AGGREGATION FEATURES")
        print("-" * 60)
        print("   Create summary statistics from your features:")
        print("   • Sum of related features")
        print("   • Average of similar features")
        print("   • Max/Min across feature groups")
        print("   • Standard deviation if you have multiple measurements")
        
        print("\n3. BINNING/DISCRETIZATION")
        print("-" * 60)
        print("   Convert continuous features into categories:")
        print("   • Age → age_group (young/middle/senior)")
        print("   • Income → income_bracket (low/medium/high)")
        print("   • Helps capture non-linear patterns")
        print("   Code: pd.cut() or pd.qcut()")
        
        print("\n4. DATETIME FEATURES (if applicable)")
        print("-" * 60)
        print("   If you have dates, extract:")
        print("   • Day of week, month, quarter, year")
        print("   • Is_weekend, is_holiday")
        print("   • Time since a reference date")
        print("   • Season (spring/summer/fall/winter)")
        
        print("\n5. ENCODING CATEGORICAL VARIABLES")
        print("-" * 60)
        print("   Make sure categories are properly encoded:")
        print("   • One-Hot Encoding for nominal categories (color, city)")
        print("   • Label Encoding for ordinal categories (low/med/high)")
        print("   • Target Encoding for high-cardinality features")
        print("   Code: pd.get_dummies() or sklearn.preprocessing.OneHotEncoder")
        
        print("\n6. HANDLING MISSING VALUES")
        print("-" * 60)
        print("   Don't just drop or fill with mean:")
        print("   • Create 'is_missing' binary indicator feature")
        print("   • Use median for skewed distributions")
        print("   • Use mode for categorical features")
        print("   • Try KNN imputation for better results")
        
        print("\n7. OUTLIER TREATMENT")
        print("-" * 60)
        print("   Handle extreme values:")
        print("   • Cap values at 95th/99th percentile")
        print("   • Use log transformation for skewed features")
        print("   • Create 'is_outlier' indicator feature")
        print("   Code: np.log1p() or winsorization")
        
        print("\n8. FEATURE SELECTION")
        print("-" * 60)
        print("   Remove features that don't help:")
        print("   • Drop features with >80% missing values")
        print("   • Remove features with zero variance")
        print("   • Use correlation matrix to drop redundant features (correlation >0.95)")
        print("   • Try SelectKBest or Recursive Feature Elimination")
        print("   Code: from sklearn.feature_selection import SelectKBest, RFE")
        
        # Check for potential issues
        print("\n\n🔍 QUICK DATA QUALITY CHECKS:")
        print("-" * 60)
        
        # Check for constant features
        if hasattr(X_train, 'var'):
            zero_var = (X_train.var() == 0).sum()
            if zero_var > 0:
                print(f"⚠ Warning: {zero_var} features have zero variance (constant values)")
                print("  → Remove these features, they provide no information")
        
        # Check feature count
        if n_features < 5:
            print(f"⚠ Warning: Only {n_features} features - this might not be enough")
            print("  → Try creating more features using techniques above")
        elif n_features > n_samples / 10:
            print(f"⚠ Warning: Too many features ({n_features}) for sample size ({n_samples})")
            print("  → Consider feature selection or dimensionality reduction (PCA)")
        
        print("\n" + "=" * 60)
    
    def interpret_results(self, X_train=None, y_train=None):
        """Provide interpretation and recommendations"""
        print("\n\nRESULTS INTERPRETATION")
        print("=" * 60)
        
        if not self.results:
            print("No models were successfully trained.")
            return
        
        # Find best model
        if self.problem_type == "classification":
            best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
            best_name = best_model[0]
            best_accuracy = best_model[1]['accuracy']
            
            print(f"\n🏆 BEST MODEL: {best_name}")
            print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            
            print("\n📊 WHAT DO THESE METRICS MEAN?")
            print("-" * 60)
            print("• Accuracy: % of predictions that were correct")
            print("• Precision: Of the items predicted as positive, how many were actually positive?")
            print("• Recall: Of all actual positive items, how many did we find?")
            print("• F1-Score: Balance between precision and recall (harmonic mean)")
            
            print("\n📈 MODEL RANKINGS:")
            print("-" * 60)
            ranked = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            for i, (name, metrics) in enumerate(ranked, 1):
                print(f"{i}. {name}: {metrics['accuracy']*100:.2f}% accuracy")
            
            print("\n💡 RECOMMENDATIONS:")
            print("-" * 60)
            if best_accuracy >= 0.90:
                print("✓ Excellent performance! Your best model is working very well.")
                print("✓ Consider deploying this model or fine-tuning it further.")
            elif best_accuracy >= 0.75:
                print("✓ Good performance! The model is decent but has room for improvement.")
                print("✓ Consider: Feature engineering, hyperparameter tuning, or collecting more data.")
            elif best_accuracy >= 0.60:
                print("⚠ Moderate performance. The model is better than random but needs work.")
                print("✓ Try: Adding more features, handling imbalanced classes, or collecting more data.")
                if X_train is not None and y_train is not None:
                    self.suggest_feature_engineering(X_train, y_train)
            else:
                print("⚠ Poor performance. The model is struggling.")
                print("✓ Recommendations:")
                print("  - Check for data quality issues")
                print("  - Ensure proper train-test split")
                print("  - Consider feature engineering")
                print("  - Check if you have enough data")
                if X_train is not None and y_train is not None:
                    self.suggest_feature_engineering(X_train, y_train)
                
        else:  # regression
            best_model = max(self.results.items(), key=lambda x: x[1]['r2_score'])
            best_name = best_model[0]
            best_r2 = best_model[1]['r2_score']
            best_rmse = best_model[1]['rmse']
            
            print(f"\n🏆 BEST MODEL: {best_name}")
            print(f"   R² Score: {best_r2:.4f}")
            print(f"   RMSE: {best_rmse:.4f}")
            
            print("\n📊 WHAT DO THESE METRICS MEAN?")
            print("-" * 60)
            print("• RMSE: Average prediction error (lower is better)")
            print("• MAE: Average absolute error (lower is better)")
            print("• R² Score: How much variance the model explains (0-1, higher is better)")
            print("  - 1.0 = perfect predictions")
            print("  - 0.0 = model is as good as predicting the mean")
            print("  - negative = worse than predicting the mean")
            
            print("\n📈 MODEL RANKINGS:")
            print("-" * 60)
            ranked = sorted(self.results.items(), key=lambda x: x[1]['r2_score'], reverse=True)
            for i, (name, metrics) in enumerate(ranked, 1):
                print(f"{i}. {name}: R²={metrics['r2_score']:.4f}, RMSE={metrics['rmse']:.4f}")
            
            print("\n💡 RECOMMENDATIONS:")
            print("-" * 60)
            if best_r2 >= 0.80:
                print("✓ Excellent performance! Your model explains most of the variance.")
                print("✓ Consider deploying this model or fine-tuning it further.")
            elif best_r2 >= 0.60:
                print("✓ Good performance! The model captures the main patterns.")
                print("✓ Consider: Feature engineering, hyperparameter tuning, or polynomial features.")
            elif best_r2 >= 0.30:
                print("⚠ Moderate performance. The model captures some patterns but misses many.")
                print("✓ Try: Adding more relevant features, handling outliers, or transforming features.")
                if X_train is not None and y_train is not None:
                    self.suggest_feature_engineering(X_train, y_train)
            else:
                print("⚠ Poor performance. The model is struggling to capture patterns.")
                print("✓ Recommendations:")
                print("  - Check for data quality issues and outliers")
                print("  - Look for non-linear relationships")
                print("  - Consider feature engineering or feature selection")
                print("  - Ensure you have relevant features")
                if X_train is not None and y_train is not None:
                    self.suggest_feature_engineering(X_train, y_train)
        
        print("\n" + "=" * 60)

# Example usage
def run_model_advisor(X_train, X_test, y_train, y_test):
    """
    Main function to run the ML Model Advisor
    
    Parameters:
    X_train, X_test: Training and test features
    y_train, y_test: Training and test target values
    """
    advisor = MLModelAdvisor()
    advisor.ask_questions()
    advisor.recommend_models()
    
    proceed = input("\nDo you want to evaluate these models now? (yes/no): ").strip().lower()
    if proceed in ['yes', 'y']:
        advisor.evaluate_models(X_train, X_test, y_train, y_test)
        advisor.interpret_results(X_train, y_train)  # Pass data for feature engineering suggestions
    else:
        print("\nEvaluation skipped. You can call evaluate_models() later.")
    
    return advisor

# Example with sample data
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("⚠️  IMPORTANT REMINDER: DATA SCALING")
    print("=" * 60)
    print("\nBefore using this advisor, make sure you've scaled your data!")
    print("\n✓ CORRECT ORDER:")
    print("  1. Train-test split")
    print("  2. Fit scaler on TRAINING data only")
    print("  3. Transform BOTH training and test data")
    print("  4. Run this advisor")
    print("\n📝 Example code:")
    print("  from sklearn.preprocessing import StandardScaler")
    print("  scaler = StandardScaler()")
    print("  X_train_scaled = scaler.fit_transform(X_train)  # Fit on train")
    print("  X_test_scaled = scaler.transform(X_test)        # Transform test")
    print("  advisor = run_model_advisor(X_train_scaled, X_test_scaled, y_train, y_test)")
    print("\n" + "=" * 60)
    
    print("\n\nGENERATING SAMPLE DATA FOR DEMONSTRATION...")
    print("(Replace this with your actual train-test split data)\n")
    
    # Create sample classification data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                               n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the data (demonstrating proper technique)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Run the advisor
    advisor = run_model_advisor(X_train_scaled, X_test_scaled, y_train, y_test)