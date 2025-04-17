# Fraud Detection Pipeline (Without Topological Data Analysis)
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_recall_curve,
                            average_precision_score, f1_score, precision_score,
                            recall_score, roc_curve, auc, confusion_matrix,
                            classification_report)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.metrics import classification_report_imbalanced
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import logging
import psutil
from collections import Counter
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def check_memory(required_gb):
    """Check if sufficient memory is available"""
    available = psutil.virtual_memory().available / (1024 ** 3)
    if available < required_gb:
        raise MemoryError(f"Need {required_gb}GB RAM, only {available:.2f}GB available")
    print(f"✓ {available:.2f}GB RAM available (required: {required_gb}GB)")


def hybrid_resampling(X_train, y_train, strategy='auto', balance_threshold=0.4, n_jobs=-1):
    """Combine smart resampling with mini-batch processing"""
    n_samples = X_train.shape[0]
    class_counts = Counter(y_train)
    minority_ratio = min(class_counts.values()) / n_samples

    if minority_ratio >= balance_threshold:
        print(f"Data is balanced (minority ratio = {minority_ratio:.1%}). Skipping resampling.")
        return X_train, y_train

    if strategy == 'auto':
        if minority_ratio < 0.05:
            method = SMOTEENN(random_state=42)
            print("Extreme imbalance: Using SMOTEENN")
        elif minority_ratio < 0.1:
            method = SMOTE(sampling_strategy='auto', random_state=42)
            print("Moderate imbalance: Using SMOTE")
        else:
            method = ADASYN(sampling_strategy='auto', random_state=42)
            print("Mild imbalance: Using ADASYN")
    else:
        method = {
            'smote': SMOTE(sampling_strategy='auto', random_state=42),
            'adasyn': ADASYN(sampling_strategy='auto', random_state=42),
            'smoteenn': SMOTEENN(random_state=42)
        }[strategy]
        print(f"Using {strategy.upper()} (user-specified)")

    if n_samples <= 50000:
        print("Processing full dataset (<=50K samples)")
        try:
            X_res, y_res = method.fit_resample(X_train, y_train)
        except ValueError as e:
            print(f"Resampling failed: {e}. Falling back to SMOTE")
            X_res, y_res = SMOTE(sampling_strategy='auto').fit_resample(X_train, y_train)
    else:
        batch_size = 10000
        print(f"Processing in mini-batches (>{50000} samples, batch_size={batch_size})")

        X_res, y_res = [], []
        for i in range(0, n_samples, batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            if len(np.unique(y_batch)) >= 2:
                try:
                    X_rs, y_rs = method.fit_resample(X_batch, y_batch)
                    X_res.append(X_rs)
                    y_res.append(y_rs)
                except:
                    X_res.append(X_batch)
                    y_res.append(y_batch)
            else:
                X_res.append(X_batch)
                y_res.append(y_batch)

        X_res = np.vstack(X_res)
        y_res = np.concatenate(y_res)

    print(f"Resampling complete. Before: {n_samples} samples, After: {len(X_res)} samples")
    print(f"New class distribution: {Counter(y_res)}")
    return X_res, y_res


def evaluate_model(y_true, y_pred, y_scores, model_name=""):
    """Comprehensive evaluation for binary classification models"""
    print(f"\n{'=' * 50}\nEvaluation for {model_name}\n{'=' * 50}")

    print("\nStandard Classification Report:")
    print(classification_report(y_true, y_pred))

    print("\nImbalanced Classification Report:")
    print(classification_report_imbalanced(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=["Predicted Normal", "Predicted Fraud"],
                yticklabels=["Actual Normal", "Actual Fraud"])
    plt.title(f"Confusion Matrix Without TDA - {model_name}")
    plt.savefig(f"Confusion_Matrix_{model_name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, label=f'AP={ap_score:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.savefig(f"Precision_Recall_Curve_{model_name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(f"ROC_Curve_{model_name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("\nKey Metrics:")
    print(f"- AUC-ROC: {roc_auc:.4f}")
    print(f"- Average Precision: {ap_score:.4f}")
    print(f"- Fraud Detection Rate (Recall): {recall_score(y_true, y_pred, pos_label=1):.2%}")
    print(f"- Precision: {precision_score(y_true, y_pred, pos_label=1):.2%}")
    print(f"- F1 Score: {f1_score(y_true, y_pred, pos_label=1):.4f}")

    return {
        'model_name': model_name,
        'roc_auc': roc_auc,
        'average_precision': ap_score,
        'recall': recall_score(y_true, y_pred, pos_label=1),
        'precision': precision_score(y_true, y_pred, pos_label=1),
        'f1': f1_score(y_true, y_pred, pos_label=1)
    }


def run_xgboost(X_train, X_test, y_train, y_test):
    """XGBoost model implementation"""
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='aucpr',
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    results = evaluate_model(y_test, y_pred, y_scores, "XGBoost")
    return model, results


def run_random_forest(X_train, X_test, y_train, y_test):
    """Random Forest model implementation"""
    class_weight = {0: 1, 1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])}

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    results = evaluate_model(y_test, y_pred, y_scores, "Random Forest")
    return model, results


def get_batch_size(n_samples):
    """Dynamically sets batch size based on dataset size"""
    if n_samples <= 50_000:
        return 64
    elif n_samples <= 100_000:
        return 128
    elif n_samples <= 500_000:
        return 256
    else:
        return 512


def run_neural_network(X_train, X_test, y_train, y_test):
    """Neural Network model implementation"""
    batch_size = get_batch_size(len(X_train))
    epochs = max(15, min(100, int(200_000 / len(X_train) * 100)))

    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=[
            'AUC',
            tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
            'Precision',
            'Recall'
        ]
    )

    early_stop = EarlyStopping(
        monitor='val_pr_auc',
        patience=5,
        mode='max',
        min_delta=0.001,
        restore_best_weights=True
    )

    reduce_LR = ReduceLROnPlateau(
        monitor='val_pr_auc',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_LR],
        verbose=1
    )

    y_scores = model.predict(X_test).flatten()
    y_pred = (y_scores >= 0.5).astype(int)

    results = evaluate_model(y_test, y_pred, y_scores, "Neural Network")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['pr_auc'], label='Train PR AUC')
    plt.plot(history.history['val_pr_auc'], label='Validation PR AUC')
    plt.title('Precision-Recall AUC History')
    plt.ylabel('PR AUC')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"NN_Training_History_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

    return model, results


def compare_models(X_train, X_test, y_train, y_test):
    """Run and compare multiple models"""
    model_functions = {
        'XGBoost': run_xgboost,
        'Random Forest': run_random_forest,
        'Neural Network': run_neural_network
    }

    results = []
    models = {}

    for name, func in model_functions.items():
        print(f"\n{'=' * 50}\nRunning {name}\n{'=' * 50}")
        model, metrics = func(X_train, X_test, y_train, y_test)
        models[name] = model
        results.append(metrics)

    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df[['model_name', 'roc_auc', 'average_precision', 'recall', 'precision', 'f1']])

    plt.figure(figsize=(12, 6))
    results_df.set_index('model_name')[['roc_auc', 'average_precision', 'f1']].plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("model_comparison_no_tda.png", dpi=300, bbox_inches='tight')
    plt.show()

    return models, results_df


def generate_summary_report(models, results_df, X_test, y_test):
    """Generate summary report of model performance"""
    print("\n" + "=" * 50)
    print("FRAUD DETECTION SUMMARY REPORT (NO TDA)")
    print("=" * 50 + "\n")

    print(f"• Test Set Transactions: {len(X_test):,}")
    print(f"• Fraud Rate: {y_test.mean():.2%}")
    print(f"• Baseline Accuracy: {max(y_test.mean(), 1 - y_test.mean()):.2%}\n")

    print(results_df[['model_name', 'roc_auc', 'recall', 'precision', 'f1']])
    print("\n")

    best_idx = results_df['roc_auc'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model_name']
    best_model = models[best_model_name]

    if hasattr(best_model, 'predict_proba'):
        y_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        y_pred = best_model.predict(X_test)
        y_proba = None

    print(f"Best Model: {best_model_name}")
    print(f"- AUC-ROC: {roc_auc_score(y_test, y_proba if y_proba is not None else y_pred):.4f}")
    print(f"- Recall: {recall_score(y_test, y_pred):.2%}")
    print(f"- Precision: {precision_score(y_test, y_pred):.2%}")
    print(f"- F1: {f1_score(y_test, y_pred):.4f}")
    if y_proba is not None:
        print(f"- Avg Precision: {average_precision_score(y_test, y_proba):.4f}")

    return best_model


# Main Execution
if __name__ == "__main__":
    check_memory(1)  # Verify sufficient memory

    # Load dataset
    # df = pd.read_csv("Base.csv")
    #df = pd.read_csv("Base_Modified 5% Fraud_95% Legit_20,000.csv")
    #df = pd.read_csv("Base_Modified 10% Fraud_90% Legit_20,000.csv")
    #df = pd.read_csv("Base_Modified 25% Fraud_75% Legit_20,000.csv")
    df = pd.read_csv("Base_Modified 50% Fraud_50% Legit_20,000.csv")

    # Selected features (same as original but without TDA-specific features)
    selected_cols = [
        "income", "credit_risk_score", "velocity_6h", "zip_count_4w", "intended_balcon_amount", "fraud_bool",
        "name_email_similarity", "prev_address_months_count", "current_address_months_count", "customer_age",
        "days_since_request", "payment_type", "velocity_24h", "velocity_4w", "bank_branch_count_8w",
        "date_of_birth_distinct_emails_4w", "employment_status", "email_is_free", "housing_status",
        "phone_home_valid", "phone_mobile_valid", "bank_months_count", "has_other_cards",
        "proposed_credit_limit", "foreign_request", "source", "session_length_in_minutes", "device_os",
        "keep_alive_session", "device_distinct_emails_8w", "device_fraud_count", "month"
    ]

    # Prepare data
    df.columns = df.columns.str.strip()
    df = df[selected_cols].copy()
    target_col = "fraud_bool"

    # Feature engineering
    df_features = df.drop(columns=[target_col]).copy()
    df_target = df[target_col].copy()
    df_features.replace(-1, np.nan, inplace=True)

    # Missing value handling
    missing_cols = [col for col in df_features.columns if df_features[col].isnull().sum() > 0]
    if missing_cols:
        for col in missing_cols:
            df_features[col + "_missing"] = df_features[col].isnull().astype(int)

    # Feature engineering (same as original)
    df_features["income_to_credit_ratio"] = df_features["income"] / (df_features["credit_risk_score"] + 1)
    df_features["velocity_to_transactions_ratio"] = df_features["velocity_6h"] / (df_features["zip_count_4w"] + 1)

    # Log transforms
    df_features[["income", "credit_risk_score", "velocity_6h"]] = df_features[
        ["income", "credit_risk_score", "velocity_6h"]].map(lambda x: x if x > 0 else 1e-5)
    df_features[["income_log", "credit_log", "velocity_6h_log"]] = np.log(
        df_features[["income", "credit_risk_score", "velocity_6h"]])

    df_features["device_fraud_ratio"] = df_features["device_fraud_count"] / (
                df_features["session_length_in_minutes"] + 1)
    df_features['prev_address_months_count'].fillna(-1, inplace=True)
    df_features['prev_address_missing'] = df_features['prev_address_months_count'] == -1
    df_features['velocity_ratio'] = df_features['velocity_6h'] / (df_features['velocity_24h'] + 1e-5)

    # Encode categorical variables
    categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_features[col] = le.fit_transform(df_features[col].astype(str))
        label_encoders[col] = le

    # Final data preparation
    df_features.fillna(-9999, inplace=True)
    df_sample = df.sample(n=20000, random_state=42) if len(df) > 20000 else df.copy()
    X_sample = df_features.loc[df_sample.index].values
    y_sample = df_target.loc[df_sample.index].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_sample, test_size=0.3, random_state=42, stratify=y_sample
    )

    # Resample training data
    X_res, y_res = hybrid_resampling(X_train, y_train)
    X_res = X_res.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Train and compare models
    trained_models, results_df = compare_models(X_res, X_test, y_res, y_test)

    # Generate summary
    best_model = generate_summary_report(trained_models, results_df, X_test, y_test)

    # Save results
    results_df.to_csv("model_comparison_results_no_tda.csv", index=False)
    logging.info("Non-TDA pipeline completed successfully.")