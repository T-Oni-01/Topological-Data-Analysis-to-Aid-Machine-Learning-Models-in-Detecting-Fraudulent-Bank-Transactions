# My Regular Code
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import persim
import seaborn as sns
import umap
import logging
import numpy as np
import tensorflow as tf
import kmapper as km
import gc
from persim import PersistenceImager
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gtda.homology import VietorisRipsPersistence
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, average_precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, precision_recall_curve, auc
from sklearn.base import clone
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from autogluon.tabular import TabularPredictor
import keras_tuner as kt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from joblib import Parallel, delayed
from scipy.stats import entropy as scipy_entropy
from scipy.stats import entropy


import psutil #For Memory Management
def check_memory(required_gb):
    """Check if sufficient memory is available"""
    available = psutil.virtual_memory().available / (1024**3)
    if available < required_gb:
        raise MemoryError(f"Need {required_gb}GB RAM, only {available:.2f}GB available")
    print(f"✓ {available:.2f}GB RAM available (required: {required_gb}GB)")


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Defined Functions
def compute_persistence_statistics(diagrams):
    """Compute summary statistics for each homology dimension separately"""
    stats = []
    for diagram in diagrams:
        dim_stats = []
        for dim in [0, 1, 2]:
            dim_diagram = diagram[diagram[:, 2] == dim]
            if len(dim_diagram) == 0:
                dim_stats.extend([0, 0, 0])  # mean, std, sum
            else:
                lifetimes = dim_diagram[:, 1] - dim_diagram[:, 0]
                dim_stats.extend([
                    np.mean(lifetimes),
                    np.std(lifetimes) if len(lifetimes) > 1 else 0,
                    np.sum(lifetimes)
                ])
        stats.append(dim_stats)
    return np.array(stats)


def compute_persistence_entropy(diagrams):
    """Compute entropy for each homology dimension separately"""
    entropies = []
    for diagram in diagrams:
        dim_entropies = []
        for dim in [0, 1, 2]:  # For each homology dimension
            dim_diagram = diagram[diagram[:, 2] == dim]
            if len(dim_diagram) == 0:
                dim_entropies.append(0)
            else:
                lifetimes = dim_diagram[:, 1] - dim_diagram[:, 0]
                total = np.sum(lifetimes)
                if total == 0:  # Handle zero-division case
                    dim_entropies.append(0)
                else:
                    probs = lifetimes / total
                    dim_entropies.append(entropy(probs))
        entropies.append(dim_entropies)
    return np.array(entropies)


def compute_persistence_images(diagrams, resolution=20, sigma=0.1):
    """Robust persistence image computation"""
    pimgr = PersistenceImager(pixel_size=0.2, kernel_params={'sigma': sigma})
    pimgr.fit(diagrams)

    images = []
    for diagram in diagrams:
        if len(diagram) > 0:
            # Only use (birth, death) pairs
            img = pimgr.transform(diagram[:, :2])
        else:
            img = np.zeros((resolution, resolution))
        images.append(img.flatten())

    return np.array(images)


def extract_features_from_persistence_diagrams(diagrams, pixel_size=0.2):
    """Robust topological feature extraction with proper error handling"""
    # Initialize outputs with empty arrays
    stats_features = []
    entropy_features = []  # Renamed to avoid conflict

    for diagram in diagrams:
        # Process each homology dimension
        dim_stats = []
        dim_entropy = []

        for dim in [0, 1]:  # Just H0 and H1
            dim_dgm = diagram[diagram[:, 2] == dim]

            # Statistics
            lifetimes = dim_dgm[:, 1] - dim_dgm[:, 0] if len(dim_dgm) > 0 else [0]
            dim_stats.extend([
                np.mean(lifetimes),
                np.std(lifetimes) if len(lifetimes) > 1 else 0,
                np.sum(lifetimes)
            ])

            # Entropy features
            if len(dim_dgm) > 0:
                birth_hist = np.histogram(dim_dgm[:, 0], bins=5)[0]
                life_hist = np.histogram(lifetimes, bins=5)[0]
                dim_entropy.extend([
                    scipy_entropy(birth_hist + 1e-10),  # Using renamed import
                    scipy_entropy(life_hist + 1e-10)
                ])
            else:
                dim_entropy.extend([0, 0])

        stats_features.append(dim_stats)
        entropy_features.append(dim_entropy)

    # Convert to arrays
    stats_array = np.array(stats_features)
    entropy_array = np.array(entropy_features)

    # Simple persistence images fallback
    persistence_images = np.zeros((len(diagrams), 1))

    return stats_array, entropy_array, persistence_images


def compute_persistence_diagrams(X, max_edge_length=np.inf, homology_dimensions=[0, 1]):
    """Fixed diagram computation with proper input reshaping"""
    # Ensure correct input shape (n_samples, n_points, n_features)
    if X.ndim == 2:
        X = X.reshape(1, *X.shape) if len(X) == X.shape[0] else X.reshape(X.shape[0], -1, 1)

    # Auto-determine max edge length
    if max_edge_length == np.inf:
        from sklearn.metrics import pairwise_distances
        sample = X[0] if X.ndim == 3 else X
        dists = pairwise_distances(sample[:1000])  # Sample first 1000 points
        max_edge_length = np.percentile(dists[dists > 0], 95)  # Exclude zeros
        print(f"Auto-computed max edge length: {max_edge_length:.2f}")

    VR = VietorisRipsPersistence(
        metric="euclidean",
        homology_dimensions=homology_dimensions,
        max_edge_length=max_edge_length,
        collapse_edges=True,
        n_jobs=-1
    )
    return VR.fit_transform(X)


def compute_umap_embedding(X, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
   # Will compute UMAP embedding in mini-batches if dataset > 50K.
    n_samples = X.shape[0]

    if n_samples <= 50000:
        # Full-batch UMAP
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )
        embedding = reducer.fit_transform(X)
        print("Full-batch UMAP computed.")

    else:
        # Mini-batch UMAP (approximate)
        batch_size = 10000
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            init='random'  # Critical for mini-batch
        )

        # Warm-up with a small subset
        reducer.fit(X[:5000])

        # Process in chunks
        embedding = np.zeros((n_samples, n_components))
        for i in tqdm(range(0, n_samples, batch_size), desc="UMAP Mini-Batch"):
            batch = X[i:i + batch_size]
            embedding[i:i + batch_size] = reducer.transform(batch)

        print("Mini-batch UMAP computed.")

    return embedding


def hybrid_resampling(X_train, y_train, strategy='auto', balance_threshold=0.4, n_jobs=-1):
    # tHis will ComBine my last smart resampling + mini-batch functions.
    # If you wanna disable parallel processing, just set n_jobs =1)
    n_samples = X_train.shape[0]
    class_counts = Counter(y_train)
    minority_ratio = min(class_counts.values()) / n_samples

    # Skip if data is balanced enough
    if minority_ratio >= balance_threshold:
        print(f"Data is balanced (minority ratio = {minority_ratio:.1%}). Skipping resampling.")
        return X_train, y_train

    # Select resampling method
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

    # Mini-batch processing for large datasets
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

        def process_batch(i):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            if len(np.unique(y_batch)) >= 2:  # Only resample if both classes exist
                try:
                    return method.fit_resample(X_batch, y_batch)
                except:
                    return X_batch, y_batch
            return X_batch, y_batch

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_batch)(i) for i in range(0, n_samples, batch_size)
        )
        X_res = np.vstack([r[0] for r in results])
        y_res = np.concatenate([r[1] for r in results])

    # Verify output shape
    print(f"Resampling complete. Before: {n_samples} samples, After: {len(X_res)} samples")
    print(f"New class distribution: {Counter(y_res)}")
    return X_res, y_res

def generate_kmapper_visualizations(X_umap, X_sampled, y_sample):
    import kmapper as km
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler

    # Initialize KMapper
    mapper = km.KeplerMapper(verbose=0)

    # Prepare lens data (using your UMAP results)
    lens = MinMaxScaler().fit_transform(X_umap)

    # Create the graph
    graph = mapper.map(
        lens,
        X=X_sampled,
        clusterer=KMeans(n_clusters=8, n_init='auto'),
        cover=km.Cover(n_cubes=10, perc_overlap=0.3)
    )

    # Generate HTML visualization
    html_path = "fraud_topological_network.html"
    mapper.visualize(
        graph,
        path_html=html_path,
        title="Fraud Detection Topology",
        custom_tooltips=y_sample,  # Shows values when hovering
        color_values=y_sample,
        color_function_name="Fraud Label",
        node_color_function="mean"
    )
    import webbrowser
    webbrowser.open(html_path)
    return html_path
# -------_-------:)--------
# Model Definitions


def evaluate_model(y_true, y_pred, y_scores, model_name=""):
    """Comprehensive evaluation for binary classification models"""
    print(f"\n{'=' * 50}\nEvaluation for {model_name}\n{'=' * 50}")

    # Classification reports
    print("\nStandard Classification Report:")
    print(classification_report(y_true, y_pred))

    print("\nImbalanced Classification Report:")
    print(classification_report_imbalanced(y_true, y_pred))

    # Confusion matrix with percentages
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=["Predicted Normal", "Predicted Fraud"],
                yticklabels=["Actual Normal", "Actual Fraud"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"Confusion Matrix_{model_name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Precision-Recall Curve
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

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(f"ROC Curve_{model_name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Key metrics
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
    """XGBoost model implementation with comprehensive evaluation"""
    # Calculate class weight ratio
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    xgb_params = {
        'scale_pos_weight': [1, scale_pos_weight, 10],  # Test different weights
    }

    # Initialize XGBoost
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

    # Train model
    model.fit(X_train, y_train)

    # Get predictions
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    # Evaluate
    results = evaluate_model(y_test, y_pred, y_scores, "XGBoost")

    return model, results


def run_random_forest(X_train, X_test, y_train, y_test):
    """Random Forest model implementation"""
    # Calculate class weights
    class_weight = {0: 1, 1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])}

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )

    # Train model
    model.fit(X_train, y_train)

    # Get predictions
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    # Evaluate
    results = evaluate_model(y_test, y_pred, y_scores, "Random Forest")

    return model, results

#This will help the neural network model adjust for datset size change
def get_batch_size(n_samples):
    """Dynamically sets batch size based on dataset size"""
    if n_samples <= 50_000:
        return 64
    elif n_samples <= 100_000:
        return 128
    elif n_samples <= 500_000:
        return 256
    else:
        return 512  # For 500K–1M samples

def run_neural_network(X_train, X_test, y_train, y_test):
    """Neural Network model implementation with class weighting"""

    # Dynamic batch size
    batch_size = get_batch_size(len(X_train))

    # Adjust epochs inversely with dataset size
    base_epochs = 100
    epochs = max(15, min(base_epochs, int(200_000 / len(X_train) * base_epochs)))

    # Learning rate adjustment
    learning_rate = 0.001 if len(X_train) > 50_000 else 0.0005

    # Class weights for imbalance
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))

    # Model architecture (scaled for larger data)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),  # Increased from 0.3
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.5),  # Increased from 0.2
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Reduced from 0.001
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    early_stop = EarlyStopping(
        monitor='val_pr_auc',
        patience=5,
        mode='max',
        min_delta=0.001,  # Minimum improvement required
        restore_best_weights=True
    )

    reduce_LR = ReduceLROnPlateau(
        monitor='val_pr_auc',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    # Calculate weights
    class_weights = get_class_weights(y_res)

    history = model.fit(
        X_train_enhanced, y_res,
        validation_data=(X_test_enhanced, y_test),
        class_weight=class_weights,  # <-- This is critical
        callbacks=callbacks,
        epochs=100,
        batch_size=batch_size
    )

    sample_weights = np.where(y_train == 1, 10.0, 1.0)  # Upweight fraud samples

    # Get predictions
    y_scores = model.predict(X_test).flatten()
    y_pred = (y_scores >= 0.5).astype(int)

    # 1. Get proper feature indices
    tda_start_idx = X_res.shape[1]

    # 2. Evaluate each model properly
    for name, model in trained_models.items():
        print(f"\n=== {name} ===")
        if name == "Neural Network (Tuned)":
            y_scores = model.predict(X_test_enhanced).flatten()
        else:
            y_scores = model.predict_proba(X_test_enhanced)[:, 1]

        y_pred = (y_scores >= 0.5).astype(int)

    # Evaluate
    #results = evaluate_model(y_test, y_pred, y_scores, "Neural Network")

    results = []
    for name, model in trained_models.items():
        if 'Neural' in name:
            y_scores = model.predict(X_test_enhanced).flatten()
        else:
            y_scores = model.predict_proba(X_test_enhanced)[:, 1]
        y_pred = (y_scores >= 0.5).astype(int)
        results.append(evaluate_model(y_test, y_pred, y_scores, name))

    results_df = pd.concat(results, ignore_index=True)


    fig = plt.figure(figsize=(12, 4))

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
    fig.savefig(f"NN_Training_History_{timestamp}.png",
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.show()

    return model, results

def cross_validate_model(model, X, y, n_splits=5):
    """Perform cross-validation and return metrics"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Define scoring metrics
    scoring = {
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision',
        'f1': 'f1',
        'recall': 'recall',
        'precision': 'precision'
    }

    # Perform cross-validation
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    # Calculate mean scores
    cv_results = {
        'roc_auc_mean': np.mean(scores['test_roc_auc']),
        'roc_auc_std': np.std(scores['test_roc_auc']),
        'ap_mean': np.mean(scores['test_average_precision']),
        'ap_std': np.std(scores['test_average_precision']),
        'f1_mean': np.mean(scores['test_f1']),
        'f1_std': np.std(scores['test_f1']),
        'recall_mean': np.mean(scores['test_recall']),
        'recall_std': np.std(scores['test_recall']),
        'precision_mean': np.mean(scores['test_precision']),
        'precision_std': np.std(scores['test_precision'])
    }

    return cv_results


def compare_models(X_train, X_test, y_train, y_test):
    """Run and compare multiple models with different tuning approaches"""
    model_functions = {
        'XGBoost (Grid Search)': run_xgboost_with_gridsearch,
        'Random Forest (Grid Search)': run_random_forest_with_gridsearch,
        'Neural Network (Keras Tuner)': run_neural_network_with_tuner,

       #Lets not run AutoGluon due to emeory issues
       # 'AutoGluon': lambda X_train, X_test, y_train, y_test:
          #  run_autogluon_automl(X_train, X_test, y_train, y_test, time_limit=1800)  # 30 mins
    }

    results = []
    models = {}

    for name, func in model_functions.items():
        print(f"\n{'=' * 50}\nRunning {name}\n{'=' * 50}")
        model, metrics = func(X_train, X_test, y_train, y_test)
        models[name] = model
        results.append(metrics)

    # Convert results to DataFrame for comparison
    results_df = pd.DataFrame(results)

    # Print comparison
    print("\nModel Comparison:")
    print(results_df[['model_name', 'roc_auc', 'average_precision', 'recall', 'precision', 'f1']])

    # Plot comparison
    plt.figure(figsize=(12, 6))
    results_df.set_index('model_name')[['roc_auc', 'average_precision', 'f1']].plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.gcf().savefig("model_comparison.png", dpi=300, bbox_inches='tight', facecolor='w')
    plt.show()

    return models, results_df


def analyze_tda_feature_importance(model, feature_names, original_feature_count):
    """Safe feature importance analysis with bounds checking"""
    if not hasattr(model, 'feature_importances_'):
        print("Model doesn't support feature importance analysis")
        return None, None

    importances = model.feature_importances_

    # Ensure we don't exceed available feature names
    n_features = min(len(feature_names), len(importances))
    feature_names = feature_names[:n_features]
    importances = importances[:n_features]

    # Split importance
    original_importance = sum(importances[:original_feature_count])
    tda_importance = sum(importances[original_feature_count:])

    print(f"\n=== Feature Importance ===")
    print(f"Original features contribution: {original_importance:.2%}")
    print(f"TDA features contribution: {tda_importance:.2%}")

    # Plot top features safely
    plt.figure(figsize=(12, 6))
    top_n = min(20, n_features)  # Don't try to plot more than available
    sorted_idx = np.argsort(importances)[-top_n:]

    # Filter out-of-bounds indices
    valid_idx = [i for i in sorted_idx if i < len(feature_names)]
    plt.barh(np.array(feature_names)[valid_idx], importances[valid_idx])
    plt.title("Top Features by Importance")
    plt.xlabel("Feature Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    plt.show()

    print("First few feature names:", all_feature_names[:5])
    print("Model feature count:", best_model.n_features_in_)
    print("Feature importance array length:", len(best_model.feature_importances_))
    return original_importance, tda_importance


def visualize_tda_features(tda_features, labels):
    """Better visualization for potentially sparse TDA features"""
    plt.figure(figsize=(15, 10))

    # Find features with some variance
    variances = np.var(tda_features, axis=0)
    valid_features = np.where(variances > 1e-6)[0]

    if len(valid_features) == 0:
        print("No meaningful TDA features to visualize")
        return

    for i, feat_idx in enumerate(valid_features[:5]):  # Show top 5
        plt.subplot(len(valid_features[:5]), 1, i + 1)
        for label in [0, 1]:
            subset = tda_features[labels == label, feat_idx]
            if len(subset) > 0:
                sns.kdeplot(subset, label=f"Class {label}", fill=True)
        plt.title(f"TDA Feature {feat_idx} (σ²={variances[feat_idx]:.2e})")
        plt.legend()

    plt.tight_layout()
    plt.savefig("tda_features_visualization.png", dpi=300)
    plt.show()


def generate_summary_report(models, results_df, X_test_enhanced, y_test, model_comparison=True):
    """
    Robust report generation using enhanced features only

    Args:
        models: dict or list of trained models
        results_df: DataFrame with model metrics
        X_test_enhanced: Enhanced test features (original + TDA)
        y_test: Test labels
        model_comparison: Whether to show model comparison
    """
    try:
        print("\n" + "=" * 50)
        print("FRAUD DETECTION SUMMARY REPORT (ENHANCED FEATURES)")
        print("=" * 50 + "\n")

        # Basic dataset info
        print(f"• Test Set Transactions: {len(X_test_enhanced):,}")
        print(f"• Fraud Rate: {y_test.mean():.2%}")
        print(f"• Baseline Accuracy: {max(y_test.mean(), 1 - y_test.mean()):.2%}\n")

        # Validate inputs
        if not isinstance(results_df, pd.DataFrame):
            raise ValueError("results_df must be a DataFrame")

        if 'roc_auc' not in results_df.columns:
            raise ValueError("results_df must contain 'roc_auc' column")

        # Model comparison table
        if model_comparison:
            print(results_df[['model_name', 'roc_auc', 'recall', 'precision', 'f1']])
            print("\n")

        # Get best model
        best_idx = results_df['roc_auc'].idxmax()
        best_model_name = results_df.loc[best_idx, 'model_name']

        # Handle both dict and list model storage
        if isinstance(models, dict):
            best_model = models.get(best_model_name)
        else:
            model_names = results_df['model_name'].values
            try:
                best_model = models[model_names.tolist().index(best_model_name)]
            except (ValueError, IndexError):
                best_model = None

        if best_model is None:
            raise ValueError(f"Best model '{best_model_name}' not found")

        # Generate predictions using enhanced features
        if hasattr(best_model, 'predict_proba'):
            y_proba = best_model.predict_proba(X_test_enhanced)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
        else:
            y_pred = best_model.predict(X_test_enhanced)
            y_proba = None

        # Key metrics
        print(f"Best Model: {best_model_name}")
        print(f"- AUC-ROC: {roc_auc_score(y_test, y_proba if y_proba is not None else y_pred):.4f}")
        print(f"- Recall: {recall_score(y_test, y_pred):.2%}")
        print(f"- Precision: {precision_score(y_test, y_pred):.2%}")
        print(f"- F1: {f1_score(y_test, y_pred):.4f}")
        if y_proba is not None:
            print(f"- Avg Precision: {average_precision_score(y_test, y_proba):.4f}")

        return best_model

    except Exception as e:
        print(f"\n⚠️ Error generating report: {str(e)}")
        print("\nBasic Results:")
        print(results_df[['model_name', 'roc_auc', 'recall', 'precision']])
        return None

def save_model_report(best_model, X_test_enhanced, y_test, filename):
    """Save model performance report using enhanced features"""
    with open(filename, 'w') as f:
        # Get predictions
        y_pred = best_model.predict(X_test_enhanced)
        y_proba = best_model.predict_proba(X_test_enhanced)[:, 1] if hasattr(best_model, 'predict_proba') else None

        # Write report contents
        f.write(f"Model Type: {type(best_model).__name__}\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("=== Performance Metrics ===\n")
        f.write(f"AUC-ROC: {roc_auc_score(y_test, y_proba if y_proba is not None else y_pred):.4f}\n")
        f.write(f"Recall (Fraud Detection Rate): {recall_score(y_test, y_pred):.2%}\n")
        f.write(f"Precision: {precision_score(y_test, y_pred):.2%}\n")
        f.write(f"F1 Score: {f1_score(y_test, y_pred):.4f}\n")

        if y_proba is not None:
            f.write(f"Average Precision: {average_precision_score(y_test, y_proba):.4f}\n")

        # Confusion matrix
        f.write("\n=== Confusion Matrix ===\n")
        cm = confusion_matrix(y_test, y_pred)
        f.write(np.array2string(cm, separator='\t'))

        # Feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            f.write("\n\n=== Top 10 Features ===\n")
            importances = best_model.feature_importances_
            tda_importance = sum(importances[len(final_tda_features):])
            print(f"TDA features contribute {tda_importance * 100:.1f}% of total importance")
            features = X_test.columns if hasattr(X_test, 'columns') else [f"Feature_{i}" for i in
                                                                          range(X_test.shape[1])]
            for feat, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"{feat}: {imp:.4f}\n")

        f.write("\n=== Model Parameters ===\n")
        if hasattr(best_model, 'get_params'):
            params = best_model.get_params()
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
        else:
            f.write("Parameter information not available\n")




# AutoGluon Implementation
#Update: I'm Not using AutoGluon because of memory management
# =============================================
""""
def run_autogluon_automl(X_train, X_test, y_train, y_test, time_limit=3600):
   # Run AutoGluon with fixes for the observed issues
    print(f"\n{'='*50}\nRunning AutoGluon (Time Limit: {time_limit//60} mins)\n{'='*50}")

    # Prepare data
    train_data = pd.DataFrame(X_train)
    train_data['target'] = y_train
    test_data = pd.DataFrame(X_test)

    # Clear previous runs
    import shutil
    try:
        shutil.rmtree('autogluon_models')
    except:
        pass

    # AutoGluon predictor configuration
    predictor = TabularPredictor(
        label='target',
        problem_type='binary',
        eval_metric='roc_auc',
        path='autogluon_models',
        verbosity=2
    )

    # Fit with simplified settings
    try:
        predictor.fit(
            train_data,
            presets='best_quality',
            time_limit=time_limit,
            hyperparameters={
                'GBM': {},
                'XGB': {},
                'CAT': {},
                'RF': {'criterion': 'gini', 'max_depth': 15}
            },
            num_bag_sets=1,
            num_bag_folds=3,
            num_stack_levels=1
        )

        # Get predictions
        y_scores = predictor.predict_proba(test_data, as_multiclass=False)
        y_pred = predictor.predict(test_data)

        # Evaluate
        results = evaluate_model(y_test, y_pred, y_scores, "AutoGluon")
        return predictor, results

    except Exception as e:
        print(f"AutoGluon failed: {str(e)}")
        # Fallback to simple model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_scores = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        results = evaluate_model(y_test, y_pred, y_scores, "Fallback RF")
        return model, results
"""


# =============================================
# Grid Search for XGBoost and Random Forest
# =============================================

def run_xgboost_with_gridsearch(X_train, X_test, y_train, y_test):
    """XGBoost with comprehensive grid search"""
    scale_pos_weight = len(y_res[y_res == 0]) / len(y_res[y_res == 1])

    # Base model
    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )

    # Expanded parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [0, 0.1],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0, 0.1],
        'min_child_weight': [1, 3]
    }

    # Configure grid search
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=2,
        refit=True
    )

    print("Running XGBoost grid search...")
    grid_search.fit(X_train, y_train)

    # Results
    print("\nGrid Search Results:")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Score (AUC): {grid_search.best_score_:.4f}")

    # Best model
    model = grid_search.best_estimator_

    # Get predictions
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    # Evaluate
    results = evaluate_model(y_test, y_pred, y_scores, "XGBoost (Grid Search)")

    return model, results


def run_random_forest_with_gridsearch(X_train, X_test, y_train, y_test):
    """Random Forest with comprehensive grid search"""
    class_weight = {0: 1, 1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])}

    # Base model
    rf = RandomForestClassifier(
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )

    # Expanded parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }

    # Configure grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=2,
        refit=True
    )

    print("Running Random Forest grid search...")
    check_memory(0.5)  # Check for 1GB available before grid search
    grid_search.fit(X_train, y_train)

    # Results
    print("\nGrid Search Results:")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Score (AUC): {grid_search.best_score_:.4f}")

    # Best model
    model = grid_search.best_estimator_

    # Get predictions
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    # Evaluate
    results = evaluate_model(y_test, y_pred, y_scores, "Random Forest (Grid Search)")

    return model, results


# =============================================
# Neural Network Hyperparameter Tuning with Keras Tuner
# =============================================
def get_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights * 2))  # Double weights for stronger effect


def build_nn_model(hp, input_shape):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))

    # Hidden layers
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            activation=hp.Choice('activation', ['relu', 'tanh', 'elu'])
        ))
        model.add(layers.Dropout(hp.Float(f'dropout_{i}', 0.1, 0.5, step=0.1)))
        model.add(layers.BatchNormalization())

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='log')),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model


def run_neural_network_with_tuner(X_train, X_test, y_train, y_test, max_trials=20, executions_per_trial=1):
    """Neural network hyperparameter tuning with Keras Tuner"""
    print(f"\n{'=' * 50}\nRunning Neural Network Hyperparameter Tuning\n{'=' * 50}")

    # Dynamic batch size based on dataset size
    batch_size = get_batch_size(len(X_train))

    # Class weights for imbalance
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))

    input_shape = (X_train.shape[1],)

    def model_builder(hp):
        return build_nn_model(hp, input_shape)

    # Initialize tuner
    tuner = kt.BayesianOptimization(
        model_builder,
        objective=kt.Objective('val_pr_auc', direction='max'),
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='nn_tuning',
        project_name='fraud_detection',
        overwrite=True
    )

    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_pr_auc',
        patience=5,
        mode='max',
        restore_best_weights=True
    )

    # Search for best hyperparameters
    tuner.search(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1
    )

    # Get best model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.get_best_models(num_models=1)[0]

    print("\nBest Hyperparameters:")
    print(f"Number of layers: {best_hps.get('num_layers')}")
    for i in range(best_hps.get('num_layers')):
        print(f"Layer {i + 1} units: {best_hps.get(f'units_{i}')}")
        print(f"Layer {i + 1} dropout: {best_hps.get(f'dropout_{i}')}")
    print(f"Activation: {best_hps.get('activation')}")
    print(f"Learning rate: {best_hps.get('lr')}")

    # Get predictions
    y_scores = model.predict(X_test).flatten()
    y_pred = (y_scores >= 0.5).astype(int)

    # Evaluate
    results = evaluate_model(y_test, y_pred, y_scores, "Neural Network (Tuned)")

    return model, results




#---------------------------------------_----:)----
# Main Execution
# Load dataset
#df = pd.read_csv("Base.csv")
#df = pd.read_csv("Base_Modified 5% Fraud_95% Legit_20,000.csv")
df = pd.read_csv("Base_Modified 10% Fraud_90% Legit_20,000.csv")
#df = pd.read_csv("Base_Modified 25% Fraud_75% Legit_20,000.csv")
#df = pd.read_csv("Base_Modified 50% Fraud_50% Legit_20,000.csv")

check_memory(0.5)  # Verify 1GB available for entire pipeline

# Updated selected columns based on your dataset
selected_cols = [
    "income", "credit_risk_score", "velocity_6h", "zip_count_4w", "intended_balcon_amount", "fraud_bool",
    "name_email_similarity", "prev_address_months_count", "current_address_months_count", "customer_age",
    "days_since_request", "payment_type", "velocity_24h", "velocity_4w", "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w", "employment_status", "email_is_free", "housing_status",
    "phone_home_valid", "phone_mobile_valid", "bank_months_count", "has_other_cards",
    "proposed_credit_limit", "foreign_request", "source", "session_length_in_minutes", "device_os",
    "keep_alive_session", "device_distinct_emails_8w", "device_fraud_count", "month"
]

# Ensure column names match
df.columns = df.columns.str.strip()
df = df[selected_cols].copy()

# Separate target column 'fraud_bool' from the features
target_col = "fraud_bool"
df_features = df.drop(columns=[target_col]).copy()
df_target = df[target_col].copy()

# Replace `-1` with NaN to correctly detect missing values
df_features.replace(-1, np.nan, inplace=True)

# Print missing value count before processing
print("Missing values per column before processing:\n", df_features.isnull().sum())

# Identify columns with missing values
missing_cols = [col for col in df_features.columns if df_features[col].isnull().sum() > 0]

if missing_cols:
    for col in missing_cols:
        df_features[col + "_missing"] = df_features[col].isnull().astype(int)

# Feature Engineering
df_features["income_to_credit_ratio"] = df_features["income"] / (df_features["credit_risk_score"] + 1)
df_features["velocity_to_transactions_ratio"] = df_features["velocity_6h"] / (df_features["zip_count_4w"] + 1)

# Log transformations
df_features[["income", "credit_risk_score", "velocity_6h"]] = df_features[["income", "credit_risk_score", "velocity_6h"]].map(lambda x: x if x > 0 else 1e-5)
df_features[["income_log", "credit_log", "velocity_6h_log"]] = np.log(df_features[["income", "credit_risk_score", "velocity_6h"]])

df_features["device_fraud_ratio"] = df_features["device_fraud_count"] / (df_features["session_length_in_minutes"] + 1)

# Handle missing values more carefully
df_features['prev_address_months_count'].fillna(-1, inplace=True)  # Special value for missing
df_features['prev_address_missing'] = df_features['prev_address_months_count'] == -1

# Add interaction features
df_features['velocity_ratio'] = df_features['velocity_6h'] / (df_features['velocity_24h'] + 1e-5)


# Encode categorical variables
categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_features[col] = le.fit_transform(df_features[col].astype(str))
    label_encoders[col] = le

# Fill NaNs with a placeholder for missingness detection
df_features.fillna(-9999, inplace=True)

# Remove Highly Correlated Features
correlation_matrix = df_features.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
print("Highly correlated features to drop:", to_drop)

df_features = df_features.drop(columns=to_drop)

# Recombine with target
df = pd.concat([df_features, df_target], axis=1)
df_sample = df.sample(n=20000, random_state=42)  # Modify to change sample size

# Save processed data
df_sample.to_csv("processed_data_no_high_corr.csv", index=False)
print("Processed data saved as 'processed_data_no_high_corr.csv'")

# Split into features (X_sample) and target (y_sample) after sampling
X_sample = df_sample.drop(columns=[target_col])
y_sample = df_sample[target_col].values

# Normalize numerical features (using X_sample)
scaler = StandardScaler()
X_sampled = scaler.fit_transform(X_sample)  # Use X_sample here

#For use in generating my report
feature_names = list(X_sample.columns)  # Original DataFrame columns

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sampled)

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["fraud_bool"] = df_sample[target_col].values

# Apply UMAP
umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
X_umap = compute_umap_embedding(X_sampled)

# Convert to DataFrame for easier plotting
df_umap = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
df_umap["fraud_bool"] = df_sample[target_col].values

# Get explained variance
explained_variance = pca.explained_variance_ratio_
print(f"PCA Explained Variance Ratio: {explained_variance}")

# Create a Biplot
fig, ax = plt.subplots(figsize=(10, 7))

# Scatter plot of PCA-transformed data
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df_sample[target_col], cmap="coolwarm", alpha=0.7)
ax.set_xlabel(f"PC1 ({explained_variance[0]*100:.2f}% Variance)")
ax.set_ylabel(f"PC2 ({explained_variance[1]*100:.2f}% Variance)")
ax.set_title("PCA Biplot")

# Overlay feature vectors
components = pca.components_.T  # Transpose to align with feature axes
feature_names = df_sample.drop(columns=[target_col]).columns

for i, feature in enumerate(feature_names):
    ax.arrow(0, 0, components[i, 0] * 2, components[i, 1] * 2, color='red', alpha=0.5)
    ax.text(components[i, 0] * 2.2, components[i, 1] * 2.2, feature, color='black', fontsize=9)

plt.colorbar(scatter, label="Fraud (0 = Non-Fraud, 1 = Fraud)")
plt.grid()
plt.savefig("pca_biplot.png", dpi=300)
plt.show()

# Compute absolute influence of features on PCA components
pca_feature_importance = np.abs(pca.components_[:2]).sum(axis=0)  # Sum influence from first 2 PCs
feature_names = df_sample.drop(columns=[target_col]).columns

# Sort features by importance
sorted_indices = np.argsort(pca_feature_importance)[::-1]
sorted_features = [feature_names[i] for i in sorted_indices]
sorted_importance = pca_feature_importance[sorted_indices]

# Extract top 15 features from PCA
top_15_pca_features = sorted_features[:15]

# **🔹 t-SNE Analysis**
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_sampled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df_sample[target_col], cmap="coolwarm", alpha=0.7)
plt.colorbar(label="Fraud (0 = Non-Fraud, 1 = Fraud)")
plt.title("t-SNE of Expanded Features (With Missingness Indicator)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.savefig("tsne_expanded_features.png", dpi=300)
plt.show()

# Compute correlation between t-SNE components and original features
X_tsne_df = pd.DataFrame(X_tsne, columns=["t-SNE1", "t-SNE2"])
df_features_reset = df_sample.drop(columns=[target_col]).reset_index(drop=True)

# Compute absolute correlation
correlation_matrix = pd.concat([X_tsne_df, df_features_reset], axis=1).corr()
correlation_scores = correlation_matrix.iloc[:2, 2:].abs().mean(axis=0).sort_values(ascending=False)

# Extract top 15 most correlated features with t-SNE
top_15_tsne_features = correlation_scores.index[:15].tolist()

# Combine and remove duplicates using set()
final_tda_features = list(set(top_15_pca_features + top_15_tsne_features))

logging.info(f"Total unique features selected: {len(final_tda_features)}")
logging.info(f"Final feature list: {final_tda_features}")

# Print number of features before and after removing duplicates
total_before_deduplication = len(top_15_pca_features) + len(top_15_tsne_features)
total_after_deduplication = len(final_tda_features)
duplicates_removed = total_before_deduplication - total_after_deduplication

print(f"Total features before removing duplicates: {total_before_deduplication}")
print(f"Total unique features selected: {total_after_deduplication}")
print(f"Number of duplicate features removed: {duplicates_removed}")

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x=sorted_importance[:15], y=sorted_features[:15], hue=correlation_scores.index[:15], palette="coolwarm", legend=False)
plt.xlabel("Feature Importance Score (PCA)")
plt.ylabel("Features")
plt.title("Top 15 Most Important Features in PCA")
plt.savefig("pca_feature_importance.png", dpi=300)
plt.show()

# Compute cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o", linestyle="-", color="b")
plt.axhline(y=0.95, color="r", linestyle="--", label="95% Variance Threshold")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance (PCA)")
plt.legend()
plt.grid()
plt.savefig("pca_cumulative_variance.png", dpi=300)
plt.show()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix.iloc[:2, 2:], cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation with t-SNE Components")
plt.savefig("tsne_feature_correlation.png", dpi=300)
plt.show()

# Plot the top correlated features with t-SNE components
plt.figure(figsize=(12, 6))
sns.barplot(x=correlation_scores[:15], y=correlation_scores.index[:15], hue=correlation_scores.index[:15], palette="coolwarm",legend=False)
plt.xlabel("Mean Absolute Correlation with t-SNE Components")
plt.ylabel("Features")
plt.title("Top 15 Most Important Features in t-SNE")
plt.savefig("tsne_feature_importance.png", dpi=300)
plt.show()

#Applying U-map
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    df_umap["UMAP1"], df_umap["UMAP2"], c=df_umap["fraud_bool"], cmap="coolwarm", alpha=0.7
)
plt.colorbar(label="Fraud (0 = Non-Fraud, 1 = Fraud)")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.title("UMAP Projection Colored by Fraud Labels")
plt.grid()
plt.savefig("umap_fraud_overlay.png", dpi=300)
plt.show()

html_path = generate_kmapper_visualizations(X_umap, X_sampled, y_sample)
print(f"KMapper visualization saved to {html_path}")

#----------------------------------------------------------
# 1. First split your data (BEFORE any resampling or TDA)
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.3, random_state=42, stratify=y_sample
)


# 2. Apply resampling ONLY to training data
X_res, y_res = hybrid_resampling(X_train, y_train)
print(f"Resampled data shape: {X_res.shape}")
# X_res = X_res.astype(np.float32)
# X_test = X_test.astype(np.float32)

# Option A: IF I WANT SMOTE
# X_res, y_res = hybrid_resampling(X_train, y_train, strategy='smote')
# Option B: ADASYN
# X_res, y_res = hybrid_resampling(X_train, y_train, strategy='adasyn')
# OR Option C: SMOTEENN
# X_res, y_res = hybrid_resampling(X_train, y_train, strategy='smoteenn')

print(f"Before resampling - Fraud cases: {sum(y_train==1)}")
print(f"After resampling - Fraud cases: {sum(y_res==1)}")

# Verify the new class distribution
print(f"Class distribution after resampling: {pd.Series(y_res).value_counts()}")


# ------------------------------------------------------
# **🔹 TDA - Vietoris-Rips Persistence**
# 3. Select features and scale ONLY on resampled data
X_tda = X_res[final_tda_features]
scaler = StandardScaler().fit(X_tda)
X_tda_scaled = scaler.transform(X_tda)

# Reshape for TDA (n_samples, n_points, n_features)
X_tda_reshaped = X_tda_scaled.reshape(len(X_tda_scaled), -1, 1)

# Compute persistence diagrams with auto edge length
print("\nComputing persistence diagrams...")
diagrams = compute_persistence_diagrams(X_tda_reshaped)

# Verify diagrams
print("\nDiagram Statistics:")
for i, dim in enumerate(["H0", "H1"]):
    dim_points = [len(d[d[:,2]==i]) for d in diagrams]
    print(f"{dim}: Avg {np.mean(dim_points):.1f} points per diagram")


# After computing diagrams:
print("\n=== Persistence Diagram Diagnostics ===")
print(f"Total diagrams: {len(diagrams)}")
print(f"First diagram shape: {diagrams[0].shape}")
print("Sample diagram points:")
print(diagrams[0][:3])  # Print first 3 points

# Count non-trivial features
for dim in [0, 1, 2]:
    count = sum(len(d[d[:,2]==dim]) for d in diagrams)
    print(f"Total H{dim} features across all samples: {count}")

# Extract features from persistence diagrams
# Get the features with proper error handling
stats, entropy_features, persistence_images = extract_features_from_persistence_diagrams(diagrams)
tda_features = np.hstack([stats, entropy_features, persistence_images])

# Verify shapes
print("\n=== Feature Shapes ===")
print(f"Stats: {stats.shape}")
print(f"Entropy Features: {entropy_features.shape}")
print(f"Persistence Images: {persistence_images.shape}")

# Handle persistence images carefully
if isinstance(persistence_images, (list, np.ndarray)):
    if isinstance(persistence_images, list):
        # Convert list of arrays to single numpy array
        try:
            persistence_images = np.stack(persistence_images)
        except ValueError as e:
            print(f"Could not stack persistence images: {e}")
            # Fallback to zeros if stacking fails
            persistence_images = np.zeros((len(stats), 1))

    # Ensure 2D shape
    if persistence_images.ndim > 2:
        persistence_images = persistence_images.reshape(len(persistence_images), -1)
    print(f"Persistence images shape: {persistence_images.shape}")
else:
    print(f"Unexpected persistence_images type: {type(persistence_images)}")
    # Create fallback array
    persistence_images = np.zeros((len(stats), 1))

# Safe concatenation
try:
    tda_features = np.hstack([
        stats.reshape(len(stats), -1),
        entropy_features.reshape(len(entropy_features), -1),
        persistence_images.reshape(len(persistence_images), -1)
    ])
    print(f"\nCombined TDA features shape: {tda_features.shape}")
except Exception as e:
    print(f"\nFailed to combine TDA features: {e}")
    # Fallback to just stats and entropy
    tda_features = stats

# Combine with original features with dimension check
if X_res.shape[0] == tda_features.shape[0]:
    X_train_enhanced = np.hstack([X_res, tda_features])
    print(f"Final combined features shape: {X_train_enhanced.shape}")
else:
    print("\n! Shape mismatch between X_res and TDA features !")
    print(f"X_res: {X_res.shape}, TDA: {tda_features.shape}")
    # Align by taking minimum samples
    min_samples = min(X_res.shape[0], tda_features.shape[0])
    X_train_enhanced = np.hstack([
        X_res[:min_samples],
        tda_features[:min_samples]
    ])

# Create feature names with safety checks
original_feature_names = final_tda_features if hasattr(final_tda_features, '__len__') else []
tda_feature_names = []
try:
    tda_feature_names.extend([f"TDA_stat_{i}" for i in range(stats.shape[1])])
    tda_feature_names.extend([f"TDA_entropy_{i}" for i in range(entropy.shape[1])])
    tda_feature_names.extend([f"TDA_img_{i}" for i in range(persistence_images.shape[1])])
except AttributeError:
    print("Could not generate all TDA feature names")
    # Fallback names
    tda_feature_names = [f"TDA_feat_{i}" for i in range(tda_features.shape[1])]

all_feature_names = original_feature_names + tda_feature_names
print(
    f"\nTotal features: {len(all_feature_names)} (Original: {len(original_feature_names)}, TDA: {len(tda_feature_names)})")

# Visualize only if we have meaningful features
if np.var(tda_features) > 1e-6:  # Check for non-zero variance
    print("\nVisualizing TDA features...")
    visualize_tda_features(tda_features, y_res[:len(tda_features)])  # Ensure label alignment
else:
    print("\nSkipping TDA visualization - no meaningful variance in features")

# Combine all TDA features
# 5. Combine TDA features with resampled features
X_train_enhanced = np.hstack([X_res, tda_features])

# Extract H0, H1, and H2 for all samples
H0_all = []  # List to store H0 features for all samples
H1_all = []  # List to store H1 features for all samples
H2_all = []  # List to store H2 features for all samples

for diagram in diagrams:
    # Extract H0, H1, and H2 for the current diagram
    H0 = diagram[diagram[:, -1] == 0]  # H0: Connected components
    H1 = diagram[diagram[:, -1] == 1]  # H1: Loops
    H2 = diagram[diagram[:, -1] == 2]  # H2: Voids

    # Append to the lists
    H0_all.append(H0)
    H1_all.append(H1)
    H2_all.append(H2)

# Convert lists to numpy arrays for easier manipulation
H0_all = np.array(H0_all, dtype=object)
H1_all = np.array(H1_all, dtype=object)
H2_all = np.array(H2_all, dtype=object)

# Example: Print the number of H0, H1, and H2 features for the first sample
print(f"Number of H0 features in the first sample: {len(H0_all[0])}")
print(f"Number of H1 features in the first sample: {len(H1_all[0])}")
print(f"Number of H2 features in the first sample: {len(H2_all[0])}")

print(len(diagrams))  # Should be equal to the number of samples
print(diagrams[0])    # Should print the first transaction's persistence diagram

# Plot Persistence Diagram for the first sample
plt.figure(figsize=(8, 6))

# Plot H0
if len(H0_all[0]) > 0:
    plt.scatter(H0_all[0][:, 0], H0_all[0][:, 1], color='red', label='H0 (Connected Components)', s=50, alpha=0.6)

# Plot H1
if len(H1_all[0]) > 0:
    plt.scatter(H1_all[0][:, 0], H1_all[0][:, 1], color='blue', label='H1 (Loops)', s=50, alpha=0.6)

# Plot H2
if len(H2_all[0]) > 0:
    plt.scatter(H2_all[0][:, 0], H2_all[0][:, 1], color='green', label='H2 (Voids)', s=50, alpha=0.6)

plt.title('Persistence Diagram for the First Sample')
plt.xlabel('Birth')
plt.ylabel('Death')
plt.legend()
plt.grid()
plt.savefig("tda_persistence_diagram_first_sample.png", dpi=300)
plt.show()

# 6. Prepare test data (using same scaler and features)
X_test_tda = scaler.transform(X_test[final_tda_features])
X_test_reshaped = X_test_tda.reshape(len(X_test_tda), -1, 1)
test_diagrams = compute_persistence_diagrams(X_test_reshaped)
test_stats, test_entropy, test_images = extract_features_from_persistence_diagrams(test_diagrams)
test_tda_features = np.hstack([test_stats, test_entropy, test_images])
X_test_enhanced = np.hstack([X_test, test_tda_features])

# Check diagram diversity
unique_diagrams = len(set(tuple(d.flatten()) for d in diagrams))
print(f"Unique diagrams: {unique_diagrams}/{len(diagrams)}")

# Plot distribution of persistence lifetimes
lifetimes = []
for d in diagrams:
    if len(d) > 0:
        lifetimes.extend(d[:,1] - d[:,0])
plt.hist(lifetimes, bins=50)
plt.title("Persistence Lifetimes Distribution")
plt.show()

# For  Updating neural network training with better class weighting
class_weights = get_class_weights(y_res)


#------------------------------------------
#Call Training Model
# Run and compare models
check_memory(0.5)  # Check before training


import shutil
shutil.rmtree('nn_tuning/fraud_detection_48features', ignore_errors=True)

# Train models
trained_models, results_df = compare_models(
        X_train_enhanced, X_test_enhanced, y_res, y_test)

#Generating a summary of results
best_model = generate_summary_report(
    models=trained_models,
    results_df=results_df,
    X_test_enhanced=X_test_enhanced,
    y_test=y_test
)

best_model = trained_models["XGBoost (Grid Search)"]
original_feature_count = len(final_tda_features)
all_feature_names = (
    list(X_res.columns) +  # Original 37 features
    [f"TDA_stat_{i}" for i in range(stats.shape[1])] +
    [f"TDA_entropy_{i}" for i in range(entropy_features.shape[1])] +
    [f"TDA_img_{i}" for i in range(persistence_images.shape[1])]
)

# Verify counts
print(f"Original features: {X_res.shape[1]}")
print(f"TDA features: {tda_features.shape[1]}")
print(f"Total features: {len(all_feature_names)}")
assert len(all_feature_names) == X_train_enhanced.shape[1], "Feature count mismatch"

# Now run analysis
orig_imp, tda_imp = analyze_tda_feature_importance(
    best_model,
    all_feature_names,
    len(final_tda_features)
)


# Add to your evaluation metrics
if orig_imp and tda_imp:
    results_df.loc[results_df['model_name'] == "XGBoost (Grid Search)",
                  'tda_contribution'] = tda_imp



# Save results
results_df.to_csv("model_comparison_results.csv", index=False)
print(f"[INFO] X_test_original shape: {X_test.shape}")
print(f"[INFO] X_test_enhanced shape: {X_test_enhanced.shape}")
save_model_report(
    best_model,
    X_test_enhanced=X_test_enhanced,  # Only pass enhanced features
    y_test=y_test,
    filename="Fraud_Detection_Report.txt"
)
logging.info("Model comparison results saved to 'model_comparison_results.csv'")
logging.info("Report Summary Saved to fraud_detection_report.txt")
logging.info("Anomaly detection pipeline completed successfully.")
logging.info("Hey Taiwo, Go get your saved files")