"""
Segmentation pipeline for the Census income project.

This script separates the unsupervised customer segmentation section from the
classification task. It follows the notebook logic:
1. Load and preprocess the data.
2. Exclude sensitive demographic variables from the main segmentation feature set.
3. One-hot encode categorical features and scale numerical features.
4. Reduce dimensionality with TruncatedSVD.
5. Evaluate K-Means using inertia and silhouette score.
6. Fit the final K-Means model and produce segment profiles.

Run example:
python segmentation_model.py \
    --data census-bureau.data \
    --columns census-bureau.columns \
    --output-dir outputs_segmentation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import ensure_output_dir, prepare_model_dataframe


SEGMENTATION_FEATURES = [
    "age",
    "education_ordinal",
    "major industry code",
    "major occupation code",
    "class of worker",
    "full or part time employment stat",
    "weeks worked in year",
    "num persons worked for employer",
    "marital_group",
    "detailed household summary in household",
    "own business or self employed",
    "wage per hour_log",
    "capital gains_is_nonzero",
    "capital losses_is_nonzero",
    "dividends from stocks_is_nonzero",
]

CATEGORICAL_PROFILE_COLS = [
    "education",
    "major occupation code",
    "major industry code",
    "class of worker",
    "full or part time employment stat",
    "marital_group",
    "detailed household summary in household",
]


def build_segmentation_matrix(df_model: pd.DataFrame, n_components: int = 30):
    """Prepare segmentation features, preprocess them, and reduce dimensions."""
    available_features = [col for col in SEGMENTATION_FEATURES if col in df_model.columns]
    df_segment = df_model[available_features].copy()

    numeric_features = df_segment.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = df_segment.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X_processed = preprocessor.fit_transform(df_segment)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X_processed)

    print("Segmentation features used:")
    for feature in available_features:
        print("-", feature)
    print("Explained variance from SVD:", svd.explained_variance_ratio_.sum())

    return df_segment, X_reduced, preprocessor, svd


def evaluate_kmeans_range(X_reduced, output_dir: Path, min_k: int = 2, max_k: int = 8) -> pd.DataFrame:
    """Evaluate K-Means models by inertia and silhouette score."""
    rows = []
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_reduced)
        rows.append(
            {
                "k": k,
                "inertia": kmeans.inertia_,
                "silhouette_score": silhouette_score(X_reduced, labels),
            }
        )

    k_results = pd.DataFrame(rows)
    k_results.to_csv(output_dir / "kmeans_k_evaluation.csv", index=False)
    print("\nK-Means evaluation:")
    print(k_results)

    # Elbow plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_results["k"], k_results["inertia"], marker="o")
    plt.xlabel("Number of clusters, k")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for K-Means")
    plt.tight_layout()
    plt.savefig(output_dir / "kmeans_elbow_plot.png", dpi=300)
    plt.close()

    # Silhouette plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_results["k"], k_results["silhouette_score"], marker="o")
    plt.xlabel("Number of clusters, k")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score by Number of Clusters")
    plt.tight_layout()
    plt.savefig(output_dir / "kmeans_silhouette_plot.png", dpi=300)
    plt.close()

    return k_results


def top_categories_by_segment(
    df: pd.DataFrame,
    segment_col: str,
    category_col: str,
    top_n: int = 5,
) -> pd.DataFrame:
    result = (
        df.groupby(segment_col)[category_col]
        .value_counts(normalize=True)
        .rename("percentage")
        .reset_index()
    )
    result["percentage"] = result["percentage"] * 100
    result = (
        result.sort_values([segment_col, "percentage"], ascending=[True, False])
        .groupby(segment_col)
        .head(top_n)
    )
    return result


def create_segment_profiles(df_model: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Create numeric and categorical profile summaries for each segment."""
    segment_profile = (
        df_model.groupby("segment")
        .agg(
            count=("label", "count"),
            over_50k_rate=("label", "mean"),
            avg_age=("age", "mean"),
            avg_education=("education_ordinal", "mean"),
            avg_weeks_worked=("weeks worked in year", "mean"),
            avg_wage=("wage per hour", "mean"),
            capital_gain_rate=("capital gains_is_nonzero", "mean"),
            dividend_rate=("dividends from stocks_is_nonzero", "mean"),
        )
        .reset_index()
    )

    percent_cols = ["over_50k_rate", "capital_gain_rate", "dividend_rate"]
    for col in percent_cols:
        segment_profile[col] = segment_profile[col] * 100

    segment_profile.to_csv(output_dir / "segment_numeric_profile.csv", index=False)
    print("\nSegment numeric profile:")
    print(segment_profile)

    all_category_profiles: List[pd.DataFrame] = []
    for col in CATEGORICAL_PROFILE_COLS:
        if col not in df_model.columns:
            continue
        top_df = top_categories_by_segment(df_model, "segment", col, top_n=5)
        top_df.insert(0, "profile_column", col)
        all_category_profiles.append(top_df)

        pivot_df = top_df.pivot(index=col, columns="segment", values="percentage").fillna(0)
        plt.figure(figsize=(12, 6))
        pivot_df.plot(kind="bar", figsize=(12, 6))
        plt.ylabel("Percentage within Segment")
        plt.xlabel(col)
        plt.title(f"Top 5 {col} Categories by Segment")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Segment")
        plt.tight_layout()
        safe_name = col.replace(" ", "_").replace("/", "_").replace("'", "")
        plt.savefig(output_dir / f"segment_top_categories_{safe_name}.png", dpi=300)
        plt.close()

    if all_category_profiles:
        category_profile = pd.concat(all_category_profiles, axis=0, ignore_index=True)
        category_profile.to_csv(output_dir / "segment_top_category_profiles.csv", index=False)

    return segment_profile


def assign_business_labels(segment_profile: pd.DataFrame) -> pd.DataFrame:
    """Add simple business-oriented labels based on profile patterns."""
    labeled = segment_profile.copy()

    def label_row(row):
        if row["avg_age"] < 18:
            return "Children / Dependents"
        if row["over_50k_rate"] >= 20:
            return "High-Income Professional Segment"
        if row["avg_weeks_worked"] >= 35 and row["over_50k_rate"] >= 8:
            return "Working Adult Segment"
        if row["avg_weeks_worked"] < 10 and row["avg_age"] >= 50:
            return "Older / Low Employment Segment"
        return "General Low-to-Mid Income Segment"

    labeled["business_label"] = labeled.apply(label_row, axis=1)
    return labeled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="census-bureau.data")
    parser.add_argument("--columns", default="census-bureau.columns")
    parser.add_argument("--output-dir", default="outputs_segmentation")
    parser.add_argument("--n-components", type=int, default=30)
    parser.add_argument("--final-k", type=int, default=4, help="Final number of clusters. Notebook used four segments.")
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output_dir)

    df_model = prepare_model_dataframe(args.data, args.columns)

    # Race, sex, and hispanic origin are intentionally not included in SEGMENTATION_FEATURES
    # because they are sensitive demographic features and should not drive the main segmentation model.
    _, X_reduced, _, _ = build_segmentation_matrix(df_model, n_components=args.n_components)
    evaluate_kmeans_range(X_reduced, output_dir, min_k=2, max_k=8)

    kmeans = KMeans(n_clusters=args.final_k, random_state=42, n_init=10)
    df_model["segment"] = kmeans.fit_predict(X_reduced)

    segment_profile = create_segment_profiles(df_model, output_dir)
    labeled_profile = assign_business_labels(segment_profile)
    labeled_profile.to_csv(output_dir / "segment_profile_with_business_labels.csv", index=False)

    print("\nSegment profile with business labels:")
    print(labeled_profile)
    print("\nSaved segmentation outputs to:", output_dir)


if __name__ == "__main__":
    main()
