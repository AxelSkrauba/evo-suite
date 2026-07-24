"""04 - Custom candidates and Pipeline integration.

EvoEnsembleClassifier is a final estimator (not a transformer), so it slots
into a scikit-learn Pipeline as the last step. This example also shows how
to restrict the candidate pool instead of using the built-in defaults.

Run:
    uv run python examples/evo-ens/04_custom_candidates_and_pipeline.py
"""

from __future__ import annotations

from evo_ens import EvoEnsConfig, EvoEnsembleClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "ensemble",
                EvoEnsembleClassifier(
                    candidate_estimators=[
                        RandomForestClassifier(n_estimators=100, random_state=0),
                        GradientBoostingClassifier(n_estimators=100, random_state=0),
                        LogisticRegression(max_iter=500),
                        SVC(kernel="rbf", probability=True),
                    ],
                    config=EvoEnsConfig(
                        population_size=40, n_generations=30, random_seed=0, verbose=False
                    ),
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)

    print(f"Test accuracy: {pipe.score(X_test, y_test):.4f}")
    print(pipe.named_steps["ensemble"].get_ensemble_info().to_string(index=False))


if __name__ == "__main__":
    main()
