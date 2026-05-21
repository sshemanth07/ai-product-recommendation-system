from clearml import Task
from clearml.automation import UniformParameterRange, DiscreteParameterRange
from clearml.automation import RandomSearch, HyperParameterOptimizer


# ============================================================
# ClearML HPO Controller Task
# ============================================================

task = Task.init(
    project_name="AI-Based Product Recommendation System",
    task_name="XGBoost HyperParameter Optimizer",
    tags=["HPO", "XGBoost", "RandomSearch", "Sprint 3"]
)


# ============================================================
# HyperParameter Optimizer
# ============================================================

optimizer = HyperParameterOptimizer(
    base_task_id="3cb1c6dd6cb3484c971a85eaed23d520",

    hyper_parameters=[
        DiscreteParameterRange(
            "General/max_depth",
            values=[3, 5, 7, 9]
        ),
        UniformParameterRange(
            "General/learning_rate",
            min_value=0.01,
            max_value=0.30,
            step_size=0.05
        ),
        DiscreteParameterRange(
            "General/n_estimators",
            values=[50, 100, 150, 200]
        ),
        UniformParameterRange(
            "General/subsample",
            min_value=0.60,
            max_value=1.00,
            step_size=0.10
        ),
        UniformParameterRange(
            "General/colsample_bytree",
            min_value=0.60,
            max_value=1.00,
            step_size=0.10
        )
    ],

    objective_metric_title="Validation",
    objective_metric_series="F1-Score",
    objective_metric_sign="max",

    max_number_of_concurrent_tasks=2,
    optimizer_class=RandomSearch,

    execution_queue="default",
    time_limit_per_job=300,
    pool_period_min=0.5
)

optimizer.set_time_limit(in_minutes=15)

print("Starting XGBoost HyperParameter Optimizer...")
optimizer.start()

while not optimizer.reached_time_limit():
    top_experiments = optimizer.get_top_experiments(top_k=3)
    print("Current top experiments:")
    print(top_experiments)

optimizer.wait()
optimizer.stop()

best_experiment = optimizer.get_top_experiments(top_k=1)

print("HPO completed.")
print("Best experiment:")
print(best_experiment)

print(task.get_output_log_web_page())

task.close()