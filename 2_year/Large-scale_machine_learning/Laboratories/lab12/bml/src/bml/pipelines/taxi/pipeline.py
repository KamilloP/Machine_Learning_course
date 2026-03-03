from kedro.pipeline import Node, Pipeline
from .nodes import prepare_data, train_model, log_accuracy


def create_pipeline(**kwargs):
    return Pipeline(
        [
            Node(
                prepare_data,
                ["taxi_data", "params:taxi_options"],
                ["X_train", "X_test", "y_train", "y_test"],
                name="prepare_taxi_data",
            ),
            Node(
                train_model,
                ["X_train", "X_test", "y_train", "y_test", "params:taxi_options"],
                ["taxi_model", "taxi_accuracy_report"],
                name="train_taxi_model",
            ),
            Node(log_accuracy,
                 ["taxi_accuracy_report"],
                 None,
                 name="accuracy_results"
            ),
        ]
    )
    
# from kedro.pipeline import Node, Pipeline

# from .nodes import predict, report_accuracy, train_model


# def create_pipeline(**kwargs):
#     return Pipeline(
#         [
#             Node(
#                 train_model,
#                 ["example_train_x", "example_train_y", "params:model_options"],
#                 "example_model",
#                 name="train",
#             ),
#             Node(
#                 predict,
#                 dict(model="example_model", test_x="example_test_x"),
#                 "example_predictions",
#                 name="predict",
#             ),
#             Node(
#                 report_accuracy,
#                 ["example_predictions", "example_test_y"],
#                 None,
#                 name="report",
#             ),
#         ]
#     )
