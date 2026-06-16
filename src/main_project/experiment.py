from main_project.evaluate import run_evaluation
from main_project.optimal_transport import save_transformations
from main_project.train import train
from main_project.hyperparameter_optimization import hyperparameter_optimization

if __name__ == "__main__":
    # hyperparameter_optimization()
    save_transformations()
    run_evaluation()