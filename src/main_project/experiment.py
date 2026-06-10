from src.main_project.evaluate import run_evaluation
from src.main_project.optimal_transport import save_transformations
from src.main_project.train import train

if __name__ == "__main__":
    train(epochs=1000, val_split=0.2)
    save_transformations()
    run_evaluation()