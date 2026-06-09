import equinox as eqx
import jax
from main_project.model import AEv2
from main_project.train import train_model



def objective(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 16, 128, log=True)
    
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Initialize model and train
    seed = 3456
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key, 2)
    model = AEv2(subkey, latent_dim=latent_dim)

    # Train the model and return the validation loss
    val_loss = train_model(model, learning_rate=learning_rate, batch_size=batch_size)
    
    return val_loss
