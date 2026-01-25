import matplotlib.pyplot as plt
import wandb
import os

def plot_and_log_metrics(history, project_name="mvrce-chess", run_name="overfit_test"):
    """
    Plots training metrics and logs them to Weights & Biases.
    
    Args:
        history (dict): Dictionary containing lists of metrics.
                        Expected keys: 'loss', 'policy_loss', 'value_loss', 'mate_loss', 'accuracy'.
        project_name (str): W&B project name.
        run_name (str): W&B run name.
    """
    
    # 1. Plotting
    epochs = range(1, len(history['loss']) + 1)
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Losses
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history['loss'], label='Total Loss', linewidth=2)
    plt.plot(epochs, history['policy_loss'], label='Policy Loss', linestyle='--')
    plt.plot(epochs, history['value_loss'], label='Value Loss', linestyle='--')
    plt.plot(epochs, history['mate_loss'], label='Mate Loss', linestyle='--')
    plt.title('Training Losses')
    plt.xlabel('Step (x10)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history['accuracy'], label='Top-1 Accuracy', color='green')
    plt.title('Policy Accuracy')
    plt.xlabel('Step (x10)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = "training_metrics.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()

    # 2. W&B Logging
    # Check if W&B is available and logged in, or just init (will ask for key if not)
    # For automated environments, ensure WANDB_API_KEY is set.
    try:
        wandb.init(project=project_name, name=run_name, config={"batch_size": "single_batch"})
        
        # Log all metrics at once (as a summary or history?)
        # Since we have the full history, we can log step by step or just the final plot.
        # Let's log step by step for interactive charts.
        
        for i in range(len(history['loss'])):
            wandb.log({
                "train/loss": history['loss'][i],
                "train/policy_loss": history['policy_loss'][i],
                "train/value_loss": history['value_loss'][i],
                "train/mate_loss": history['mate_loss'][i],
                "train/accuracy": history['accuracy'][i],
                "step": (i + 1) * 10 # Assuming logging interval is 10
            })
            
        # Log the plot image
        wandb.log({"training_plot": wandb.Image(plot_path)})
        
        print("Logged metrics to Weights & Biases.")
        wandb.finish()
        
    except Exception as e:
        print(f"Failed to log to W&B: {e}")
        print("Ensure 'wandb' is installed and configured.")
