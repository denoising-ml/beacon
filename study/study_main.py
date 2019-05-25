from datetime import datetime
import study.module.module_workflow as workflow


if __name__ == "__main__":

    study_number = datetime.now().strftime('%Y%m%d_%H%M%S')

    run_number = 0
    epochs = 500

    hidden_layer_combo = [
        [16, 8, 8]
    ]

    for hidden_layer in hidden_layer_combo:
        config = workflow.generate_config(
            epochs=epochs,
            sae_hidden_dim=hidden_layer,
            lstm_cell_neurons=8,
            lstm_time_step=4,
            lstm_batch_size=50
        )
        run_number += 1
        workflow.study_hsi(config=config, run_number=run_number, study_number=study_number)
