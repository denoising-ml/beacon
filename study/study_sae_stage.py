import study.module.module_workflow as workflow

if __name__ == "__main__":

    config = {
        'sae_layer': {
            'hidden_dim': [10, 10],
            'epochs': 1000,
        },
    }

    file_names = workflow.StudyFilenames(0, '20190623_183627')

    workflow.run_sae(config, file_names)