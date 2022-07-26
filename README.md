# Kuzushiji_MNIST_Fullstack

## Dataset

- Name: Kuzushiji-MNIST
- Task: Classify handwritten characters in ancient Japanese manuscripts
- Kaggle link: [Click here](https://www.kaggle.com/datasets/anokas/kuzushiji)

## How to use

Download the dataset and extract to Data folder.

## Folder structure
  ```
  Kuzushiji_MNIST_Fullstack
  │
  ├── Configs/
  ├── Data/
  ├── EDA/
  ├── Modeling/
  │    ├── DL_pytorch_hydra/
  │    │    │
  │    │    ├── train.py - main script to start training
  │    │    ├── test.py - evaluation of trained model
  │    │    │
  │    │    ├── config.json - holds configuration for training
  │    │    ├── parse_config.py - class to handle config file and cli options
  │    │    │
  │    │    ├── new_project.py - initialize new project with template files
  │    │    │
  │    │    ├── base/ - abstract base classes
  │    │    │   ├── base_data_loader.py
  │    │    │   ├── base_model.py
  │    │    │   └── base_trainer.py
  │    │    │
  │    │    ├── data_loader/ - anything about data loading goes here
  │    │    │   └── data_loaders.py
  │    │    │
  │    │    ├── data/ - default directory for storing input data
  │    │    │
  │    │    ├── model/ - models, losses, and metrics
  │    │    │   ├── model.py
  │    │    │   ├── metric.py
  │    │    │   └── loss.py
  │    │    │
  │    │    ├── saved/
  │    │    │   ├── models/ - trained models are saved here
  │    │    │   └── log/ - default logdir for tensorboard and logging output
  │    │    │
  │    │    ├── trainer/ - trainers
  │    │    │   └── trainer.py
  │    │    │
  │    │    ├── logger/ - module for tensorboard visualization and logging
  │    │    │   ├── visualization.py
  │    │    │   ├── logger.py
  │    │    │   └── logger_config.json
  │    │    │  
  │    │    └── utils/ - small utility functions
  │    │         ├── util.py
  │    │         └── ...
  │    └──DL_pytorch
  │  
  ├── .gitignore
  ├── LICENSE
  └──README.md
 ```

## Results

## License

This project is licensed under the MIT License. See LICENSE for more details

## Acknowledgements

[Pytorch project template](https://github.com/victoresque/pytorch-template) by [victoresque](https://github.com/victoresque)
