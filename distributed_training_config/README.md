config files for distributed training using huggingface accelerate (https://huggingface.co/docs/accelerate)

If you're using a single node, multiGPU - use `singlenode_multigpu_training_config/config.yaml`

If you're using GPUs accross multiple nodes: use `multinode_training_config/accelerate_config_host.yaml` in the main node and `multinode_training_config/accelerate_config_client0.yaml` in the client node. 

Make copies of ..client0.yaml for more nodes and make necessary changes. See here for configuration documentation: https://huggingface.co/docs/accelerate/usage_guides/explore