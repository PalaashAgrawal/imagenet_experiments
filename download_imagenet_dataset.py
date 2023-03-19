from datasets import load_dataset
dataset = load_dataset("imagenet-1k", split="train",  use_auth_token=True)