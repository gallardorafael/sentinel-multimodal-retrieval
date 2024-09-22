import argparse

from transformers import AutoModel


def main(model_name):
    print(f"Downloading model: {model_name}")
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Could not load model {model_name}. Error: {e}")
        return
    print(f"Model {model_name} downloaded successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a pretrained model.")
    parser.add_argument("model_name", type=str, help="The name of the model to download.")
    args = parser.parse_args()
    main(args.model_name)
