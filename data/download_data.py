"""
Download and prepare the CLINC150 dataset for training.
CLINC150: 150 intent classes, 23,700 utterances, includes out-of-scope class.
Source: https://huggingface.co/datasets/clinc_oos
"""

from datasets import load_dataset
import pandas as pd
import os

def download_and_prepare():
    print("Downloading CLINC150 dataset...")
    dataset = load_dataset("clinc_oos", "plus")

    for split in ["train", "validation", "test"]:
        df = pd.DataFrame(dataset[split])
        # Map integer labels to intent names
        label_names = dataset[split].features["intent"].names
        df["intent_name"] = df["intent"].apply(lambda x: label_names[x])
        out_path = os.path.join(os.path.dirname(__file__), f"{split}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved {split}: {len(df)} rows -> {out_path}")

    # Save label map
    label_map = {i: name for i, name in enumerate(label_names)}
    pd.DataFrame(list(label_map.items()), columns=["id", "intent"]).to_csv(
        os.path.join(os.path.dirname(__file__), "label_map.csv"), index=False
    )
    print(f"Saved label map: {len(label_map)} intents")
    print("Done.")

if __name__ == "__main__":
    download_and_prepare()
