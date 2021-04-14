import pandas as pd
from simpletransformers.classification import ClassificationModel
import time

def train_model(model_type, model_name, amount):
    print('Starting run:', model_type, model_name, amount)

    train_df = pd.read_csv(f'data/imdb/train_{amount}.csv', header=None)
    train_df["text"] = train_df.iloc[:, 0]
    train_df = train_df.drop(train_df.columns[[0]], axis=1)
    train_df.columns = ["labels", "text"]
    train_df["labels"] = train_df["labels"].apply(
        lambda x: 1 if x == "positive" else 0
    )

    eval_df = pd.read_csv("data/imdb/test.csv", header=None)
    eval_df["text"] = eval_df.iloc[:, 0]
    eval_df = eval_df.drop(eval_df.columns[[0]], axis=1)
    eval_df.columns = ["labels", "text"]
    eval_df["labels"] = eval_df["labels"].apply(
        lambda x: 1 if x == "positive" else 0
    )

    t0 = time.time()
    train_args = {
        'output_dir': f'model-outputs/imdb/new/{model_type}-{model_name}-outputs',
        'max_seq_length': 256,
        'num_train_epochs': 5,
        'train_batch_size': 16,
        'eval_batch_size': 32,
        'gradient_accumulation_steps': 1,
        'learning_rate': 5e-5,
        'save_steps': 50000,
        'evaluate_during_training': True,
        'evaluate_during_training_steps': 40000,
        'reprocess_input_data': True,
        'save_model_every_epoch': False,
        'overwrite_output_dir': True,
        'no_cache': True,
        'use_early_stopping': True,
        'early_stopping_patience': 3,
        'manual_seed': 4,
    }

    model = ClassificationModel(model_type, model_name, num_labels=2, args=train_args)
    model.train_model(train_df, eval_df=eval_df)
    print('Run finished')
    t1 = time.time()
    total = t1 - t0

    print('Time:', total)
    print('--------------------')

if __name__ == '__main__':
    model_types = ["bert", "distilbert", "roberta", "albert", "xlnet"]
    model_names = ["bert-base-cased", "distilbert-base-cased", "roberta-base", "albert-base-v2", "xlnet-base-cased"]

    for i in range(len(model_types)):
        for j in ["5000", "10000"]:
            train_model(model_types[i], model_names[i], j)
