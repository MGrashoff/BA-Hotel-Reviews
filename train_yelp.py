import pandas as pd
from simpletransformers.classification import ClassificationModel
import time

def train_model(model_type, model_name, amount):
    print('Starting run:', model_type, model_name)

    train_df = pd.read_csv(f'data/yelp/train_{amount}.csv', header=None)
    train_df.head()

    eval_df = pd.read_csv('data/yelp/test.csv', header=None)
    eval_df.head()

    train_df[0] = (train_df[0] == 2).astype(int)
    eval_df[0] = (eval_df[0] == 2).astype(int)

    train_df = pd.DataFrame({
        'text': train_df[1].replace(r'\n', ' ', regex=True),
        'label': train_df[0]
    })

    eval_df = pd.DataFrame({
        'text': eval_df[1].replace(r'\n', ' ', regex=True),
        'label': eval_df[0]
    })

    t0 = time.time()
    train_args = {
        'output_dir': f'model-outputs/yelp/{model_type}-{model_name}-{amount}-outputs',
        'max_seq_length': 256,
        'num_train_epochs': 3,
        'train_batch_size': 16,
        'eval_batch_size': 32,
        'gradient_accumulation_steps': 1,
        'learning_rate': 5e-5,
        'save_steps': 50000,
        'evaluate_during_training': True,
        'evaluate_during_training_steps': 100000,
        'save_model_every_epoch': False,
        'overwrite_output_dir': True,
        'reprocess_input_data': False,
        'use_early_stopping': True,
        'early_stopping_patience': 3,
        'manual_seed': 4,
    }

    model = ClassificationModel(model_type, model_name, args=train_args)
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
        for j in ["1000", "5000", "10000", "100000"]:
            train_model(model_types[i], model_names[i], j)
