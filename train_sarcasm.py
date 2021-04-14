import pandas as pd
from simpletransformers.classification import ClassificationModel
import time

# Collecting balanced training data from an unbalanced set
def prepare_data():
    df = pd.read_csv("data/sarcasm/train.csv", header=None)
    df.columns = ['label', 'comment', 'author', 'subreddit', 'score', 'ups', 'downs', 'date', 'created_utc', 'parent_comment']
    df = df.drop(['author', 'subreddit', 'score', 'ups', 'downs', 'date', 'created_utc', 'parent_comment'], axis=1)

    sarcasm = 0
    real = 0

    ds1000 = pd.DataFrame()
    ds5000 = pd.DataFrame()
    ds10000 = pd.DataFrame()
    ds100000 = pd.DataFrame()
    test1000 = pd.DataFrame()

    for i in range(df.shape[0]):
        if df.iloc[i]['label'] == 1 and sarcasm < 500:
            ds1000 = ds1000.append(df.iloc[[i]])
            sarcasm = sarcasm + 1

        if df.iloc[i]['label'] == 0 and real < 500:
            ds1000 = ds1000.append(df.iloc[[i]])
            real = real + 1

        if sarcasm == 500 and real == 500:
            break

    sarcasm = 0
    real = 0
    ds1000 = ds1000.sample(frac=1).reset_index(drop=True)
    pd.DataFrame(ds1000).to_csv("data/sarcasm/train_1000.csv", index=None, header=None)

    for i in range(df.shape[0]):
        if df.iloc[i]['label'] == 1 and sarcasm < 2500:
            ds5000 = ds5000.append(df.iloc[[i]])
            sarcasm = sarcasm + 1

        if df.iloc[i]['label'] == 0 and real < 2500:
            ds5000 = ds5000.append(df.iloc[[i]])
            real = real + 1

        if sarcasm == 2500 and real == 2500:
            break

    sarcasm = 0
    real = 0
    ds5000 = ds5000.sample(frac=1).reset_index(drop=True)
    pd.DataFrame(ds5000).to_csv("data/sarcasm/train_5000.csv", index=None, header=None)

    for i in range(df.shape[0]):
        if df.iloc[i]['label'] == 1 and sarcasm < 5000:
            ds10000 = ds10000.append(df.iloc[[i]])
            sarcasm = sarcasm + 1

        if df.iloc[i]['label'] == 0 and real < 5000:
            ds10000 = ds10000.append(df.iloc[[i]])
            real = real + 1

        if sarcasm == 5000 and real == 5000:
            break

    sarcasm = 0
    real = 0
    ds10000 = ds10000.sample(frac=1).reset_index(drop=True)
    pd.DataFrame(ds10000).to_csv("data/sarcasm/train_10000.csv", index=None, header=None)

    for i in range(df.shape[0]):
        if df.iloc[i]['label'] == 1 and sarcasm < 50000:
            ds100000 = ds100000.append(df.iloc[[i]])
            sarcasm = sarcasm + 1

        if df.iloc[i]['label'] == 0 and real < 50000:
            ds100000 = ds100000.append(df.iloc[[i]])
            real = real + 1

        if sarcasm == 50000 and real == 50000:
            break

    sarcasm = 0
    real = 0
    ds100000 = ds100000.sample(frac=1).reset_index(drop=True)
    pd.DataFrame(ds100000).to_csv("data/sarcasm/train_100000.csv", index=None, header=None)

    for i in range(300000, df.shape[0]):
        if df.iloc[i]['label'] == 1 and sarcasm < 500:
            test1000 = test1000.append(df.iloc[[i]])
            sarcasm = sarcasm + 1

        if df.iloc[i]['label'] == 0 and real < 500:
            test1000 = test1000.append(df.iloc[[i]])
            real = real + 1

        if sarcasm == 500 and real == 500:
            break

    test1000 = test1000.sample(frac=1).reset_index(drop=True)
    pd.DataFrame(test1000).to_csv("data/sarcasm/test.csv", index=None, header=None)

    print(ds1000.shape[0])
    print(ds5000.shape[0])
    print(ds10000.shape[0])
    print(ds100000.shape[0])
    print(test1000.shape[0])


def train_model(model_type, model_name, amount):
    print('Starting run:', model_type, model_name, amount)

    train_df = pd.read_csv(f"data/sarcasm/train_{amount}.csv", header=None)
    train_df.columns = ["labels", "text"]

    eval_df = pd.read_csv("data/sarcasm/test.csv", header=None)
    eval_df.columns = ["labels", "text"]

    t0 = time.time()
    train_args = {
        'output_dir': f'model-outputs/sarcasm/{model_type}-{model_name}-{amount}-outputs',
        'max_seq_length': 256,
        'num_train_epochs': 5,
        'train_batch_size': 16,
        'eval_batch_size': 32,
        'gradient_accumulation_steps': 1,
        'learning_rate': 5e-5,
        'save_steps': 50000,
        'evaluate_during_training': True,
        'evaluate_during_training_steps': 1000,
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
        for j in ["1000", "5000", "10000", "100000"]:
            train_model(model_types[i], model_names[i], j)