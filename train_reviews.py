import pandas as pd
import numpy as np
from simpletransformers.classification import ClassificationModel
import time


def train_model(model_type, model_name):
    print('Starting run:', model_type, model_name)

    train_df = pd.read_csv("data/reviews/new_training.csv", header=None)
    train_df.columns = ["text", "labels"]

    eval_df = pd.read_csv("data/reviews/new_test.csv", header=None)
    eval_df.columns = ["text", "labels"]

    t0 = time.time()
    train_args = {
        'output_dir': f'model-outputs/reviews/{model_name}-outputs',
        'max_seq_length': 256,
        'num_train_epochs': 5,
        'train_batch_size': 16,
        'eval_batch_size': 32,
        'learning_rate': 5e-5,
        'evaluate_during_training': True,
        'evaluate_during_training_steps': 5000,
        'save_model_every_epoch': False,
        'overwrite_output_dir': True,
        'no_cache': True,
        'use_early_stopping': True,
        'early_stopping_patience': 3,
        'manual_seed': 4,
        'regression': True,
        'best_model_dir': f'outputs/{model_type}/best_model'
    }

    # 'wandb_project': 'hotel-reviews',

    model = ClassificationModel(model_type, model_name, num_labels=1, args=train_args)
    model.train_model(train_df, eval_df=eval_df)
    print('Run finished')
    t1 = time.time()
    total = t1 - t0

    print('Time:', total)
    print('--------------------')

    predictions, raw_outputs = model.predict(["Good hotel, nothing great but is enough to have a decent experience and fun with the family"])
    print('Prediction:', predictions)
    print('Raw Outputs:', raw_outputs)


def get_evaluation_parameter(model):
    eval_df = pd.read_csv("data/reviews/new_test.csv", header=None)
    eval_df.columns = ["text", "labels"]

    model_type = f'outputs/{model}/best_model'
    model = ClassificationModel(model, model_type)
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    print('Results:', result)
    print('Outputs:', model_outputs)

    plots = []
    differences = []
    max_difference = 0
    min_difference = 5
    for i in range(len(model_outputs)):
        value = round(abs(model_outputs[i] - eval_df['labels'][i]), 2)
        actual = round(eval_df['labels'][i], 2)
        plots.append([actual, model_outputs[i], value])

        if value > max_difference:
            max_difference = value
        if value < min_difference:
            min_difference = value

        differences.append(value)

    print('Max Difference:', max_difference)  # 3.8447265625
    print('Min Difference:', min_difference)  # 0.0

    parameter = sum(differences) / len(differences)
    print('Parameter:', parameter)  # 0.40202807008058644

    pd.DataFrame(differences).to_csv("test.csv", index=None)
    pd.DataFrame(plots).to_csv("plots.csv", index=None)


def evaluate_rating(comment, rating, model_type, model_path):
    max_diff = 1.5
    model = ClassificationModel(model_type, model_path)
    predictions, raw_outputs = model.predict([comment])

    if abs(predictions - rating) > max_diff:
        print('Prediction:', predictions)
        print('Rating:', rating)
        print(comment)


def use_model(model_type, model_path):
    model = ClassificationModel(model_type, model_path)

    predictions, raw_outputs = model.predict(
        ["Good hotel, nothing great but is enough to have a decent "
         "experience and fun with the family"])
    print('Prediction:', predictions)
    print('Raw Outputs:', raw_outputs)

    predictions, raw_outputs = model.predict(["Horrible experience, the staff was rude and the food terrible"])
    print('Prediction:', predictions)
    print('Raw Outputs:', raw_outputs)

    predictions, raw_outputs = model.predict(["I truly loved it! Great staff and delicious food. 5/5"])
    print('Prediction:', predictions)
    print('Raw Outputs:', raw_outputs)

def collectTrainingData():
    df = pd.read_csv("data/reviews/test.csv", header=None)
    df.columns = ["text", "labels"]

    new_training_data = []
    visualisation = []
    one_values = 0
    two_values = 0
    three_values = 0
    four_values = 0
    five_values = 0
    MAX_COUNT = 500
    for i in range(len(df)):
        if df['labels'][i] == 1 and one_values < MAX_COUNT:
            one_values += 1
            new_training_data.append([df['text'][i], df['labels'][i]])
            visualisation.append(df['labels'][i])
        elif df['labels'][i] == 2 and two_values < MAX_COUNT:
            two_values += 1
            new_training_data.append([df['text'][i], df['labels'][i]])
            visualisation.append(df['labels'][i])
        elif df['labels'][i] == 3 and three_values < MAX_COUNT:
            three_values += 1
            new_training_data.append([df['text'][i], df['labels'][i]])
            visualisation.append(df['labels'][i])
        elif df['labels'][i] == 4 and four_values < MAX_COUNT:
            four_values += 1
            new_training_data.append([df['text'][i], df['labels'][i]])
            visualisation.append(df['labels'][i])
        elif df['labels'][i] == 5 and five_values < MAX_COUNT:
            five_values += 1
            new_training_data.append([df['text'][i], df['labels'][i]])
            visualisation.append(df['labels'][i])

    print('One:', one_values)
    print('Two:', two_values)
    print('Three:', three_values)
    print('Four:', four_values)
    print('Five:', five_values)

    pd.DataFrame(new_training_data).to_csv("data/reviews/new_test.csv", index=None)
    pd.DataFrame(visualisation).to_csv("data/reviews/vis_test.csv", index=None)

if __name__ == '__main__':
    # collectTrainingData()
    # train_model("albert", "albert-base-v2")
    get_evaluation_parameter('albert')
    # use_model("albert", "outputs/albert/best_model")

    # eval_df = pd.read_csv("data/reviews/test.csv", header=None)
    # eval_df.columns = ["text", "labels"]
    #
    # for i in range(len(eval_df['text'])):
    #     evaluate_rating(eval_df['text'][i], eval_df['labels'][i], 'albert', 'outputs/best_model')
