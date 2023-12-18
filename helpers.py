from emoji import demojize
from nltk.tokenize import TweetTokenizer
import numpy as np
from datasets import load_dataset,Dataset,DatasetDict#, load_metric
import evaluate


from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput, SequenceClassifierOutput
# from transformers import AdamW

import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from torch.optim import AdamW

from tqdm import tqdm
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import pandas as pd
import time


# def tokenize(batch):
#   return tokenizer(batch["tweets"], truncation=True, max_length=128)

class ClassifierModel(nn.Module):
    def __init__(self, checkpoint, num_labels):
        super(ClassifierModel, self).__init__()
        self.num_labels = num_labels

        config = AutoConfig.from_pretrained(checkpoint, output_hidden_states=True, output_attentions=True)
        self.model = AutoModel.from_pretrained(checkpoint, config=config)

        self.classifier = nn.Linear(768, num_labels)


    def forward(self, input_ids = None, attention_mask=None, labels=None):
        outputs = self.model(input_ids = input_ids, attention_mask =attention_mask)

        last_hidden_state = outputs[0]

        sequence_outputs = last_hidden_state
        
        logits = self.classifier(sequence_outputs[:, 0,  :].view(-1, 768))

        loss = None

        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss = loss, logits = logits, hidden_states = outputs.hidden_states, attentions = outputs.attentions), loss


# Function to load weights
def load_weights(model, weights_path):
    
    map_location = 'cpu' if not torch.cuda.is_available() else None

    model.load_state_dict(torch.load(weights_path, map_location=map_location))
    model.eval()

# def load_weights(model, weights_path):
#     # Load the state dictionary and map it to the CPU
#     model.load_state_dict(torch.load(weights_path, map_location=map_location))

#     # Set the model to evaluation mode
#     model.eval()







def normalizeToken(token):
    tokenizer = TweetTokenizer()

    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tokenizer = TweetTokenizer()

    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())


if __name__ == "__main__":
    print(
        normalizeTweet(
            "SC has first two presumptive cases of coronavirus, DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier"
        )
    )

def loadModel(checkpoint,device,weights_1_file_path,weights_2_file_path,weights_3_file_path):
    # Load the models
    model_classfier_1 = ClassifierModel(checkpoint = checkpoint, num_labels =2).to(device)
    model_classfier_2 = ClassifierModel(checkpoint = checkpoint, num_labels =2).to(device)
    model_classfier_3 = ClassifierModel(checkpoint = checkpoint, num_labels =2).to(device)



    # Load the 3 Best models
    load_weights(model_classfier_1, weights_1_file_path)
    load_weights(model_classfier_2, weights_2_file_path)
    load_weights(model_classfier_3, weights_3_file_path)

    return model_classfier_1, model_classfier_2, model_classfier_3

def loadTrainTweets(positive_file_path, negative_file_path):
    # Read the file and split lines on "\n"
    with open(positive_file_path, 'r', encoding='utf-8') as file:
        lines_pos = file.read().split('\n')

    # Read the file and split lines on "\n"
    with open(negative_file_path, 'r', encoding='utf-8') as file:
        lines_neg = file.read().split('\n')

    # Create a DataFrame with a single column 'tweets'
    df_pos = pd.DataFrame({'tweets': lines_pos})
    df_pos['labels'] = 1

    df_neg = pd.DataFrame({'tweets': lines_neg})
    df_neg['labels'] = 0

    df = pd.concat([df_pos, df_neg], ignore_index=True)

    df = df.sample(frac=1).reset_index(drop=True)

    return df

def loadTestTweets(test_file_path):
    # Read the file and split lines on "\n"
    with open(test_file_path, 'r', encoding='utf-8') as file:
        lines_test = file.read().split('\n')

    # Create a DataFrame with a single column 'tweets'
    df_test= pd.DataFrame({'tweets': lines_test})

    # Split the "index" column into two columns: "number" and "text"
    df_test[['number', 'text']] = df_test['tweets'].str.split(',', n=1, expand=True)

    # Drop the original "index" column if you no longer need it
    df_test = df_test.drop(columns=['tweets'])

    # Convert the "number" column to numeric type
    df_test['number'] = pd.to_numeric(df_test['number'])
    df_test.columns = ['number', 'tweets']
    df_test.dropna(axis=0,inplace=True)
    return df_test


def normalizeTweets(df_test):
    tqdm.pandas()
    df_test['tweets'] = df_test['tweets'].progress_apply(normalizeTweet)
    return df_test


def tokenizeTweets(checkpoint,df,mode):
    # Create Dataset instance
    dataset_ = Dataset.from_pandas(df)

    if mode == 'train':
        # Train Test Valid Split
        train_testvalid = dataset_.train_test_split(test_size=0.2,seed=15)
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5,seed=15)

        dataset_hf = DatasetDict({
            'train': train_testvalid['train'],
            'test': test_valid['test'],
            'valid': test_valid['train']})

        dataset_hf;
    
    elif mode == 'eval':
        dataset_.remove_columns('number')
        dataset_hf = DatasetDict({
            'test': dataset_})
        dataset_hf;


    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.model_max_len=128
    def tokenize(batch):
        return tokenizer(batch["tweets"], truncation=True, max_length=128)
    tokenized_dataset = dataset_hf.map(tokenize, batched=True)

    if mode =='train':
        tokenized_dataset.set_format('torch', columns=["input_ids", "labels"] )
    elif mode == 'eval':
        tokenized_dataset.set_format('torch', columns=["input_ids"] )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    if mode == 'train':
        train_dataloader = DataLoader(tokenized_dataset['train'], shuffle =True, batch_size = 16, collate_fn = data_collator)
        eval_dataloader = DataLoader(tokenized_dataset['valid'], shuffle =False, batch_size = 16, collate_fn = data_collator)
        test_dataloader = DataLoader(tokenized_dataset['test'], shuffle =False, batch_size = 16, collate_fn = data_collator)
        return train_dataloader, eval_dataloader, test_dataloader
    elif mode == 'eval':
        test_dataloader = DataLoader(tokenized_dataset['test'], shuffle =False, batch_size = 32, collate_fn = data_collator)
        return test_dataloader


def infer(test_dataloader,model_classfier_1,model_classfier_2,model_classfier_3,device):
    progress_bar_eval = tqdm(range(len(test_dataloader)))
    predictions_list_1 = []
    predictions_list_2 = []
    predictions_list_3 = []
    for batch in test_dataloader:
        batch = { k: v.to(device) for k, v in batch.items() }
        with torch.no_grad():
            outputs1, pure_loss1 = model_classfier_1(**batch)
            outputs2, pure_loss2 = model_classfier_2(**batch)
            outputs3, pure_loss3 = model_classfier_3(**batch)

        logits1 = outputs1.logits
        logits2 = outputs2.logits
        logits3 = outputs3.logits

        predictions1 = torch.argmax(logits1, dim = -1 )
        predictions2 = torch.argmax(logits2, dim = -1 )
        predictions3 = torch.argmax(logits3, dim = -1 )

        predictions_list_1.extend(predictions1.cpu().numpy().tolist())
        predictions_list_2.extend(predictions2.cpu().numpy().tolist())
        predictions_list_3.extend(predictions3.cpu().numpy().tolist())

        progress_bar_eval.update(1)
    return predictions_list_1, predictions_list_2, predictions_list_3

def ensembleVote(predictions_list_1, predictions_list_2, predictions_list_3):
    np_predictions_1 = np.array(predictions_list_1)
    np_predictions_2 = np.array(predictions_list_2)
    np_predictions_3 = np.array(predictions_list_3)

    np_predictions_1[np_predictions_1==0]=-1
    np_predictions_2[np_predictions_2==0]=-1
    np_predictions_3[np_predictions_3==0]=-1

    predictions = np_predictions_1 + np_predictions_2 + np_predictions_3

    predictions[predictions<0]=-1
    predictions[predictions>0]=1

    return predictions

def saveSubmission(df_test, predictions, verbose=True):
    df_test['pred'] = predictions
    df_test['number'] = df_test['number'].astype(int)
    result_df = df_test[['number','pred']]
    result_df.columns = ['Id','Prediction']

    # Save the submission to a CSV file
    result_df.to_csv('submissions/submission.csv', index=False)

    if verbose:
        # Display the submission DataFrame
        print(result_df.head())

def train(model_classfier,train_dataloader,eval_dataloader,device):
    # optimizer = AdamW(model_classfier.parameters(), lr = 1e-6)
    optimizer = torch.optim.AdamW(model_classfier.parameters(), lr=1e-6)

    num_epochs = 10
    num_training_steps = num_epochs * len(train_dataloader)

    # metric = load_metric("accuracy")
    metric = evaluate.load("accuracy")


    progress_bar_train = tqdm(range(num_training_steps))
    progress_bar_eval = tqdm(range(num_epochs * len(eval_dataloader) ))


    Acc_max = 0

    for epoch in range(num_epochs):
        model_classfier.train()
        for batch in train_dataloader:
            batch = { k: v.to(device) for k, v in batch.items() }
            # outputs, pure_loss = model_classfier(**batch)
            _, pure_loss = model_classfier(**batch)

            pure_loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            progress_bar_train.update(1)

        print(f"Epoch: {epoch}\n")
        model_classfier.eval()
        for batch in eval_dataloader:
            batch = { k: v.to(device) for k, v in batch.items() }
            with torch.no_grad():
                outputs, pure_loss = model_classfier(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim = -1 )
            metric.add_batch(predictions = predictions, references = batch['labels'] )
            progress_bar_eval.update(1)
        Acc = metric.compute()
        print(f"Accuracy{Acc}\n", flush=True)
        if Acc['accuracy']>Acc_max:
            Acc_max=Acc['accuracy']
            best_weights = model_classfier.state_dict()

    return best_weights

def plot_TF_IDF_results (dfResults, val_opt, fixed_param, var_param) :
    """
    Plot TF-IDF hyperparameter search results

    params :
        dfResults : pandas dataframe containing the results of the hyperparameter search
        val_opt : optimal value for either max_df or min_df
        fixed_param : string indicating which of min_df or max_df is the fixed parameter 
                      for which we're doing the plots
        var_param : string indicating which of min_df or max_df is the variable parameter 
                    for which we're doing the plots
    """
    

    # Collect results corresponding to val_opt
    dfResults_trunc = dfResults[dfResults[fixed_param] == val_opt].copy()

    # Extract unique values corresponding to the string fixed param
    vals = dfResults_trunc[var_param].unique()

    # Plotting
    for val in vals:

        # Filter results for the specified parameter
        lambda_values = (dfResults_trunc[dfResults_trunc[var_param] == val])['Lambda'].copy().values
        accuracy_values = (dfResults_trunc[dfResults_trunc[var_param] == val])['Accuracy'].copy().values

        indices = np.argsort(lambda_values)
        lambda_values = lambda_values[indices]
        accuracy_values = accuracy_values[indices]

        # Plotting for each variable parameter value
        plt.plot(lambda_values, accuracy_values, linewidth = 1 , marker='o', linestyle='-', label=f"{var_param} = {val}")

    # Plot details
    plt.title(f'Accuracy vs Lambda for {fixed_param} = {val_opt}')
    plt.xlabel('Lambda')
    plt.xscale('log')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()