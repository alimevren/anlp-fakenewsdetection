## ANLP Fake News Detection Project

In the recent years, if we had to mention a trending term for the field of communication, this wouldbe undoubtedly fake news. This has been provoked and pushed given the worldwide surge of socialmedia. Fake news term refers to news that contain purposely false information, which causes a negative impact on both readers and society as a whole. Considering this, it’s no surprise that different projects and companies such as Newscheck, Newtralor Factmata have aroused to tackle this problem of detecting news with deceptive words which make online users infected and deceived from this false information. 

For Natural Language Processing, this means a extremely promising field of research, as not only the words but also the sentence build-up influence how readers assume the information. Despite the amount of attention this topic has received in the last few years, it wasn’t very accessible to find fake news data sets that ease the task until recently.

In our project, a model will be trained to distinguish real news from fake news. Additionally, datasets with the inclusion of satirical news and headlines from the onion and actual news that sound satirical from r/NotTheOnion will be added to find whether the implemented model is able to differentiate between these two types of news. This last field of work has gotten less attention than simply fake news detection.

### Related Work

### Model

```markdown

    # Setting input data
    titles = d_all_fnn.Title.values
    labels = d_all_fnn.Label.values
    input_ids, attention_masks, labels = tokenize(titles, labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    # Create a 80-10-10 train-validation-test split.

    # Calculate the number of samples to include in each set.
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size= len(dataset) - train_size - val_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    batch_size = 8

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )

    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    
    model = BertForSequenceClassification.from_pretrained(
        bert_model_name,  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=args_nclasses,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    ).to(device)

    model.train()

    # Tell pytorch to run this model on the GPU.
    model.cuda()

    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

    total_steps = len(train_dataloader) * args_epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    random.seed(args_seed)
    np.random.seed(args_seed)
    torch.manual_seed(args_seed)
    torch.cuda.manual_seed_all(args_seed)

    # TEST SET
    prediction_dataloader = DataLoader(
        test_dataset,  # The test samples.
        sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially.
        batch_size=1  # Evaluate with this batch size.
    )

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels, logits_predictions = [], [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs.logits

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        pred_labels_i = np.argmax(logits, axis=1).flatten()
        predictions.extend(pred_labels_i)

        pred_labels_i = logits.squeeze()[1]
        logits_predictions.append(pred_labels_i)

        true_labels.extend(label_ids)


```

### Data

#### FakeNewsNet

#### SatiricLR

#### OnionOrNot

### Experiments

### Discussion

### Conclusion

```markdown
Syntax highlighted code block

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).