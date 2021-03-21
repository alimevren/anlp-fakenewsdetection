## ANLP Fake News Detection Project

In the recent years, if we had to mention a trending term for the field of communication, this would be undoubtedly fake news. This has been provoked and pushed given the worldwide surge of socialmedia. Fake news term refers to news that contain purposely false information, which causes a negative impact on both readers and society as a whole. Considering this, it’s no surprise that different projects and companies such as Newscheck, Newtralor Factmata have aroused to tackle this problem of detecting news with deceptive words which make online users infected and deceived from this false information. 

For Natural Language Processing, this means a extremely promising field of research, as not only the words but also the sentence build-up influence how readers assume the information. Despite the amount of attention this topic has received in the last few years, it wasn’t very accessible to find fake news data sets that ease the task until recently.

In our project, a model will be trained to distinguish real news from fake news. Additionally, datasets with the inclusion of satirical news and headlines from the onion and actual news that sound satirical from r/NotTheOnion will be added to find whether the implemented model is able to differentiate between these two types of news. Finally, project aims if it is possible to detect AI-generated headlines of realnews and how do these results compare in relation to human-generated fake news. The effectiveness of our proposed model will be concluded, leading to the analysis of what is to come in the fight against fake news.

### Related Work
Previous studies on fake news detection contain different deep learning approaches. LSTMs and BERT are techniques largely used in those studies. Some of those models utilize multi-modal features instead of only textual context to improve classification results. Alternatively, some other studies only use text corpus and metadata such as source of information, date and author for fake news detection with different word embeddings and classification models. Our approach currently focusing detection using title/headline of the article only and as a future research could be extended with more input data for better detection.

### Approach
In our project, we formulated the problem as a classification problem for fake news. Either satirical or not, we would like to differentiate fake news from the real ones using a BERT [Link](https://github.com/google-research/bert) pre-trained language model for English. BERT is a method of pre-training language representations, pre-trained general-purpose "language understanding" model on a large text corpus (like Wikipedia). BERT outperforms previous methods mainly due it being the first unsupervised, deeply bidirectional system for pre-training NLP.

BERT is currently one of the state of the art methods in natural language processing with its parameter size and pre-training approach. Even though there are bigger models with higher parameter sizes such as GPT-2 and GPT-3, we had to select a computationally feasible model for the project. Since we have limited computing resources in terms of GPU and disk size, it was preferred to try BERT and experiment its performance for fake news detection classification problem. Google BERT team released different versions of the pre-trained BERT model which BERT-base model stands out as the most commonly used. Instead of using BERT-base-uncased model which still takes a long time to train, we preferred to use BERT-medium one released from the same team mentioned in the paper Well-Read students learn better [Link](https://github.com/google-research/bert). The BERT-medium model used comes from the HuggingFace transformers library [Link](https://huggingface.co/prajjwal1/bert-medium). This allowed us to be able run multiple training and evaluation cycles, as well as more iterations during project experiments.

In our implementation part we used Chris McCormick's BERT Fine-Tuning Tutorial with PyTorch [Link](http://mccormickml.com/2019/07/22/BERT-fine-tuning/) as a basis, and adding necessary functions whenever needed. Firstly, we created pandas DataFrame objects with title and label values (0, 1 - being 0 for fake news, and 1 for real news) for the model training and testing. After loading our pre-trained BERT model and tokenizer using BertForSequenceClassification for two classes, the data set is split into a train, validation and test set (using a 80-10-10 ratio).

```markdown

```

### Data
Datasets used during the project listed as follows.
1. FakeNewsNet
2. ISOT Fake News Dataset
3. SatiricLR
4. OnionOrNot
5. RealNews

### Conclusion
Extensive experiments have been performed during the project, obtaining different results. We can positively highlight that the detection of fake news with our trained models on the ISOT data set achieved confidently positive accuracy values, being thus our biggest success for our model. Unfortunately, our satirical news experiments achieved mixed results. Although retraining our model to include the SatiricLR data set gave us an even higher accuracy than our previous trained model, we did not notice any significant difference between our two differently trained model tested on the OnionOrNot data set. As we already highlighted this lack of expected improvement could be the size difference between the ISOT and SatiricLR data sets. Although, another big factor could be the missing news article content itself, as we only trained and tested on headlines. It seems that the context of the news articles is more important when working with satirical news than with regular news, but this is a topic for future research. Another future research possibility in the topic of actual satirical news and satirical seeming news, could be to compare the average accuracy of a model and the average accuracy of real people on differentiating between the two.
For the topic of machine generated headlines we have observed that these generated headlines seem to be a much bigger struggle for our implementation than we originally thought. The improvement could lie in considering the generated news in the training phase.
The topic of fake news detection has many more problems and research proposals that are still arising, making this a very interesting field of work for the upcoming year. However, the detection of satirical news has not grabbed as much attention in comparison, and considering our results in this project we could conclude that this field potentially needs more factors to be taking into account.


**Bold** and _Italic_ and `Code` text

 and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
