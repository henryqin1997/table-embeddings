# table-embeddings
This project is to create embeddings for (web)tables as to classify the label of tables.
## What we did
From http://www.webdatacommons.org/webtables/ we randomly selected 100k webtables. We use Standford-NER as well as hard-coded features (12 binary features for each column) for those table, and use a 2 layer fully-connected neural network to predict their column name. We got 0.82-0.93 accuracy for those tables (accuracy vary with columns).
## Further plan
Webtables are actually sequence(columns) of sequence(rows) of sequence(cell can have multiple word). Using multiple LSTM is a way to create embeddings, while training time could be very long. Together with the explicit features, we will be able to achieve an even better accuracy.

* wordlist_v2.json *25 110576 (input_old)
* wordlist_v3.json *5 1677590 (input_all)
* wordlist_v4.json *35 115859 (input)
* wordlist_v5.json 1083541 (1m_files.json)
* wordlist_v6.json 108236 (100k_files.json)
