### Q1 - CRF

- We have extracted tags from the FIGER dataset and loaded them separately for use in `tags.pickle` file. Make sure it is in the code directory while running the codes.
- Make Sure to use GPU for training the model for faster results.
- We have used the `CRF` model using `tensorflow` library. Make sure you install all the python modules before running.
- Change the path to training file `drive/MyDrive/train.json` and testing file `drive/MyDrive/test.json` in `CRF.ipynb` to the path of the `data` directory.
- Run all the cell sequentially to first train the data and then test the data and then calculate the metrics for analysis.

### Q2- LSTM

- We have used the `LSTM` model using `tensorflow` library. Make sure you install all the python modules before running.
- Change the path to training file `drive/MyDrive/train.json` and testing file `drive/MyDrive/test.json` in `LSTM.ipynb` to the path of the `data` directory.
- Run all the cell sequentially to first train the data and then test the data and then calculate the metrics for analysis.

### Q3- BERT
- We have used the `BertForTokenClassification` model. Make sure you install all the python modules before running.
- Copy the data folder containing train.json, dev.json and test.json to the current folder.
- Run all the cells in train_BERT.ipynb sequentially to train the model on train.json. The code will save the trained models as model_0, model_1 and so on for each training epoch
- To evaluate the trained models on test.json, run eval_BERT.ipynb sequentially. You may need to tweak the model_name that should be loaded and used for evaluation.


### Outputs

- Metrics Outputs are stored in `outputs` directory.

### Report

- For detailed report, please refer to `report.pdf` file.
