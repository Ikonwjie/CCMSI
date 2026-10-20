# Data Preparation

## Downloads and Directory Structure

- ETT: [ETDataset](https://github.com/zhouhaoyi/ETDataset)
- Weather, Electricity, and Traffic: [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- M4: [M4-methods](https://github.com/Mcompetitions/M4-methods)

Run the following commands from the project root:

```bash
python scripts/prepare_data.py --dataset ETT
python scripts/prepare_data.py --dataset Weather
python scripts/prepare_data.py --dataset Electricity
python scripts/prepare_data.py --dataset Traffic
python scripts/prepare_data.py --dataset M4
```

The ETT command downloads all four subsets together. The CSV files for Weather, Electricity, and Traffic are downloaded from the Time-Series-Library [dataset repository on Hugging Face](https://huggingface.co/datasets/thuml/Time-Series-Library). Each download resolves the source repository to a specific commit and records that revision, the download URL, and the file's SHA-256 hash in `download_manifest.json` in the destination directory. Use `--revision` to specify a source repository revision and `--output` to specify the output directory. Existing files are not overwritten.

Arrange the data as shown below. Extract the M4 Naive2 archive to obtain the CSV file:

```text
dataset/
  ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv
  weather/weather.csv
  electricity/electricity.csv
  traffic/traffic.csv
  m4/
    Train/{Yearly,Quarterly,Monthly,Weekly,Daily,Hourly}-train.csv
    Test/{Yearly,Quarterly,Monthly,Weekly,Daily,Hourly}-test.csv
    submission-Naive2.csv
```

Standard CSV files must retain the `date` column and the numeric columns. The first column of each M4 file contains the series ID.

## Data Splits and Preprocessing

- ETT is split chronologically into 12 months for training, 4 months for validation, and 4 months for testing. Weather, Electricity, and Traffic use a 7:1:2 split.
- Standardization parameters are computed from the training data only and reused for validation and testing. Low-data tasks use the specified percentage of the earliest training windows.
- M4 uses the official training and test sets. The last H observations of each training series are held out for validation. By default, the model is then refitted on the full training set for the selected number of epochs.
- The forecasting horizons H for the six M4 frequencies are 6/8/18/13/14/48, respectively, with an input length of 2H. Predictions are restored to their original units before computing sMAPE, MASE, and OWA relative to Naive2.
