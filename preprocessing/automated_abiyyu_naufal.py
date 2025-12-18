import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

      #menghapus missing value
    df = df.dropna()

    #menghapus duplikat
    df = df.drop_duplicates()

    #seleksi kolom
    cols_to_drop = ["TransactionID","AccountID","DeviceID","IP Address","MerchantID"]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    #memilah kolom
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    #scaling kolom numerik
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    #encoding kolom kategorikal
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    #menghapus outlier
    def remove_outliers_iqr(data, columns):
        for col in columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            data = data[
                (data[col] >= lower) &
                (data[col] <= upper)
            ]
        return data

    df = remove_outliers_iqr(df, num_cols)

    #melakukan bining
    if "TransactionAmount" in df.columns:
        df["TransactionAmount_bin"] = pd.cut(
            df["TransactionAmount"],
            bins=3,
            labels=[0, 1, 2]
        )

    df.to_csv(output_path, index=False)
    print("Preprocessing selesai.")

if __name__ == "__main__":
    preprocess_data(
        "bank_transactions_data_preprocessing/bank_transactions_data_edited.csv",
        "bank_transactions_data_preprocessing/bank_transactions_preprocessed.csv"
    )
