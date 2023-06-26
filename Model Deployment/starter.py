import pickle
import pandas as pd
import sys

year = sys.argv[1]
month = sys.argv[2]


with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)


categorical = ["PULocationID", "DOLocationID"]


def read_data(filename):
    print("---Dowloading")
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


df = read_data(
    f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet"
)


def main(df):
    print("--Modelling")
    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f"The mean predicted duration is {y_pred.mean()}")


if __name__ == "__main__":
    main(df)
