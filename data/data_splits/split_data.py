import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_clean_csv(filepath):
    df = pd.read_csv(filepath)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


def aggregate_sentences(dataset):
    # Get sents
    aggregated = dataset.groupby("sentence_num")["word"].apply(" ".join).reset_index()
    return aggregated


def split_sentences(dataset, test_size=0.4, random_state=12321):
    unique_sentence_ids = dataset["sentence_num"].unique()
    train_ids, test_ids = train_test_split(
        unique_sentence_ids, test_size=test_size, random_state=random_state
    )
    train = dataset[dataset["sentence_num"].isin(train_ids)].reset_index(drop=True)
    test = dataset[dataset["sentence_num"].isin(test_ids)].reset_index(drop=True)
    return train, test


def remove_duplicates_within_dataset(dataset):
    # Remove within duplicates
    unique_sentences = dataset.drop_duplicates(subset="word").reset_index(drop=True)
    return unique_sentences


def remove_duplicates_across_datasets(train, test):
    # Remove cross diplicates
    test_cleaned = test[~test["word"].isin(train["word"])].reset_index(drop=True)
    return train, test_cleaned


def deaggregate_sentences(aggregated, original_dataset):
    # Back to the original format
    return original_dataset[
        original_dataset["sentence_num"].isin(aggregated["sentence_num"])
    ].reset_index(drop=True)


def check_overlap(train, test):
    overlap = pd.merge(train, test, on=["sentence_num", "word"])
    return len(overlap) == 0


def main():
    provo_bos = load_and_clean_csv("original_data/provo_bos.csv")
    zuco = load_and_clean_csv("original_data/zuco.csv")
    dundee = load_and_clean_csv("original_data/dundee.csv")

    provo_bos_aggregated = aggregate_sentences(provo_bos)
    zuco_aggregated = aggregate_sentences(zuco)
    dundee_aggregated = aggregate_sentences(dundee)

    provo_bos_aggregated = remove_duplicates_within_dataset(provo_bos_aggregated)
    zuco_aggregated = remove_duplicates_within_dataset(zuco_aggregated)
    dundee_aggregated = remove_duplicates_within_dataset(dundee_aggregated)

    provo_bos_train_agg, provo_bos_test_agg = split_sentences(provo_bos_aggregated)
    zuco_train_agg, zuco_test_agg = split_sentences(zuco_aggregated)
    dundee_train_agg, dundee_test_agg = split_sentences(dundee_aggregated)

    provo_bos_train_agg, provo_bos_test_agg = remove_duplicates_across_datasets(
        provo_bos_train_agg, provo_bos_test_agg
    )
    zuco_train_agg, zuco_test_agg = remove_duplicates_across_datasets(
        zuco_train_agg, zuco_test_agg
    )
    dundee_train_agg, dundee_test_agg = remove_duplicates_across_datasets(
        dundee_train_agg, dundee_test_agg
    )

    provo_bos_train = deaggregate_sentences(provo_bos_train_agg, provo_bos)
    provo_bos_test = deaggregate_sentences(provo_bos_test_agg, provo_bos)
    zuco_train = deaggregate_sentences(zuco_train_agg, zuco)
    zuco_test = deaggregate_sentences(zuco_test_agg, zuco)
    dundee_train = deaggregate_sentences(dundee_train_agg, dundee)
    dundee_test = deaggregate_sentences(dundee_test_agg, dundee)

    provo_bos_train.to_csv("provo_bos_train.csv", index=True)
    provo_bos_test.to_csv("provo_bos_test.csv", index=True)
    zuco_train.to_csv("zuco_train.csv", index=True)
    zuco_test.to_csv("zuco_test.csv", index=True)
    dundee_train.to_csv("dundee_train.csv", index=True)
    dundee_test.to_csv("dundee_test.csv", index=True)

    check = {
        "provo_bos": check_overlap(provo_bos_train, provo_bos_test),
        "zuco": check_overlap(zuco_train, zuco_test),
        "dundee": check_overlap(dundee_train, dundee_test),
    }
    print("Overlap check:")
    for dataset, result in check.items():
        print(f"{dataset}: {'No overlap' if result else 'Overlap detected'}")


if __name__ == "__main__":
    main()
