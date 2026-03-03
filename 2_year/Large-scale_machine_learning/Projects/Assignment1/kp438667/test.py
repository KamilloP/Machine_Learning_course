import train
import pandas as pd

def test_text_cleaning():
    text = "\r aS$df AD_AS& 66345]}_ ?/U\t"
    expected = "\r asdf adas  u\t"
    assert train.text_cleaning(text) == expected, f"Wrong result, got: '{train.text_cleaning(text)}', but expected: '{expected}'"
    print("Test text cleaning passed")

def test_dataset():
    df = pd.read_csv(f"amazon_reviews_2M.csv", header=None)
    df.columns = ["text", "label"]
    train.dataset_summary(df)
    df = train.dataset_cleaning(df)
    print(df.head())
    texts = df['text'].to_list()
    d = train.find_words(texts)
    for i, (k,v) in enumerate(d.items()):
        print(f"{i}th pair: ({k}, {v})")
        if i == 4:
            break
    # df.head()

test_text_cleaning()
test_dataset()