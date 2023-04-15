import nltk
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import seaborn as sns
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
nltk.download('stopwords')
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import warnings
warnings.filterwarnings("ignore")


def read_data(path):
    with open(path, encoding="utf-8-sig") as file:
        data = file.readlines()
        return data


def preprocess_data(data, line=5):
    # ilk kaç satır temizlenmeli ?
    data = data[line:]

    # boş satırlar silinmeli
    data2 = [i for i in data if len(i) != 1]

    # birden fazla satırdan oluşan mesajların düzeltilmesi
    new_data = []
    gecici=""
    for idx, i in enumerate(data2):
        cc = len(i.split(": "))
        cl = i.split(": ")
        if cc > 1:
            gecici = cl[0]
            new_data.append(": ".join(cl))
        if cc <= 1:
            # new_str = gecici + cl[0]
            new_data.append(": ".join([gecici, cl[0]]))

    # zoom davetiyeleri vb istenmeyen mesajların yakalanması
    new_data2 = [i for i in new_data if i[0] == "["]
    # numaralar ve mesajların ayırılması
    new_data3 = [i.split(": ")[1] for i in new_data2]
    new_data4 = [i.split(": ")[0].split("] ")[1] for i in new_data2]
    new_data5 = [i.split(": ")[0].split("] ")[0].split("[")[1] for i in new_data2]

    df = pd.DataFrame(new_data4, columns=["numbers"])
    df["text"] = new_data3
    df["date"] = new_data5

    return df


def process_nlp(dataframe, language="turkish", ignore_words=[]):
    dataframe['text'] = dataframe['text'].str.lower()
    dataframe['text'] = dataframe['text'].str.replace('[^\w\s]', '')
    sw = stopwords.words(language)
    sw = sw + ignore_words
    dataframe['text'] = dataframe['text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    temp_df = pd.Series(' '.join(dataframe['text']).split()).value_counts()
    drops = temp_df[temp_df <= 1]
    #dataframe['text'] = dataframe['text'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))
    dataframe["ddate"] = dataframe["date"].apply(lambda x: x.split(" ")[0])

    return dataframe



def draw_wordcloud(dataframe, save=False, word_count=50):
    text = " ".join(i for i in dataframe.text)
    if save:
        wordcloud = WordCloud(max_font_size=50,
                              max_words=word_count,
                              background_color="white",
                              colormap="magma").generate(text)
        plt.figure(figsize=(20,10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig('wordcloud.jpg')
        plt.show(block=True)
    else:
        wordcloud = WordCloud(max_font_size=50,
                              max_words=word_count,
                              background_color="white",
                              colormap="magma").generate(text)
        plt.figure(figsize=(20,10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show(block=True)


def draw_personFrequency(dataframe, save=True):
    dd = dataframe["numbers"].value_counts()
    dd = pd.DataFrame(dd)
    dd = dd.reset_index()
    dd.columns = ["numbers", "counts"]
    if save:
        plt.figure(figsize=(20, 10))
        sns.barplot(data=dd, y="numbers", x="counts", palette="ch:r=.6,s=-.2")
        plt.savefig('personFrequency.jpg')
        plt.show(block=True)
    else:
        plt.figure(figsize=(20, 10))
        sns.barplot(data=dd, y="numbers", x="counts", palette="ch:r=.6,s=-.2")
        plt.show(block=True)




def draw_timeFrequency(dataframe, save=False):
    dd = dataframe.groupby("ddate").agg({"text": "count"}).reset_index()
    dd["ddate"] = dd["ddate"].apply(lambda x: x.replace(".", "/"))
    dd["ddate"] = pd.to_datetime(dd["ddate"], dayfirst=True)
    dd = dd.sort_values(by="ddate")
    dd = dd.set_index("ddate")
    if save:
        dd.plot()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("timeFrequency.jpg")
        plt.show(block=True)

    else:
        dd.plot()
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show(block=True)



def get_positive_ratio(dataframe):
    nlp_df = dataframe.copy()
    # linklerin silinmesi
    index_links = [idx for idx, row in nlp_df.iterrows() if "http" in row["text"]]
    nlp_df.drop(index_links, inplace=True)

    # sayıların silinmesi
    nlp_df["text"] = nlp_df["text"].str.replace('\d+', '')

    # boş satırların silinmesi
    nlp_df["text"].apply(lambda x: x.strip())
    nlp_df = nlp_df[nlp_df["text"] != ""]
    print("veri temizlendi")


    model = AutoModelForSequenceClassification.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
    tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
    sa = pipeline("sentiment-analysis", tokenizer=tokenizer, model=model)
    print("model indirildi")

    sentiment_list = []
    for idx, row in nlp_df.iterrows():
        p = sa(row["text"])
        sentiment_list.append(p)
        print(f"{idx} tamamlandı")
    print("veriler işlendi")

    df_sentiment = pd.DataFrame(sentiment_list)
    nlp_df["label"] = [i[0]["label"] for i in sentiment_list]
    nlp_df["score"] = [i[0]["score"] for i in sentiment_list]

    nlp_df["new_label"] = ["negative" if row["score"] > 0.90 and row["label"] == "negative" else "positive" for idx, row in nlp_df.iterrows() ]

    # olumlu/olumsuzluk oranının belirlenmesi
    number_list = nlp_df["numbers"].unique()
    number_and_value = {}
    for i in number_list:
        value = nlp_df[nlp_df["numbers"] == i]["new_label"].value_counts()
        number_and_value[i] = value
    c = pd.DataFrame(number_and_value).T
    c.dropna(inplace=True)
    c["ratio"] = c["positive"] / (c["negative"] + c["positive"])
    print("model grafiği çizildi")
    sns.barplot(x=c.index, y=c["ratio"])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("positive_ratio.jpg")
    plt.show(block=True)
    return c, nlp_df
