from src.utilities import *
import sys
import warnings
warnings.filterwarnings("ignore")


def main():
    try:
        text_path = sys.argv[1]
        data = read_data(text_path)
    except:
        print("Error: Give me WhatsApp text!")
        exit()

    df = preprocess_data(data)
    df2 = process_nlp(df)

    draw_wordcloud(df2, save=True)
    draw_personFrequency(df2, save=True)
    draw_timeFrequency(df2, save=True)
    #positive_ratio, nlp_df = get_positive_ratio(df2)





if __name__ == "__main__":
    main()