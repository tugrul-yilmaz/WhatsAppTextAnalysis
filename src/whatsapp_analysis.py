import pandas as pd
from src.utilities import *


data = read_data("src/_chat.txt")

df = preprocess_data(data)

df2 = process_nlp(df)

draw_wordcloud(df2, save=True)

draw_personFrequency(df2, save=True)

draw_timeFrequency(df2, save=True)




#positive_ratio, nlp_df = get_positive_ratio(df2)
