import pandas as pd
from utils import load_from_json
from sklearn.model_selection import train_test_split

def json_to_csv(json_path:str, csv_path:str) -> None:
    """
    1)Convert data from json to csv\n
    2)Adding target feature (likes_div_views, rep_div_views)\n
    3)Remove all posts with attachments
    """
    data_json = load_from_json(json_path)
    data_json = {key:val for key,val in data_json.items() if not val['attachments'] and val['views'] != -1}
    dataframe = pd.DataFrame.from_dict(data_json, orient='index')
    dataframe['likes_div_views'] = dataframe['likes'] / dataframe['views']
    dataframe['rep_div_views'] = dataframe['reposts'] / dataframe['views']

    df_train, df_val = train_test_split(dataframe)
    print('df_train.shape:', df_train.shape, 'df_val.shape:', df_val.shape)

    dataframe.to_csv(csv_path, index=False)

if __name__ == '__main__':
    json_to_csv('./data/parsed_data.json', './data/df_all.csv')