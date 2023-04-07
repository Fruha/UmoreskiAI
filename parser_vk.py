import requests
from utils import save_in_json, load_from_json
import os
from tqdm import tqdm

TOKEN_USER = os.getenv('VK_TOKEN')
VERSION = 5.131
DOMAIN = 'jumoreski'


def get_data_from_wall(token: str, domain: str, count: int, offset: int = 0, version: float = 5.131) -> list:
    """
    parsing posts in vk group
    Args:
        token: user vk token
        domain: name of group
        count: count of posts in batch (max 100)
        offset: offset relative to which posts are parsed
        version: version of va api
    Return:
        list of posts
    """
    params = {'access_token': token,
              'v': version,
              'domain': domain,
              'offset': offset,
              'count': count}
    response = requests.get('https://api.vk.com/method/wall.get',
                            params=params)

    data = response.json()['response']['items']
    return data


def prepare_data(data: list) -> dict:
    """
    Getting the necessary information from posts and converting it into a dictionary
    Arguments:
        data: list of posts
    Return:
        dict of posts where key is id
    """
    ans = {}
    for item in data:
        new_item = {
            'date': item['date'],
            'likes': item['likes']['count'],
            'reposts': item['reposts']['count'],
            'views': item['views']['count'] if 'views' in item.keys() else -1,
            'text': item['text'],
            'attachments': item['attachments']
        }
        ans[str(item['id'])] = new_item
    return ans


if __name__ == '__main__':
    batch_count = 100
    os.makedirs('./data', exist_ok=True)
    data = load_from_json('./data/parsed_data.json')
    try:
        for offset in tqdm(range(400*batch_count,100000,batch_count), total= int(50000/batch_count)):
            parsed_batch = get_data_from_wall(TOKEN_USER, DOMAIN, batch_count, offset)
            if not parsed_batch:
                break
            prepared_data = prepare_data(parsed_batch)
            data.update(prepared_data)
    except Exception as e:
        print(e)
    save_in_json(data, './data/parsed_data.json')
