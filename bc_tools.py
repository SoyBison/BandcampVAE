"""
A Cache of tools that collect data from bandcamp and do analysis.
"""

import requests
from bs4 import BeautifulSoup
import os
import re
import json
import multiprocessing as mp
from functools import partial
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import colorsys
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from data import Album, Session
import uuid
import time

BCAMPURL = 'https://{ARTIST}.bandcamp.com'


def load_artist_tags(loc='artist_tags'):
    with open(loc, 'r') as f:
        x = f.read().split('\n')
    return x


def add_artist_tag(tag, loc='artist_tags'):
    with open(loc, 'a+') as f:
        f.write(tag + '\n')


def cowdog(earl, loc='artist_tags', loops=4, n=0):  # 'earl' like 'url'
    """
    Collects names for you. Put in a release url as a starting point.
    """

    site = requests.get(earl)
    soup = BeautifulSoup(site.text, 'html.parser')
    footer = soup.find_all('li', attrs={'class': re.compile('recommended-album footer')})
    urls = {album.find('a', class_='go-to-album album-link')['href'] for album in footer}
    artists = {re.findall('(?<=//)[a-z0-9-_~]*(?=.)', tag)[0] for tag in urls}
    starter = re.findall('(?<=//)[a-z0-9-_~]*(?=.)', earl)[0]

    knowns = load_artist_tags(loc)
    artists.add(starter)
    for tag in artists:
        if tag not in knowns:
            add_artist_tag(tag, loc)

    if n < loops:
        for album in urls:
            cowdog(album, loc, loops, n + 1)


def get_album_covers(tag, loc='./covers/'):
    url = BCAMPURL.replace('{ARTIST}', tag)
    lib = requests.get(url)
    soup = BeautifulSoup(lib.text, 'html.parser')
    albums = soup.find('div', class_='leftMiddleColumns')
    try:
        albums = albums.find_all('a', class_=None)
    except AttributeError:
        return
    album_locs = []
    for a in albums:
        try:
            if url == a['href']:
                continue
            if a['href'].startswith('http'):
                album_locs.append(a['href'])
            else:
                album_locs.append(url + a['href'])
        except KeyError:
            pass

    session = Session()
    for album in album_locs:
        if list(session.query(Album.url).filter(Album.url == album)):
            continue
        try:
            alb = requests.get(album)
        except requests.exceptions.ConnectionError as e:
            with open('error.log', 'a') as f:
                f.write(f'Error reaching url: {album}\n {e}\n')
            time.sleep(300)
            try:
                alb = requests.get(album)
            except requests.exceptions.ConnectionError:
                continue
        except requests.exceptions.InvalidURL:
            with open('error.log', 'a') as f:
                f.write(f'Invalid URL detected: {album}\n')
            continue
        soup = BeautifulSoup(alb.text, 'html.parser')
        try:
            art = soup.find('div', id='tralbumArt')
            # Fucking noisecore artists trying to be twee by having an empty file as their album art ruining everything.
            if art.find("div", id="missing-tralbum-art"):
                continue
            imgloc = art.find('a')['href']
        except (KeyError, AttributeError):
            continue
        img = requests.get(imgloc, stream=True)
        img.raw.decode_content = True
        im = Image.open(img.raw)
        album_id = uuid.uuid4()

        titlesec = soup.find('h2', class_='trackTitle')
        artistsec = soup.find('h3').find('span')
        tags = soup.find_all('a', class_='tag')
        tags = [tag.text.strip() for tag in tags]
        try:
            album_title = re.findall(r'(?<=/)[a-z-_~0-9]*(?=[?#]|$)', album)[0]
        except IndexError as e:
            with open('error.log', 'a') as f:
                f.write(f'Error parsing {album}:\n {e}')
            continue
        data_obj = Album(
            id=album_id.hex,
            artist=artistsec.text.strip(),
            title=titlesec.text.strip(),
            tags=json.dumps(tags),
            url_title=album_title,
            store=tag,
            url=album
        )
        session.add(data_obj)
        im.save(os.path.join(loc, album_id.hex + '.jpg'))
        session.commit()

    return True


def album_cover_scrape(cover_loc='./covers/', artist_loc='artist_tags'):
    # pool = mp.Pool()
    worker = partial(get_album_covers, loc=cover_loc)
    artists = load_artist_tags(artist_loc)
    # list(pool.imap(worker, artists))
    for a in tqdm(artists):
        worker(a)
        time.sleep(0.5)
    return True


def make_colorgram(image_array, n=6):
    raise NotImplementedError('Deprecated')


def colorgram_from_file(loc, sink='./colorgrams/', n=40, del_orig=False):
    with open(loc, 'rb') as f:
        album = pickle.load(f)

    cg = make_colorgram(album['cover'], n)
    album.pop('cover', None)
    album['colorgram'] = cg

    if not os.path.exists(sink):
        os.mkdir(sink)

    with open(sink + str(hash(album['url'])), 'wb+') as f:
        pickle.dump(album, f)

    if del_orig:
        os.remove(loc)


def albums_to_colorgrams(source='./covers/', sink='./colorgrams/', n=40, del_orig=True):
    worker = partial(colorgram_from_file, sink=sink, n=n, del_orig=del_orig)
    p = mp.Pool()
    targets = [source + f for f in os.listdir(source)]
    nones = []
    for i in tqdm(p.imap(worker, targets), total=len(targets)):
        nones.append(i)


def brightness_plot(cols, tag):
    brightnesses = [(0.299 * r + 0.587 * g + 0.114 * b) for gram in cols for col in gram for r, g, b in col]
    brightnesses = np.array(brightnesses)
    rs, gs, bs = zip(*[[r, g, b] for gram in cols for col in gram for r, g, b in col])

    sns.distplot(brightnesses, color='black', hist=False)
    sns.distplot(rs, color='red', hist=False)
    sns.distplot(bs, color='blue', hist=False)
    sns.distplot(gs, color='green', hist=False).set_title(f'Brightness plot for {tag}-tagged album covers.')
    plt.show()


def luminance(color):
    r, g, b = color
    return 0.299 * r + 0.587 * g + 0.114 * b


def col_plot(cols, tag, size=48):
    colors_unorganized = np.array(sorted([col for box in cols for img in box for col in img],
                                         key=lambda rgb: colorsys.rgb_to_hls(*rgb)))
    colors_unorganized.resize(size * size * 3)
    colors_unorganized = colors_unorganized.reshape((size, -1, 3))
    colors_unorganized = colors_unorganized
    plt.matshow(colors_unorganized)
    plt.title(f'Color Plot for {tag}-tagged album covers.')
    plt.show()


def get_tag_cols(tag, data):
    tagged = data.iloc[[tag in tags for tags in data['tags']]]
    cols = tagged['cgram'].reset_index(drop=True)
    return cols


def rgb_color_converter(col, i=8, o=5):

    in_vals = 2 ** i - 1
    out_vals = 2 ** o - 1
    factor = out_vals / in_vals

    compressed = np.multiply(col, factor)
    compressed = np.round(compressed)
    converted = np.divide(compressed, factor)
    converted = np.round(converted)
    converted = np.int0(converted)

    return converted


def rgb2str(col):
    raw_hexes = [hex(c) for c in col]
    cleanhex = [c[2:].rjust(2, '0') for c in raw_hexes]
    return ''.join(cleanhex)


def str2rgb(col):
    assert len(col) == 6
    hexes = [col[i:i + 2] for i in range(0, 6, 2)]
    nums = tuple([int(h, 16) for h in hexes])
    return nums


def cg2doc(cg):
    cols = map(rgb2str, cg)
    return ' '.join(cols)


def doc2cg(doc):
    hexes = doc.split(' ')
    return [str2rgb(col) for col in hexes]


def get_topics(model, n_feats=8):
    feat_names = model['c_vec'].get_feature_names()
    comps = model['lda'].components_
    top_comps = [topic.argsort()[:-n_feats - 1:-1] for topic in comps]

    f = np.vectorize(lambda x: feat_names[x])

    str_tops = f(top_comps)

    outmat = []
    for row in str_tops:
        outtop = []
        for col in row:
            outtop.append(str2rgb(col))
        outmat.append(sorted(outtop, key=lambda rgb: colorsys.rgb_to_hls(*rgb)))

    return outmat


def mass_get_artists(tar='./targets', ring_size=3):
    with open(tar, 'r') as f:
        targets = f.readlines()
    targets = [t.split('?')[0] for t in targets]
    lil_doggie = partial(cowdog, loops=ring_size)
    for t in tqdm(targets):
        lil_doggie(t)


def collect_targets(tag, tar='./targets'):
    url = f'https://www.bandcamp.com/tag/{tag}'
    options = webdriver.FirefoxOptions()
    options.headless = True
    dr = webdriver.Firefox(options=options)
    dr.get(url)
    WebDriverWait(dr, 0.5)
    soup = BeautifulSoup(dr.page_source, 'html.parser')
    x = soup.findAll('a', target='_blank')
    targets = [ele.attrs['href'] for ele in x]
    targets = [t for t in targets if ('album' in t or 'track' in t) and ('&' not in t)]
    with open(tar, 'a+') as f:
        f.writelines('\n'.join(targets)+'\n')


if __name__ == '__main__':
    # with open('tags', 'r') as tf:
    #     tags = tf.read().splitlines()
    # for t in tags:
    #     collect_targets(t)
    # mass_get_artists()
    album_cover_scrape(cover_loc='/home/coen/Remote/Data/bandcamp/covers/')
