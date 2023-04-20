import os
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def _main():
    visualize_user_artist_play_count()


def _procedures():
    pass

def visualize_user_artist_play_count():
    print(matplotlib.get_backend()) #module://backend_interagg

    df = pd.read_csv('./data/lastfm-2k/user_artists.dat', sep='\t')
    cnt = []

    for index, row in df.iterrows():
        if row['userID'] == 2:
            cnt.append(row['weight'])
    plt.plot(cnt)
    plt.savefig('./fig.png')
    # plt.show()
    print(df)


if __name__ == '__main__':
    _main()
