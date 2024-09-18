from local_ect import *

def main():
    # Range over dataset, radius

    # dataset = WikiCS(root='/tmp/wikics')

    # For the heterophilous datasets use ROC AUC!
    # dataset = HeterophilousGraphDataset(root='/tmp/Minesweeper',name='Minesweeper')
    # dataset = HeterophilousGraphDataset(root='/tmp/Tolokers',name='Tolokers')
    # dataset = HeterophilousGraphDataset(root='/tmp/Questions',name='Questions')

    # dataset = Actor(root='/tmp/actor')

    # Reddit is hard to compute!!
    # dataset = Reddit(root='/tmp/reddit')

    # dataset = Coauthor(root='/tmp/CS',name='CS')
    dataset = Coauthor(root='/tmp/Physics', name='Physics')

    res = xgb_model(dataset,
              radius1=True,
              radius2=False,
              ECT_TYPE='points',
              NUM_THETAS=64,
              DEVICE='cpu',
              metric='accuracy',
              subsample_size=None)


if __name__ == '__main__':
    main()