from local_ect import *

def main():
    dataset = HeterophilousGraphDataset(root='/tmp/Roman-empire',name='Roman-empire')
    xgb_model(dataset,
              radius1=True,
              radius2=True,
              ECT_TYPE='points',
              NUM_THETAS=64,
              DEVICE='cpu',
              metric='accuracy')


if __name__ == '__main__':
    main()