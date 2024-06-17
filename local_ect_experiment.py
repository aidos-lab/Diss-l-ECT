from local_ect import *

def main():
    dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')
    xgb_model(dataset,
              radius1=True,
              radius2=False,
              ECT_TYPE='points',
              NUM_THETAS=64,
              DEVICE='cpu',
              metric='accuracy')


if __name__ == '__main__':
    main()