import pickle
import numpy as np
import random





def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def uniformnoise(x, y, alpha):
    images = x
    for counter, image in enumerate(images):
        if counter%1000 == 0:
            print(counter)
            print(len(x))
        for i in range(alpha):
            x = np.concatenate((x, np.column_stack(image)))
            y.append(random.randint(0,1000))
    imagelabelpairs = {'data': None, 'labels': None}
    imagelabelpairs['data'] = x
    imagelabelpairs['labels'] = y
    print("alpha size: " + str(len(imagelabelpairs['data'])))
    file = 'train_alpha_' + str(alpha)
    with open(file, 'wb') as f:
        pickle.dump(imagelabelpairs, f)


def main():
    x = []
    y = []

    train_1 = unpickle("data32x32/train_data_batch_1")
    train_1_x = train_1['data']
    train_1_y = train_1['labels']
    #train_2 = unpickle("data32x32/train_data_batch_2")
    #train_2_x = train_2['data']
    #train_2_y = train_2['labels']
    #train_3 = unpickle("data32x32/train_data_batch_3")
    #train_3_x = train_3['data']
    #train_3_y = train_3['labels']

    x = train_1_x[0:10000]
    #x.append(train_2_x)
    #x.append(train_3_x)
    #trainX = np.concatenate(x)

    y = train_1_y[0:10000]
    #y.append(train_2_y)
    #y.append(train_3_y)
    #trainY = np.concatenate(y)

    for i in range(5):
        print("preprocessing alpha " + str(i+1))
        uniformnoise(x, y, i+1)
    #print(len(train_1['data']))
    #print(len(train_1['data'])+len(train_2['data'])+len(train_3['data']))


if __name__ == "__main__":
	main()
