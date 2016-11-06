
from glob import glob
from PIL import Image
import numpy as np
import pandas as pd

def testModel(model, testPath, output_path, run_name, save_to_csv=True):
    files = glob(testPath)
    names = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

    results = []
    n = 300000
    test_batch = 1000
    print('Starting predictions')
    for x in range(0, n / test_batch):
        images = files[x * test_batch: (x * test_batch) + test_batch]
        thisBatch = []
        for img in images:
            thisBatch.append(np.asarray(Image.open(img), dtype="float32") / 255.0)

        preds = model.predict(thisBatch)
        for i in range(0, test_batch):
            p = np.argmax(preds[i], 0) 
            results.append([images[i].split('/')[-1].split('.')[0], names[p]])

    print('Predictions done, saving file.')
    df = pd.DataFrame(results)
    df.columns = ['id','label']
    df['id'] = df['id'].apply(lambda X: int(X))
    df = df.sort_values('id')
    if save_to_csv:
        df.to_csv(output_path + run_name + '.csv', index=None)
    return(df)




def testModelMulti(model, testPath, output_path, run_name, save_to_csv=True):
    files = glob(testPath)
    names = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

    results = []
    n = 300000
    test_batch = 1000
    print('Starting predictions')
    for x in range(0, n / test_batch):
        images = files[x * test_batch: (x * test_batch) + test_batch]
        thisBatch = []
        for img in images:
            thisBatch.append(np.asarray(Image.open(img), dtype="float32") / 255.0)

        preds = model.predict(thisBatch)
        for i in range(0, test_batch):
            p = np.argmax(preds[i], 0) 
            results.append([images[i].split('/')[-1].split('.')[0], names[p]])

    print('Predictions done, saving file.')
    df = pd.DataFrame(results)
    df.columns = ['id','label']
    df['id'] = df['id'].apply(lambda X: int(X))
    df = df.sort_values('id')
    if save_to_csv:
        df.to_csv(output_path + run_name + '.csv', index=None)
    return(df)