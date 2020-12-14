import config

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__=="__main__":
    df = pd.read_csv(config.input_file_path)
    df['kfold'] = -1
    #print(df.head())

    df = df.sample(frac=1).reset_index(drop=True)

    X = df.image_id.values
    y = df[["grapheme_root","vowel_diacritic", "consonant_diacritic"]].values

    msk = MultilabelStratifiedKFold(n_splits=5)

    for fold, (trn_, val_) in enumerate(msk.split(X,y)):
        print("TRAIN", trn_, "VAL: ", val_)
        df.loc[val_, "kfold"] = fold
    
    print(df.head())
    print(df.kfold.value_counts())
    df.to_csv("/home/nikhil/Videos/git_hub_projects/bengaliai-cv19/input/train_folds.csv", index=False)