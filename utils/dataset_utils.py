import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split


def balance_class_dataset(df_orig, target_label='income'):
    counter = Counter(df_orig[target_label])
    minority_class = None
    minimum = np.infty
    # print(counter)
    for c, v in counter.items():
        if v < minimum:
            minimum = v
            minority_class = c
    # print(f'Minority class {minority_class} \t Length of {minority_class} : {minimum} ')
    df_balanced = df_orig[df_orig[target_label] == minority_class].copy()
    classes = [label for label in df_orig[target_label].unique()
               if label != minority_class]
    for c in classes:
        df_new = df_orig[df_orig[target_label] ==
                         c].sample(minimum, random_state=42)
        df_balanced = pd.concat([df_balanced, df_new], axis=0)
    df_balanced = df_balanced.sample(
        frac=1, random_state=42).reset_index().drop('index', axis=1)
    return df_balanced


def stratify_attributes(df_orig, target, attribute, privileged_value):
    df_stratified = None
    for value in df_orig[target].unique():
        df_red = df_orig[df_orig[target] == value]
        df_alt = df_red[df_red[attribute] != privileged_value]
        df_privileged = df_red[df_red[attribute] == privileged_value]

        if len(df_privileged) > len(df_alt):
            df_privileged = df_privileged.sample(len(df_alt))
        else:
            df_alt = df_alt.sample(len(df_privileged))

        if df_stratified is None:
            df_stratified = pd.concat(
                [df_alt, df_privileged], axis=0)
        else:
            df_stratified = pd.concat(
                [df_stratified, df_alt, df_privileged], axis=0)
    df_stratified = df_stratified.sample(
        frac=1).reset_index().drop('index', axis=1)
    return df_stratified


def fit_scalers(df_orig, categorical_cols, numerical_cols, target):
    scalers = {}
    for cat in categorical_cols:
        scalers[cat] = OneHotEncoder(sparse=False).fit(df_orig[[cat]])
    for cat in numerical_cols:
        scalers[cat] = StandardScaler().fit(df_orig[[cat]])
    scalers[target] = LabelBinarizer().fit(df_orig[[target]])
    return scalers


def encode_categorical(df_orig, attribute, ohe):
    df = df_orig.copy()
    cols = df.columns.to_list()
    found_idx = -1
    for i, c in enumerate(cols):
        if c == attribute:
            found_idx = i
            break
    cols_before = cols[:found_idx]
    cols_after = cols[found_idx+1:]
    df_scaled = df.copy()
    p = ohe.transform(df_scaled[[attribute]])
    features = ohe.get_feature_names_out()
    cols_scaled = [c for c in cols_before]
    cols_scaled = cols_scaled + list(features)
    cols_scaled = cols_scaled + cols_after
    ohe_df = pd.DataFrame(p, columns=features, dtype=int)

    df_scaled = pd.concat([df_scaled.drop(attribute, axis=1), ohe_df], axis=1)[
        cols_scaled]

    return df_scaled


def encode_numerical(df_orig, attribute, sc):
    df = df_orig.copy()
    df[attribute] = sc.transform(df[[attribute]])
    return df


def encode_target(df_orig, attribute, lb):
    df = df_orig.copy()
    df[attribute] = lb.transform(df[[attribute]])
    return df


def encode_dataset(df, cat_cols, num_cols, target, scalers):
    df_enc = df.copy()
    for cat in cat_cols:
        df_enc = encode_categorical(df_enc, cat, scalers[cat])
    for num in num_cols:
        df_enc = encode_numerical(df_enc, num, scalers[num])
    df_enc = encode_target(df_enc, target, scalers[target])
    return df_enc


def decode_categorical(df_orig, attribute, ohe):
    df = df_orig.copy()
    cols = df.columns.to_list()
    found = False
    cols_before = []
    cols_after = []
    for i, c in enumerate(cols):
        if c.startswith(attribute):
            found = True
        else:
            if not found:
                cols_before.append(c)
            else:
                cols_after.append(c)
    target_cols = [c for c in df.columns if c.startswith(attribute)]
    decoding = pd.DataFrame(ohe.inverse_transform(
        df[target_cols]), columns=[attribute])
    cols = cols_before + [attribute]+cols_after
    df_decoded = pd.concat(
        [df.drop(target_cols, axis=1), decoding], axis=1)[cols]
    return df_decoded


def decode_numerical(df_orig, attribute, sc):
    df = df_orig.copy()
    df[attribute] = sc.inverse_transform(df[[attribute]])
    return df


def decode_target(df_orig, attribute, lb):
    df = df_orig.copy()
    df[attribute] = lb.inverse_transform(df[[attribute]])
    return df


def decode_dataset(df, cat_cols, num_cols, target, scalers):
    df_dec = df.copy()
    for cat in cat_cols:
        df_dec = decode_categorical(df_dec, cat, scalers[cat])
    for num in num_cols:
        df_dec = decode_numerical(df_dec, num, scalers[num])
    df_enc = decode_target(df_dec, target, scalers[target])
    return df_enc


def original_bias(df, target, bias_orig, len_new_data, sensitive, privileged_value='', binarize=True):
    df_all = None

    for income in df[target].unique():
        df_bias = None
        current_len = int(
            len_new_data*(len(df[df[target] == income]) / len(df)))
        df_target = df[df[target] == income]

        sensitive_values = df_target[sensitive].unique()
        if (len(sensitive_values) > 2) and (binarize):
            bias = bias_orig[sensitive][privileged_value][income]
            b_privileged = int(current_len*bias)
            df_bias = df_target[df_target[sensitive]
                                == privileged_value].sample(b_privileged)

            b_alt = int(current_len * (1-bias))
            df_alt = df_target[df_target[sensitive]
                               != privileged_value].sample(b_alt)

            df_bias = pd.concat(
                [df_bias, df_alt], axis=0)

        else:
            for privileged in sensitive_values:
                bias = bias_orig[sensitive][privileged][income]
                b = int(current_len*bias)
                if df_bias is None:
                    df_bias = df_target[df_target[sensitive]
                                        == privileged].sample(b)
                else:
                    df_bias = pd.concat(
                        [df_bias, df_target[df_target[sensitive]
                                            == privileged].sample(b)],
                        axis=0)

        if df_all is None:
            df_all = df_bias.copy()
        else:
            df_all = pd.concat([df_all, df_bias], axis=0)

    return df_all.sample(frac=1).reset_index().drop('index', axis=1)


def create_global_test_set(base_dir, sensitive, case):
    df_global = None
    df_global_enc = None
    for node in range(10):
        if case == 'Balanced':
            filename = f'Balanced/adult_fake_node_{node}_test_balanced_small.csv'
            filename_enc = f'Balanced/adult_fake_node_{node}_test_balanced_enc_small.csv'
        elif case == 'Stratified':
            if node < 5:
                filename = f'Balanced/adult_fake_node_{node}_test_balanced_small.csv'
                filename_enc = f'Balanced/adult_fake_node_{node}_test_balanced_enc_small.csv'
            else:
                filename = f'Stratified/{sensitive}/adult_fake_node_{node}_test_strat_small.csv'
                filename_enc = f'Stratified/{sensitive}/adult_fake_node_{node}_test_strat_enc_small.csv'
        else:
            if node < 5:
                filename = f'Balanced/adult_fake_node_{node}_test_balanced_small.csv'
                filename_enc = f'Balanced/adult_fake_node_{node}_test_balanced_enc_small.csv'
            else:
                filename = f'OriginalBias/{sensitive}/adult_fake_node_{node}_test_origbias_small.csv'
                filename_enc = f'OriginalBias/{sensitive}/adult_fake_node_{node}_test_origbias_enc_small.csv'
        df = pd.read_csv(f'{base_dir}/node_{node}/{filename}')
        df_enc = pd.read_csv(f'{base_dir}/node_{node}/{filename_enc}')
        target_label = df.pop('income').to_frame()
        df = pd.read_csv(f'{base_dir}/node_{node}/{filename}')
        if sensitive == 'race':
            df_race = df.copy()
            df_race['race'] = df_race['race'].apply(
                lambda x: 'Not-White' if x != 'White' else x)

            df_tr, df_test_race, y_tr, y_test = train_test_split(df_race, target_label,
                                                                 stratify=df_race[[
                                                                     sensitive, "income"]], test_size=0.1,
                                                                 random_state=42)
            df_test = df.iloc[df_test_race.index.values]
        else:
            df_tr, df_test, y_tr, y_test = train_test_split(df, target_label, stratify=df[[sensitive, "income"]], test_size=0.1,
                                                            random_state=42)
        df_test_enc = df_enc.iloc[df_test.index.values]
        if df_global is None:
            df_global = df_test.copy()
            df_global_enc = df_test_enc.copy()
        else:
            df_global = pd.concat([df_global, df_test], axis=0)
            df_global_enc = pd.concat([df_global_enc, df_test_enc], axis=0)
    df_global = df_global.reset_index().drop('index', axis=1)
    df_global_enc = df_global_enc.reset_index().drop('index', axis=1)
    return df_global, df_global_enc
