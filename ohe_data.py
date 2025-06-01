import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def final_data_prep(dataset):
    categorical_cols = dataset.select_dtypes(include=['object']).columns

    ct = ColumnTransformer(
        transformers=[
            ('cat_ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    transformed_data = ct.fit_transform(dataset)

    ohe_feature_names = ct.named_transformers_['cat_ohe'].get_feature_names_out(categorical_cols)

    non_cat_cols = [col for col in dataset.columns if col not in categorical_cols]

    all_columns = list(ohe_feature_names) + non_cat_cols

    final_dataset = pd.DataFrame(transformed_data, columns=all_columns)
    return final_dataset

joblib.dump(final_data_prep, 'data_preparation_function.pkl')