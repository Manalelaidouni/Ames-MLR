import random
import json
import requests
from contextlib import contextmanager

import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_squared_error

import streamlit as st
from streamlit_modal import Modal
import streamlit.components.v1 as components

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal



class StackingEnsemble:
    def __init__(self, X_train, y_train, X_test, base_models, meta_with_kfold, meta_model=None, num_folds=5, verbose_score=True):
        
        self.num_folds = num_folds
        self.kfolds = KFold(num_folds, shuffle=True, random_state=seed)
        self.base_models =  base_models
        self.meta_model = meta_model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        # Outputs
        self.test_predictions = {model_name: np.zeros((self.X_test.shape[0], num_folds)) for model_name in self.base_models} # shape = [n_samples]
        self.oof_predictions = {model_name: np.zeros(self.X_train.shape[0]) for model_name in self.base_models}
        self.scores = {model_name: [] for model_name in self.base_models.keys()}

        self.meta_with_kfold = meta_with_kfold
        self.verbose_score = verbose_score
        self.fitted_base_models_per_fold = [[] for _ in self.base_models]
        self.out_of_fold_predictions = np.zeros((self.X_train.shape[0], len(self.base_models)))


    def train_base_models(self):
        """
        Trains base models and computes out-of-fold predictions and scores.
        """
        self.X_train = self.X_train.reset_index(drop=True)  # reset index to avoid index error inside Kfold CV
        self.y_train = self.y_train.reset_index(drop=True)

        for idx, (model_name, model) in enumerate(self.base_models.items()):
            for fold, (train_idx, val_idx) in enumerate(self.kfolds.split(self.X_train, self.y_train)):
                X_trn, y_trn = self.X_train.loc[train_idx, :], self.y_train.loc[train_idx]
                X_val, y_val = self.X_train.loc[val_idx, :], self.y_train.loc[val_idx]

                model.fit(X_trn, y_trn)

                self.fitted_base_models_per_fold[idx].append(model)

                val_pred = model.predict(X_val)
                test_pred = model.predict(self.X_test)

                self.oof_predictions[model_name][val_idx] = val_pred
                self.test_predictions[model_name][:, fold] = test_pred

                # compute oof score for each model
                fold_score = StackingEnsemble.rmse(y_val, val_pred)
                self.scores[model_name].append(fold_score)


        if self.verbose_score:
            for model_name, fold_scores in self.scores.items():
                mean_score = np.mean(fold_scores)
                std_score = np.std(fold_scores)
                print(f'Out-of-fold mean score for {model_name} is: {np.round(mean_score, 4)}, std : {np.round(std_score, 4)}')

        self._construct_one_level_data()

    def _construct_one_level_data(self):
        """
        Constructs one-level data for meta-learner training and final predictions.
        """
        self.meta_input_data =  pd.DataFrame.from_dict(self.oof_predictions) # self.out_of_fold_predictions
        # compute the mean of each matrix in the values of the dictionary
        self.meta_predict_data = pd.DataFrame.from_dict({model_name: self.test_predictions[model_name].mean(axis=1) for model_name in self.base_models})

    @staticmethod
    def rmse(y, y_pred):
        return np.sqrt(mean_squared_error(y, y_pred))


    def train_meta_model(self):
        """
        Trains meta-learner model and computes final predictions.
        """
        if self.meta_model is None:
            raise ValueError('Meta model must be provided in class constructor if you invoke `train_meta_model` method.')

        if not self.meta_with_kfold:
            self.meta_model.fit(self.meta_input_data, self.y_train)
            self.fitted_meta_model = self.meta_model.copy()
            meta_test_y = self.meta_model.predict(self.meta_predict_data)

        else:
            meta_test_y = np.zeros((self.X_test.shape[0], self.num_folds))
            self.fitted_meta_model_per_fold = []
            meta_scores = []

            for fold, (train_idx, val_idx) in enumerate(self.kfolds.split(self.meta_input_data, self.y_train)):
                X_trn, y_trn = self.meta_input_data.loc[train_idx, :], self.y_train.loc[train_idx]
                X_val, y_val = self.meta_input_data.loc[val_idx, :], self.y_train.loc[val_idx]
                self.meta_model.fit(X_trn, y_trn)
                # save fitted meta model for each fold
                self.fitted_meta_model_per_fold.append(self.meta_model)

                # get oof predictions of meta_model
                oof_meta_preds = self.meta_model.predict(X_val)
                meta_score = StackingEnsemble.rmse(y_val, oof_meta_preds)
                meta_scores.append(meta_score)

                # get test predictions of meta_model
                test_meta_preds = self.meta_model.predict(self.meta_predict_data)
                meta_test_y[:, fold] = test_meta_preds

            meta_test_y = meta_test_y.mean(axis=1)
            print('final meta_test_y', meta_test_y.shape)
            if self.verbose_score:
                print(f'oof mean score for meta_learner is: {np.round(np.mean(meta_scores), 4)}, std : {np.round(np.std(meta_scores), 4)}')

        return meta_test_y


    def run_inference(self, X):
        """
        Runs inference on new data using trained base models and trained meta model.
        """
        all_base_model_preds = []
        for model_per_fold in self.fitted_base_models_per_fold:
            single_model_pred = np.column_stack([model.predict(X) for model in model_per_fold]).mean(axis=1)
            all_base_model_preds.append(single_model_pred)

        meta_test_input = np.column_stack(all_base_model_preds) #.mean(axis=1).reshape(1, -1)  shape with mean is (1, 1)
        # shape without mean is (1, 10)

        if not self.meta_with_kfold:
            prediction = self.fitted_meta_model.predict(meta_test_input)

        else:
            meta_preds = []
            for meta_model in self.fitted_meta_model_per_fold:
                meta_test_preds = meta_model.predict(meta_test_input)
                meta_preds.append(meta_test_preds)
            prediction = np.column_stack(meta_preds).mean(axis=1)

        final_prediction = np.expm1(prediction)

        return final_prediction


def footer():
    st.markdown(f"""
        <div class="container my-5">
        
        <hr style="border-top: 1px solid rgba(255,255,255,.5); width: 280px; margin: 20px auto;">
        <p style="color:#ffeeea; font-family: "Source Sans Pro", sans-serif;"> Made with   ❤️   by Manal EL Aidouni </p>
            <section class="mb-4 bg-dark p-4">
                <a class="btn btn-outline-light btn-floating m-1" style="background-color: #202020; color: white; border-radius: 50%;  padding: 9px; margin-right: 10px;box-shadow: 0 0 0 1px #ffeeea;width:14px; height:14px;" href="https://twitter.com/Manal_ELAI" role="button"><i class="fab fa-twitter"></i></a>
                <a class="btn btn-outline-light btn-floating m-1" style="background-color: #202020; color: white; border-radius: 50%;  padding: 9px; margin-right: 10px;box-shadow: 0 0 0 1px #ffeeea;width:14px; height:14px;" href="https://www.linkedin.com/in/manalelaidouni/" role="button"><i class="fab fa-linkedin-in"></i></a>
                <a class="btn btn-outline-light btn-floating m-1" style="background-color: #202020; color: white; border-radius: 50%;  padding: 9px; margin-right: 10px;box-shadow: 0 0 0 1px #ffeeea;width:14px; height:14px;" href="https://github.com/Manalelaidouni" role="button"><i class="fab fa-github"></i></a>
            </  section>
        </div>
        """, unsafe_allow_html=True)



def prediction_msg(predicted_price):
    min_price = 34900
    q25_price = 129975
    median_price = 163000
    q75_price = 214000
    max_price = 755000

    if predicted_price < min_price:
        comparison = "Wow! your imaginary house price value is lower than all of the houses we've seen in Ames dataset."
    elif min_price < predicted_price < q25_price:
        comparison = f"Interesting fact! It looks like the price of your imaginary house is lower than 75% of houses in the city of Ames."
    elif q25_price < predicted_price < median_price:
        comparison = f"Fun fact! It looks like your imaginary house price is worth more than 25% of houses in the city of Ames."
    elif median_price < predicted_price < q75_price:
        comparison = f"Fun fact! Your imaginary house price is worth more than 50% of houses in the city of Ames."
    elif q75_price < predicted_price < max_price:
        comparison = f"Fun fact! Your imaginary house price is worth more than 75% of houses in the city of Ames."
    else:
        comparison = "Wow! Your imaginary house price is worth more than all of the houses we've seen in Ames dataset."

    return comparison


@st.cache_resource()
def plot_map(ames_train_dataset):
    fig = px.scatter_mapbox(
        ames_train_dataset,
        lat="Latitude",
        lon="Longitude",
        hover_name='Neighborhood_full',
        color='SalePrice',
        mapbox_style='open-street-map',
        animation_frame='YearBuilt',
        color_continuous_scale='matter',
        category_orders={'YearBuilt': list(np.sort(ames_train_dataset['YearBuilt'].unique()))},
        range_color=(ames_train_dataset['SalePrice'].min(), ames_train_dataset['SalePrice'].max()),
        zoom=11.6
    )

    # sort the unique years in ascending order
    unique_years = sorted(ames_train_dataset['YearBuilt'].unique())

    for i, frame in enumerate(fig.frames):
        year = frame.name  # Get the current frame year

        # create a boolean mask based on the condition of year built
        mask = ames_train_dataset['YearBuilt'] <= int(year)

        # access the data using boolean indexing and update the frame attributes
        frame.data[0].lat = ames_train_dataset.loc[mask, 'Latitude']
        frame.data[0].lon = ames_train_dataset.loc[mask, 'Longitude']
        frame.data[0].hovertext = ames_train_dataset.loc[mask, 'Neighborhood_full']
        frame.data[0].marker.color = ames_train_dataset.loc[mask, 'SalePrice']

        # create hover text using the actual YearBuilt for each house in the current frame
        hover_text = "<b>%{hovertext}</b><br><br>" \
                    "Year Built: " + ames_train_dataset.loc[mask, 'YearBuilt'].astype(str) + "<br>" \
                    "Sale Price: $%{marker.color:,.2f}"

        # update the hover template  with the above text for the current frame
        frame.data[0].hovertemplate = hover_text

    fig.update_layout(height=600)
    
    # to add a framing border to the plot 
    fig.update_layout(
    shapes=[
        dict(
            type='rect',
            xref='paper',
            yref='paper',
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(
                color='black',  
                width=2  
            ))])
    
    st.plotly_chart(fig)


def preprocess_input(input_data):
    input_data['KitchenQual'] = -1 if input_data['KitchenQual'] == 'TA' else (1 if input_data['KitchenQual'] == 'Gd' else (0 if input_data['KitchenQual'] == 'Ex' else 3))
    input_data['MSSubClass_30'] = 1 if input_data['MSSubClass_30'] == 'Yes' else 0
    input_data['MSZoning_RM'] = 1 if input_data['MSZoning_RM'] == 'Yes' else 0
    input_data['ExterQual_TA'] = 1 if input_data['ExterQual_TA'] == 'Yes' else 0
    input_data['CentralAir_Y'] = 1 if input_data['CentralAir_Y'] == 'Yes' else 0
    input_data['GarageType_Detchd'] = 1 if input_data['GarageType_Detchd'] == 'Yes' else 0

    input_data = input_data.map(float).values.reshape(1, -1)
    return input_data



class CustomModal(Modal):
    def __init__(self, title, key, padding=20, max_width=None):
        super().__init__(title, key, padding, max_width)

    @contextmanager
    def container(self):
        
        if self.max_width:
            max_width = str(self.max_width) + "px"
        else:
            max_width = 'unset'

        st.markdown(
            f"""
            <style>
            div[data-modal-container='true'][key='{self.key}'] {{
                position: fixed;
                width: 100vw !important;
                left: 0;
                z-index: 1001;
            }}

            div[data-modal-container='true'][key='{self.key}'] > div:first-child {{
                margin: auto;
            }}

            div[data-modal-container='true'][key='{self.key}'] h1 a {{
                display: none
            }}

            div[data-modal-container='true'][key='{self.key}']::before {{
                    position: fixed;
                    content: ' ';
                    left: 0;
                    right: 0;
                    top: 0;
                    bottom: 0;
                    z-index: 1000;
                    background-color: rgba(0, 0, 0, 0.5); 
            }}
            div[data-modal-container='true'][key='{self.key}'] > div:first-child {{
                max-width: {max_width};
            }}

            div[data-modal-container='true'][key='{self.key}'] > div:first-child > div:first-child {{
                width: unset !important;
                background-color: rgba(252,252,252,255);
                padding: {self.padding}px;
                margin-top: {2*self.padding}px;
                margin-left: -{self.padding}px;
                margin-right: -{self.padding}px;
                margin-bottom: -{2*self.padding}px;
                z-index: 1001;
                border-radius: 5px;
            }}
            div[data-modal-container='true'][key='{self.key}'] > div > div:nth-child(2)  {{
                z-index: 1003;
                position: absolute;
            }}
            div[data-modal-container='true'][key='{self.key}'] > div > div:nth-child(2) > div {{
                text-align: right;
                padding-right: {self.padding}px;
                max-width: {max_width};
            }}

            div[data-modal-container='true'][key='{self.key}'] > div > div:nth-child(2) > div > button {{
                right: 0;
                width : 30px;
                height: 30px;
                color: red;
                margin-top: {2*self.padding + 12}px;


            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        with st.container():
            _container = st.container()
            if self.title:
                _container.markdown(
                    f"<h3>{self.title}</h3>", unsafe_allow_html=True)

            close_ = st.button('X', key=f'{self.key}-close')
            if close_:
                self.close()

        components.html(
            f"""
            <script>
            // STREAMLIT-MODAL-IFRAME-{self.key} <- Don't remove this comment. It's used to find our iframe
            const iframes = parent.document.body.getElementsByTagName('iframe');
            let container
            for(const iframe of iframes)
            {{
            if (iframe.srcdoc.indexOf("STREAMLIT-MODAL-IFRAME-{self.key}") !== -1) {{
                container = iframe.parentNode.previousSibling;
                container.setAttribute('data-modal-container', 'true');
                container.setAttribute('key', '{self.key}');
            }}
            }}
            </script>
            """,
            height=0, width=0
        )

        with _container:
            yield _container



