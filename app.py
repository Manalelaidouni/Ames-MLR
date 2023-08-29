import random
import joblib
import time
import json
import requests
import warnings
warnings.filterwarnings('ignore')
import base64

import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image, ImageOps

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import streamlit as st
import streamlit.components.v1 as components
from streamlit_modal import Modal

from utils import StackingEnsemble, plot_map, preprocess_input, CustomModal, prediction_msg, footer



# page config
st.set_page_config(page_title = "House Price Prediction", 
                   page_icon  = "noto-v1:house", 
                   layout = "centered", 
                   initial_sidebar_state = "expanded", 
                   menu_items = None,        
)


st.markdown("""
    <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.15.4/css/all.css"
        integrity="sha384-DyZ88mC6Up2uqS4h/KRgHuoeGwBcD4Ng9SiP4dIRy0EXTlnuz47vAwmeGwVChigm" crossorigin="anonymous">
    """, unsafe_allow_html=True)
        
st.markdown("""<link href="plugins/bootstrap/css/bootstrap.min.css" rel="stylesheet">""", unsafe_allow_html=True)

with open("style.css") as f:
    styles = f.read()

st.markdown(f'<style>{styles}</style>', unsafe_allow_html=True)

@st.cache_data()
def read_gif(path):
    file_ = open(path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    return data_url

@st.cache_resource
def load_model():
    return joblib.load('model/stacking_ensemble.pkl')
  
data_url = read_gif("./animations/house.gif")
       
modal_out = CustomModal(key="Output Key", title="✔ Model Result ", max_width=450) 

with st.sidebar:
    # side bar with variable selection for user
    st.subheader('Predict Price of Your Ideal Home')
    st.markdown('Answer the following questions about your fictional dream house in the real estate market of Ames.')
    st.text("")
    with st.form(key='my_form'):

        LotArea = st.slider(f"What is the size of the lot area of the house in square feet?", 1300, 215245)

        OverallQual = st.slider('How would you rate the overall material and finish of the house?', min_value=1, max_value=10, step=1)

        OverallCond = st.slider('How would you rate the overall condition of the house?', min_value=1, max_value=9, step=1)
        
        YearBuilt = st.slider('What is the year the house was built?', 1872, 2010, step=1)

        FullBath =  st.slider('What is the total number of full bathrooms in the house?', min_value=0, max_value=3, step=1)
        BsmtFullBath = st.slider('If the house has a basement, what is the number of full bathrooms in the basement?', min_value=0, max_value=3, step=1)
        
        KitchenQual =  st.slider('How would you rate the kitchen quality of the house?', min_value=1, max_value=6, step=1) # After encoding

        TotRmsAbvGrd =  st.slider('How many total rooms (excluding bathrooms) are there above ground?', min_value=1, max_value=14, step=1)
        Fireplaces =  st.slider('How many fireplaces are there in the house?', min_value=0, max_value=3, step=1)
        GarageCars =  st.slider('What is the capacity of the garage in terms of car parking?', min_value=0, max_value=4, step=1)

        WoodDeckSF = st.slider('What is the size of the wood deck area in square feet?', 0, 850)
        OpenPorchSF = st.slider('What is the size of the open porch area in square feet?', 0, 540)

        CentralAir_Y = st.selectbox('Does the house have central air conditioning?', ['Yes', 'No'])
        GarageType_Detchd = st.selectbox('Is the garage detached from the house ?',  ['Yes', 'No'])
        MSZoning_RM = st.selectbox('Does the house belong to a residential medium density zone?', ['Yes', 'No'])
        ExterQual_TA = st.selectbox('Is the quality of the material on the exterior considered average or typical?', ['Yes', 'No'])

        st.markdown("")
        st.markdown("<style> div.st.form_submit_button> button:first-child {height : 1em} </style>", unsafe_allow_html = True)
        
        submit_button = st.form_submit_button(label='Predict')
        st.markdown("")
        footer()




if submit_button:
    # conclude MSSubClass from year of built       
    MSSubClass_30 = 'Yes' if  YearBuilt <= 1945  else 'No'
    feature_list = [LotArea, OverallQual, OverallCond, YearBuilt, BsmtFullBath, FullBath, KitchenQual,
                TotRmsAbvGrd, Fireplaces, GarageCars, WoodDeckSF, OpenPorchSF,  MSSubClass_30,
                MSZoning_RM, ExterQual_TA, CentralAir_Y, GarageType_Detchd]
    
    feature_names = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'BsmtFullBath', 'FullBath', 'KitchenQual',
            'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'MSSubClass_30',
            'MSZoning_RM', 'ExterQual_TA', 'CentralAir_Y', 'GarageType_Detchd']
    
    input_data = pd.Series(feature_list, index=['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'BsmtFullBath',
    'FullBath', 'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
    'WoodDeckSF', 'OpenPorchSF', 'MSSubClass_30', 'MSZoning_RM',
    'ExterQual_TA', 'CentralAir_Y', 'GarageType_Detchd'])

    single_instance = preprocess_input(input_data)
    stacking_model = load_model()
    prediction = stacking_model.run_inference(single_instance) # returns numpy array
    msg = prediction_msg(prediction[0])
    # save the result in session_state because of the refresh of st.experimental_rerun()
    
    st.session_state['msg'] = msg
    st.session_state['prediction'] =  f'{int(prediction[0]):,}'
    st.session_state['data_url']  =  data_url
    modal_out.open()


if modal_out.is_open():
    with modal_out.container():
        st.write(" ")
        st.write('The estimated price of your house is $', st.session_state['prediction'], '.')
        st.write(st.session_state['msg'])
        data_url = st.session_state['data_url']
        st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif"  width="200" height="200" style="vertical-align:middle;;margin-left:110px">',unsafe_allow_html=True,)
        st.write(" ")



        
ames_train_dataset = pd.read_csv('data/ames_train_data_with_geo_location.csv')

st.title("From Data to Dollars : Discover the Value of Your Dream Home")
st.header(" Unveiling the Factors and Hidden Patterns Behind Home Prices and Finally Forecasting Price in Ames, Iowa.")

st.text("")
st.markdown("""
> - Investigate the various factors that influence house price negotiations and gain insights into the real estate market.
> - Explore the worth of your imaginary dream house in Ames, Iowa using advanced data analysis techniques.
""")

st.markdown("")

st.markdown("")

st.markdown("""Let's explore the houses in Ames. In each frame, when you hover over a specific house, the hover text will display the YearBuilt of that house, along with the neighborhood name and the sale price.
Make sure to click on the play button to see the expansion of neighborhoods throughout the years.""") 

plot_map(ames_train_dataset)

st.markdown("")

expander = st.expander(":green[*Expand this toggle to see how I got the geo-location data for each house, considering that the Kaggle dataset does not include this information.*]")
expander.markdown("""


    I started by merging two datasets, the Kaggle house price competition data `ames_train_dataset` which contains 1460 observations and the Ames dataset `ames_sub` that I generated by loading the tidymodels framework using R, this dataset is much larger with 2930 observations containing the essential longitude and latitude which we need for our analysis.

    For the merging process, I dropped unnecessary columns and renamed others to ensure successful merging based on shared columns, this resulted in `merged_data` with 1466 observations.

    The real challenge was to make `merged_data` identical to `ames_train_dataset` and identify the overlap while correcting erroneous data. This process involved manual inspection to spot problematic lines. The line indices were constantly readjusted after each modification to `merged_data`, and this made it impossible to detect all problematic lines at the same .

    **It's important to note that some of the problematic lines are because of repeated lines,  however, these duplicates couldn't be detected using `df.duplicates()` in code, therefore manual inspection was the only effective method to identify them.**

    To tackle this, I iteratively used the csv-diff website to visualize differences and compare the datasets. I addressed each problematic line individually, inspecting the errors, and fixing them accordingly. After each correction, I saved the modified DataFrames as csv files to reload them into the csv-diff website for the next round of inspection and error correction.

    At the end, `merged_data` had all observations from `ames_train_dataset` Kaggle dataset with their corresponding longitude and latitude coordinates.


    """)

st.markdown("")

st.markdown("""
**Insights**  :

Home price in Ames, Iowa  range from \$34,900 to \$755,000 with an average of \$180,921.

- We observe that most houses with a sale price higher than 500K are built after 1990 in the Northridge and NorthRidge Heights neighborhoods, these neighborhoods are relatively newer and stand out as having the most expensive houses. Other highly priced houses can also be found in Stone Brook.

- **Expansion of neighborhoods throughout the years** : It appears that 'OldTown' is where the city's tale began, which explains its name. Subsequently, construction expanded to areas such as 'Iowa Rail Road' and 'CrawFord', eventually reaching the 'NorthRidge Heights' neighborhood.

- **An intriguing outlier** : something that caught my attention was one of the oldest houses in 'OldTown' built in 1880 that stands out for its high value, so I wanted to dig deeper to understand the reason behind this and it turns out the house is very large with a lot area of 18,386 surpassing the 75-th percentile of value 11,603. This house also has favorable features such as an overall quality rating of 7 and an overall condition rating of 9 despite its age which is likely due to a remodeling done in 2002.

""")


st.markdown("")
st.markdown("")
st.markdown('### Exploring main predictors that influence house price')
st.markdown("")
st.markdown("")
image = Image.open('images/regplot.png')

st.image(image, caption='')

st.markdown("")
st.markdown("""Some observations that stand out: 

- It looks like **Lot Area** has a positive correlation with the house sale price, where houses with extremely large lot with 200,000 square feet are worth  $400,000 higher than houses with a lot area of 25,000 square feet.
    
    
- The average price difference between a house that is rated a quality score of 10 compared to one that is assigned a rating score of 2 is approximately $600,000. This means that remodeling the material and finish of the house would significantly increase the house value.

- Strangely, the **overall condition** of the house is not correlated with the value of the house.

- **The number of baths in the basement** increases the house value by a small value. However **number of baths in the entire house** is positively correlated with house price where a house with 3 baths is worth $100,000 more than houses with a single bath. The takeaway is constructing more baths in the house would significantly increase the house value.

- **Total number of rooms above grade**  is also positively correlated with house price in that houses with more rooms are worth significantly higher than houses with lesser number of rooms, for example, a house that has 14 rooms is worth $250,000 more than a house with 2 rooms.
- **Year Built** : As observed above in the map plot, newly constructed houses are worth higher than older houses. Consequently, selling the house  sooner would be a better idea.

- Similarly, Houses with central air conditioning are worth an average of $100,000 more than houses without central AC. 
""")

st.markdown("")
st.markdown("""
Of course, the observations drawn above  are  based on the individual relationships between each predictor and response variable, however this is a multiple Linear regression (MLR) problem where we  model the joint influence of all predictors $X$ on the response variable $y$ using a linear equation and then use this  to make  predictions of  $y$ from values of $X$.

We will see how all  of  these predictors interact together to impact house price and check if the observations made above are held true when considering the collective influence of all predictors.

A thorough  explanation of  MLR is available in the notebook where we explain the Least Squares OLS method, MLR coefficients, their formula, their interpretation, the characteristics of optimal coefficients, and the MLR assumptions that need to be satisfied. This brings us to the MLR assumptions that we will be checking below.

""")
st.markdown("")
st.markdown("")

st.markdown('###  Satisfying Linear Regression Assumptions ')
st.markdown("")

# Linearity

st.markdown("##### Outlier Detection and Removal to Achieve Linearity ")
st.markdown("")
st.markdown("> *The usual method to check for linearity is by plotting each of the independent variables against the predictor variable, however in the case of multiple linear regression problem with a large number of predictors, this quickly become not practical, therefore we examine the residuals plots for pattern to detect non-linear relationships between the response variable and the predictors.*")
st.markdown("")
st.markdown("I was able to achieve linearity by first applying a non-linear transformation (logarithm function) to the target variable and then removing outliers with large residuals. ")
st.markdown("")
st.image(Image.open('images/before_y.png'), caption='')
st.image(Image.open('images/after_y.png'), caption='Before and after applying a non-linear transformation to the response variable to achieve linearity assumption.')
st.markdown("")
st.markdown(" > *The normalization of the target variable had a positive impact on model performance with the Adjusted R² value increasing from 0.909 to 0.931.* ")

#Effect of normalizing target variable on model performance** : It looks like the Adjusted R² value has improved from 0.909 to 0.931")

st.markdown("")
st.image(Image.open('images/before_outliers.png'), caption='')
st.image(Image.open('images/after_outliers.png'), caption=' Actual vs Fitted plot on the left along with Residuals vs Fitted plot  on the right to visually inspect for the outliers')
st.markdown("")

st.markdown("- In the **Actual vs Fitted plot**, we plot the actual values of the response variable against the predicted values from the model. The points are scattered perfectly around the diagonal line and not in a non-linear pattern or a curve form, which suggests that the linearity assumption is met.")
st.markdown("- In the **residuals vs fitted plot**, we plot the residuals  against the predicted values from the model. It looks like the points are scattered randomly around the horizontal 0 line, this further confirms the linearity claim.")
st.markdown("")
st.markdown("> *After the removal of outliers from the training data we were able to achieve a notable improvement in the Adjusted R² value, which increased from 0.9311 to 0.9751. This indicates that the presence of outliers was negatively influencing the model's fit and predictive accuracy.* ")
st.markdown("> *We can conclude that the  linearity assumption is met.*")




# Multicolinearity
st.markdown("")
st.markdown("")
st.markdown("##### No Multicolinearity Assumption ")
st.markdown("")

st.markdown("""Multicollinearity occurs when two or more predictor variables are highly correlated, leading to inflated standard error of the regression coefficients causing unstable coefficients and unreliable p-values.

We can inspect for multicollinearity using a bivariate Pearson correlation matrix (pair-wise correlation coefficient) that is visualized in a heat map, where Pearson correlation coefficient is used to measure the strength of linear correlation between each pair of predictors. 


Another approach is to use the variance inflation factor (VIF) of each numerical predictor variable. VIF of a certain variable measures the extent to which the variance of a regression coefficient is inflated when multicollinearity exists in the regression model. 


""")


st.markdown("""
Here was the action plan: we needed to identify and remove problematic predictors. As a result we took into consideration the following factors to make an informed decision on which predictors to drop:

- **Large VIF values**: We computed the Variance Inflation Factor (VIF) values for each predictor and identified those with VIF values greater than 10, indicating high correlation with other predictors. These predictors were then considered for removal.

- **High Pearson correlation coefficient**: Using the heatmap, we identified pairs of predictors with a high pairwise Pearson correlation coefficient greater than 0.5,  as they measure similar information. These predictors were also flagged for potential removal.

Here was the correlation heatmap:
 """)


st.markdown("")
st.image(Image.open('images/heatmap_before.png'), caption='')
st.markdown("")

st.markdown("""
**Statistical significance between the predictor and response variable**: We evaluated the significance of predictors by computing the p-value of the T-test. The t-test statistic helped us determine the correlation between the response and the predictor variable. Our goal is to keep only those variables that are significant in predicting the target variable and have a meaningful relationship with it. Non-significant predictors, which did not contribute significantly to the model, were considered for removal.

**Correlation between each predictor and response variable**: We assessed the correlation between each predictor and the response variable. Predictors with the least association with the response variable are less useful in predicting the outcome and could negatively impact the overall performance of the model, therefore they were also considered for removal.
""")

st.markdown("")

st.markdown("Given the above information, Variables we dropped based on VIF value, Pearson correlation, statistical significance and association with target variable :")

st.markdown("I implemented all of these steps in this [class](https://gist.github.com/Manalelaidouni/172324a4181533d284bfe9e1695a00d5).")

st.markdown("""
- `LotFrontage` has low correlation with `y` and has high VIF value, it's also non-significant.
- `BsmtFinSF2` has low correlation with `y` and has high VIF value, it's also non-significant.
- `BsmtUnfSF` has low correlation with `y` and has high VIF value, it's also non-significant.
- `MiscVal` has low correlation with `y` and has high VIF value, it's also non-significant.
- `LotFrontage` has a high VIF value and it's also non-significant.
- `PoolArea` has low correlation with `y` and has high VIF value, it's also non-significant.
""")

st.markdown("After recalculating the VIF values following the initial removal of predictors with high VIF, we observed that YearRemodAdd, GarageArea, TotalBsmtSF, and GrLivArea still exhibited high VIF values. Consequently, we decided to remove these predictors from the analysis as well.")


st.markdown("> *By identifying and removing all the above predictors, I was able to bring down the VIF values to a normal range of under 10 and eliminate all highly correlated pairs in the data.*")
st.markdown("")


st.image(Image.open('images/heatmap_after.png'), caption='Pearson Correlation heatmap after removing the problematic predictors above. It shows that there are no pairs with a Pearson correlation greater than 0.5')

st.markdown("")
st.image(Image.open('images/vif.PNG'), caption='Calculating VIF values for the predictors after removing the problematic ones shows that VIF values of the remaining predictors are in a normal range.')


st.markdown("> *We can conclude that the assumption of having no multicollinearity is met*. ")


# Normality
st.markdown("")
st.markdown("")
st.markdown('##### Normality of residuals assumption ')
st.markdown("")
st.markdown("""

We use residual analysis to assess and validate assumptions related to the residuals in a regression model.

One of the assumptions related to the residuals is normality, which implies that the residuals should follow a normal distribution with a mean of zero. This is important because many statistical tests and confidence intervals assume normality.


We performed visual inspection for the normality of the residuals using Q-Q plot and density plot. Both plots show a relatively normal distribution of the residuals. 

""")
# Load the image
#image = Image.open('images/normality_assumption_yellow.png')
#img_with_border = ImageOps.expand(image , border=2, fill='grey')

st.image(Image.open('images/normality.png'), caption='The left subplot displays a Q-Q plot of the residuals, while the right subplot presents the kernel density plot of the residuals. Both plots serve the purpose of visually examining the normality of the residuals.')


st.markdown("""

Visually testing for the normality of residuals assumption using :
- **Q-Q plot** : this plot compares the distribution of a sample to a theoretical normal distribution. If the residuals follow a normal distribution, the points in the Q-Q plot will fall on a straight line and any deviation from a straight line indicates a deviation from normality.
- **Density plot** : this plots the kernel density estimate (KDE) of the distribution of a variable. In this context the density plot shows the estimated frequency distribution of the residuals, if the data is skewed, the density plot will display unequal sides, which indicates that the residuals are do not follow a normal distribution.

To confirm this visual inspection, we also run a normality statistical test, namely Shapiro-Wilk test.

""")
st.image(Image.open('images/shapiro_wilk.PNG'), caption='The Shapiro-Wilk test examines the normality of the residuals by evaluating the null hypothesis which states that the residuals follow a normal distribution. If the p-value is below the chosen significance level, then the null hypothesis is rejected.')
st.markdown("")

st.markdown("> *We can conclude that the normality of the residuals terms is met.*")


# Hetroscedasticity
st.markdown("")
st.markdown("")
st.markdown('##### Homoscedasticity of residuals assumption ')
st.markdown("")

st.markdown("""The assumption of Homoscedasticity or the homogeneity of the variance of the residuals  implies that the variance of the residuals σ² is the same across the spectrum of all values of the predictor variables, more precisely the variance should not be a function of the explanatory variables X. Violations of homoscedasticity indicate heteroscedasticity, where the variability of residuals varies systematically with the predictors.""")

#img_with_border = ImageOps.expand(Image.open('hetro_assumption.png') , border=2, fill='grey')
st.image(Image.open('images/hetroscedasticity.png'), caption='')

st.markdown("")

st.markdown("""Visually testing for the hetroscedasticity assumption. We use the following plots to check if the residuals suffer from non-constant variance or not :

- **Residuals plot** : it plot the residuals vs index plot, it's used to plot the spread of the residuals and to inspect for a systematic pattern in the spread that could imply the presence of Homoscedasticity.
- **Scale-location plot** : it plots the square root of the absolute standardized residuals against the fitted values of the model. The horizontal axis of the plot represents the fitted values of the model, while the vertical axis represents the square root of the absolute standardized residuals. By taking the square root of the absolute standardized residuals, we are transforming them to have a roughly constant variance across the range of fitted values.
- A well behaving  Scale-Location plot: has a red line that is roughly horizontal across the plot and there is no clear pattern in the spread of the residual terms.
""")
st.markdown("")

st.markdown("""
To confirm this visual inspection, we also ran the Breusch-Pagan test for hetroscedasticity, and the test concluded that the residual variance does not depend on the variables in x in the form, confirming the homoscedasticity assumption of the residuals.
""")
st.image(Image.open('images/breuschpagan.PNG'), caption='')
st.markdown("")


st.markdown("> *We can conclude that the homoscedasticity assumption is met.*")



# Autocorrelation
st.markdown("")
st.markdown("")
st.markdown('##### No Autocorrelation of residuals assumption ')
st.markdown("")
st.markdown("")

st.markdown("""

Autocorrelation occurs when the residuals of a model are dependent of each other, that is the value of one residual is influenced by the value of other residuals. If there is a departure of this assumption it indicates that there is crucial information that the model has not captured or has missed.""")


st.markdown("""
We used the Durbin-Watson test to inspect for autocorrelation in the residuals of a linear regression model. 
""")
st.markdown("")
st.image(Image.open('images/durbin_watson.PNG'), caption='')
st.markdown("> *We can conclude that the assumption of no autocorrelation of the residuals is met.* ")


# Preprocessing

st.markdown("")
st.markdown("")
st.markdown("### ⚒️ Preprocessing")
st.markdown("")

st.markdown("To avoid data leakage, we began by splitting the data into training and validation sets before proceeding with imputing missing data. We then examined the distribution of missing data across the training, validation, and testing sets using the following plot where we notice that certain features have missing values across all sets, while others only appear in the testing data with significantly fewer missing values.")

st.image(Image.open('images/missing_values.png'), caption="It's important to note that I used a logarithmic scale to represent the number of missing data on the x-axis because it helps in visualizing both small and large values on the same plot. Given the significant difference in the number of missing values between the features, using a log scale allows for the compression of larger values while ensuring that smaller values remain clearly visible. Without the log scale, the features with a smaller number of missing values, located at the bottom of the plot, would not display a bar at all.")


st.markdown("""
For meaningful missing data imputation, we used different strategies such as replacing `LotFrontage` missing values with the median value for each neighborhood since houses in the same neighborhood often have similar lot frontage values. Or for instance, for houses with a `GarageArea` of 0, we inferred the absence of a garage and set the `GarageYrBlt` of these houses to 0 to indicate the absence of a garage.
""")

st.markdown("")
st.markdown("""After handling the missing data, we proceeded to **encode the categorical features**. Categorical variables can be classified into two types, ordinal and nominal and each requires a different encoding scheme:
   
- **Ordinal Data** : are variables with categories that have an inherent order. For this type of data we use **ordinal encoding** where we map the categories to integer numbers specifying order, for this we used `pd.Categorical()` with `ordered` flag set to True to be treated as ordered categorical variables.

- **Nominal variables** : These variables have categories that do not have an inherent order or ranking. For encoding nominal variables, we have two encoding options:
    
    - **One-hot encoding** : This converts a variable with *n* categories into *n* binary variables, however it leads to multicollinearity among various variables.
    
    - **Dummy Encoding** : This approach creates *n-1* binary variables for a categorical variable with *n* categories. The final category serves as the reference and it's represented by 0 in the *n-1* binary variables. """)


st.markdown("""To encode nominal variables, we used dummy encoding  via Pandas `get_dummies()` method with the `drop_first` flag set to True. We chose this approach over one-hot encoding due to its ability to create fewer number of columns in the dataset and avoid the dummy variable trap. With dummy encoding, the presence of one dummy variable can be inferred from the absence of the others, thereby reducing multicollinearity issues.   """)

st.markdown("""We then added two essential preprocessing steps to ensure that the assumptions of multiple linear regression are met. Firstly, we addressed outliers with large residuals by carefully identifying and removing them from the dataset. Secondly, we performed a non-linear transformation on the target variable using the logarithm function. These are  discussed in detail in the section covering the validation of linear assumptions. """)



# Feature selection
st.markdown("")
st.markdown("")
st.markdown("### Feature Selection")
st.markdown("")

st.markdown("""To run a comprehensive and rigorous feature selection process we evaluated the features selected from all feature selection approaches  based on various evaluation metrics such as RMSE, AIC, BIC and Adjusted R² scores by considering every produced set of features as a distinct model,  we implemented all these metrics in [Metrics class](https://gist.github.com/Manalelaidouni/2b9d0125ebe90bd1bc689a8d104b46ea). We then compared the results of each approach and selected the best subset of features that performs well on these metrics while also considering the model's complexity. """)
st.markdown("It's important to note that there are three main approaches to performing feature selection. Here is the difference between them :")
st.image(Image.open('images/FS_methods.png'), caption='Different feature selection methods, created by the author.')

st.markdown("")

st.markdown("""> *As a reminder, AIC and BIC balance the fit and complexity of the model. Lower values of AIC and BIC indicate better model fit with fewer features, the difference between the two is that BIC applies stronger penalty for model complexity compared to AIC. In cases where these scores are negative, a higher negative value for AIC or BIC indicate better performance.*
>
> *Meanwhile RMSE metric measures the prediction error of a model in units of the response variable, a lower RMSE score indicate a better fit. Adjusted-R² is also measure goodness of fit statistic that penalizes R² for each additional predictor by taking into account both the number of parameters and the sample size, a higher adjusted-R² score suggest a stronger fit.* """)

st.markdown("")
st.markdown("")

st.markdown("We used all the above mentioned feature selection methods to generate a set of selected features and create reduced models. Since each reduced model consists of different predictors but with the same response variable, they are considered distinct models. To compare these models and perform model selection, we have used the following evaluation metrics: AIC, BIC, RMSE, and Adjusted R². Let's look at the performance of different subsets of features on validation data :")

st.markdown("")
st.image(Image.open('images/performance.png'), caption='')
st.markdown("")
st.markdown("Let's take a look at the complexity of the selected subsets from all feature selection methods :")
st.markdown("")
st.image(Image.open('images/complexity.png'), caption='')
st.markdown("")

st.markdown("")
st.markdown("""We notice `Gradient Boosting`  achieved the best performance based on AIC and BIC, and the second best performance based on Adjusted R² on the validation set. This indicates that the feature subsets selected by `Gradient Boosting` provide a good fit to the model and demonstrate strong predictive performance. These subsets effectively reduce model complexity on the validation data, resulting in a better trade-off between model complexity and fit, this makes sense since `Gradient Boosting` has the smallest model complexity consisting of only 17 features. """)
st.markdown("")
st.markdown("""Furthermore, since the performance evaluation is conducted on a hold out set, it demonstrates that the model built using this subset of features obtained from the embedded method of `Gradient Boosting` exhibits good generalization to unseen data. On the other hand,  `F-Regression` performed the worst on these metrics because it has the largest model size with 170 features and Information criterion metrics penalize large models with non-relevant features.
  """)
st.markdown("")

st.markdown(""" >**Choosing the final model**. *`Gradient Boosting` performed best according to  *AIC*, *BIC*, while third best on *Adjusted R²* 
 on the validation set, meanwhile `eli5_perm_imp` performed best on *RMSE* and *Adjusted R²*.* 
>
> *However it's important to consider the complexity of the model, the fact that the feature subset selected by `Gradient Boosting` has a lower model complexity with only 17 features compared to `eli5_perm_imp`'s 50 features makes it more practical for user input in the Streamlit web app. Therefore, the set of features obtained through the embedded method of `Gradient Boosting` will be chosen as the final set of features for modeling.* """)
st.markdown("")



# Modeling
st.markdown("")
st.markdown("")
st.markdown("### Modeling")

st.markdown("""I began by evaluating various regression models using the *RMSE* score in a cross validation setup, and the results are shown in the following bar plot : """)
st.markdown("")
st.image(Image.open('images/model_comparison.png'), caption='')
st.markdown("")
st.markdown("""Based on the evaluation, I selected the best-performing models to further optimize. To achieve this, I ran hyperparameter optimization to find the optimal set of hyperparameter values for each model.""")

st.markdown("""> *I chose **Bayesian optimization** over Grid or Random optimization to search for the best hyperparameter combinations  because Bayesian optimization intelligently explores the search space by selecting hyperparameter configurations to evaluate based on previous evaluations and their performances. This approach is more efficient than Grid search, which exhaustively tries all combinations and can be computationally expensive, or Random search which selects configurations at random.*  """)
st.markdown("")

st.markdown("""To ensure that the hyperparameter optimization process does not overfit the data, we used  cross validation setup using `BayesSearchCV`  from the `Scikit-Optimize` library or `skopt` for short.

After many trials, the best performing ensemble which I used for inference in this web app is a stacking ensemble of optimized base models and meta-model that consists of `Ridge`, `Kernel Ridge`, `CatBoost` , `Decision Tree` , `AdaBoost` , `Random Forest`, `XGBoost`,  `LGBM`, `GradientBoosting` regressors as base models and `Linear Regression` as a meta-model.  """)
st.markdown("""  """)


st.markdown("""You can find the implementation of stacking ensemble in Python in this [link](https://gist.github.com/Manalelaidouni/2a95604980a0c7b2ca6eb7d4e73d4fc3).""")


# Model interpretation
st.markdown("")
st.markdown("")
st.markdown("### Interpreting Model Result")
st.markdown("")

st.markdown(""" After modeling, we need to understand which features are most important for prediction, how they affect the output, and the degree to which they affect the prediction. This understanding help us demystify how the features interact together to construct the output. This can be achieved by calculating the shapley value of each feature. These shapley values can be easily computed using shap library (SHapley Additive exPlanations). Shap assesses the relative importance and the impact of each feature on the model’s predictions, below we use  Shap beeswarm plot to visualize this impact :
""" )

st.image(Image.open('images/shap.png'), caption='Shap beeswarm plot.', width=500)
st.markdown("")

st.markdown("""In the above beeswarm plot, each point on the x-axis represents the shap value  points associated with features on the y-axis, and each point represent the shap value of a particular instance in the data. 

The **magnitude of shap value** indicates the relative impact of the associated feature on the predicted outcome, this is reflected visually by the distance of the points from zero, where the further away they are from both sides of zero, the more the influential they are. Accordingly, the features in the beeswarm plot are visually ranked by feature importance in descending order.

This brings us to the **sign of the shap value**, which indicates the direction of the impact, with positive shap values indicating that the feature contributes positively to the model’s prediction and makes it higher, conversely a negative sign means the feature contributes negatively to the prediction and pushes its value to be lower. 

The color scale represents the change of feature values, this helps to understand how does the model prediction changes as the values of the feature change.  For example, the dark blue color indicates lower feature values, while the bright yellow color represents higher values.   

""")

st.markdown("")

st.markdown(""" 
**Insights** :

- `OverallQual` is an ordinal feature that scores from 1 to 10, the plot reveals that houses with a low overall quality scores significantly reduce the predicted house price, this is reflected by the extended dark blue points to the far left of zero. Meanwhile houses that were assigned higher quality score had a notable positive impact on the model’s prediction, this reflected by the bright yellow points that extend to the far right of the x-axis.

- Similarly smaller houses (with smaller `LotArea`) significantly negatively impact the predicted house price, meanwhile the larger the house the more positively it impact house prediction

- Houses with zero capacity of cars (that is houses that do not have garages are colored in dark blue) significantly hurt the price of the house and the larger the garage  the higher the predicted house value.

- For the `YearBuilt` feature, as expected, older houses negatively impact the predicted price for the house, meanwhile  newer houses significantly impact price prediction.

""")
st.markdown("")

st.markdown("""- When looking at the overall condition of the houses `OverallCond`, we can clearly see that the dark blue points extend much further to the left compared to other features. These points correspond to the houses with the worst overall condition and they have the most negative SHAP values. This means that **houses with really bad condition have the biggest impact on lowering the house's price**.*
""")
st.markdown("")

st.markdown("""
- Houses with no fireplaces negatively impact the predicted house price whereas **the more fireplaces the house contains the more positive the impact on the predicted house price**, this is reflected by the yellow line to the right and the dark blue cluster on the left of zero.

""")

st.markdown("")

st.markdown("""
> *Most Houses don’t have an open porch `OpenPorchSF` or wood deck  area `WoodDeckSF` this is why they have a cluster of dark blue points close to zero,  this indicates that **houses that don’t have these facilities negatively impact the predicted price by a small amount, however having these facilities do increase the value of the house, with larger open porch or wooden deck area having a greater positive impact on price**. This is evident from the yellow points extending to the right of the x-axis.*
""")
st.markdown("")
st.markdown("""
- Binary features have only two colors bright yellow that refers to the presence of the feature and dark blue color that indicates its lack of presence. Such as `ExterQual_TA`, `MSZoning_RM`,  `GarageType_Detchd`,  `MSSubClass_30` and `CentralAir_Y`.

- `ExterQual_TA`, `MSZoning_RM`,  `GarageType_Detchd`,  `MSSubClass_30` display the same pattern where the dark blue cluster is to the right of the x-axis and the yellow cluster is to the left, this means that houses that have a detached garage or that are  are located in a Residential Medium Density zone, also houses with a quality of the material on the exterior that is rated as Average or Typical  negatively impact the prediction of the price.

- Meanwhile houses that don’t have a built-in central air conditioning `CentralAir_Y` negatively impact the predicted house price, this is reflected by the points extending to the dark blue line to the left meanwhile houses that do positively impact the price.

""")
st.markdown("")

st.markdown("""
> **We found that certain features negatively influence the house price, notably:**
>
>- When the quality of the material on the exterior of the house is rated as Average or Typical, it might be perceived as unremarkable or bland or not durable and more prone to damage. 
>
>- Additionally, if the house is located in a Residential Medium Density zone.
> 
>- But here's an interesting finding,  having a detached garage from the house negatively impacts the property's value, it might be that house residents do not prefer going outside to reach their car especially during bad weather and may generally prefer the convenience and accessibility of an attached garage.
>
>- Also, as mentioned above, lacking a built-in central air conditioning or a fireplace tends to reduce the value of the house, as well as having a smaller house size.
""")


# Recommendations
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("### Data-Driven Recommendations for Higher House Value ") 
st.markdown("")

st.markdown("> Homeowners looking to increase the value of their homes often spend too much remodeling and don't get the return on investment when selling the house. Based on the insights uncovered above, **here are some valuable suggestions to increase your property’s worth before placing it on the market** : ")
st.markdown("")
st.markdown("""
- Renovate the kitchen to increase its quality, consider remodeling or installing new counter-tops and cabinets, if space permits, consider adding a kitchen island for added functionality.

- Repair any condition flaws like water damage on walls or ceiling, electrical system issues, deteriorating paint or insect infestations to improve the overall condition of the home.

- Renovate the exterior of the house with high-quality materials by opting for more durable and long-lasting materials.

- Also concerning the exterior of the house, constructing a wood deck can add both visual appeal and value as proven above.

- Install central air conditioning to provide cooling throughout the entire house.

- Install a new fireplace.

- Building a bathroom in the basement would also increase the property’s value, it would make it more of a comfortable and versatile living space that could serve as a relaxation den, an office space or a home gym.

""")

# margin-right: 10px;box-shadow: 0 0 0 1px #ffeeea;width:14px; height:14px;
st.markdown("")
st.markdown("")


st.markdown(f"""
    <div class="container my-5">
    
    <hr style="border-top: 1px solid rgba(255,255,255,.5); margin: 20px auto;">
    <p align = "center" style="color:rgb(49, 51, 63); font-family: "Source Sans Pro", sans-serif; text-align:center;"> Made with   ❤️   by Manal EL Aidouni </p>
    
    <p align = "center" style="color: #6c757d;"> <a href="https://manalelaidouni.github.io" style="text-decoration: none;color: inherit;">manalelaidouni.github.io</a></p>
        <section class="mb-4 bg-dark p-4">
            <a class="btn btn-outline-light btn-floating m-1" style="background-color: #202020; color: white; border-radius: 50%;  padding: 9px; margin-right: 10px;margin-left: 290px;box-shadow: 0 0 0 1px #ffeeea;width:14px; height:14px;" href="https://twitter.com/Manal_ELAI" role="button"><i class="fab fa-twitter"></i></a>
            <a class="btn btn-outline-light btn-floating m-1" style="background-color: #202020; color: white; border-radius: 50%;  padding: 9px; margin-right: 10px;box-shadow: 0 0 0 1px #ffeeea;width:14px; height:14px;" href="https://www.linkedin.com/" role="button"><i class="fab fa-linkedin-in"></i></a>
            <a class="btn btn-outline-light btn-floating m-1" style="background-color: #202020; color: white; border-radius: 50%;  padding: 9px; margin-right: 10px;box-shadow: 0 0 0 1px #ffeeea;width:14px; height:14px;" href="https://github.com/Manalelaidouni" role="button"><i class="fab fa-github"></i></a>
        </  section>
    </div>
    """, unsafe_allow_html=True)









