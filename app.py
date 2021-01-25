from flask import Flask, jsonify, request, abort
from flask_jsonpify import jsonpify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import math
import pickle
import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.float_format', '{:.2f}'.format)
# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)
#Reading the data from files
calendar_df = pd.read_csv('calendar.csv')
sales_eval_df = pd.read_csv('sales_train_evaluation.csv')
prices_df = pd.read_csv('sell_prices.csv')
def final_func_1(X):
    #here we are inserting the columns for the days d_1942 to d_1969 as nan for which we need to forecast sales
    for i in range(1942,1970):
        X['d_'+str(i)] = np.nan
        X['d_'+str(i)] = X['d_'+str(i)].astype(np.float16)
    
    #to transform the dataframe into vertical rows as each corresponds to each day sales of an item from a particular store
    X_melt = pd.melt(X, id_vars=['id','item_id','dept_id','cat_id','store_id','state_id'],
                       var_name='d',value_name='sales')
    #creating a single dataframe
    X_melt = X_melt.merge(calendar_df,  on='d', how='left')
    X_melt = X_melt.merge(prices_df, on=['store_id','item_id','wm_yr_wk'], how='left')
    
    #pre processing missing values of prices by transforming with mean price of that id
    X_melt['sell_price'].fillna(X_melt.groupby('id')['sell_price'].transform('mean'),inplace=True)
    
    #creating lag features such that the for a product on current day it gets it's sales upto 3 months prior.
    shifting = 28 #shift period in order to account for 28 days to forecast
    for i in range(9): #num of weeks to shift here 8 weeks we consider
        X_melt['lag_'+str(shifting+(7*i))] = X_melt.groupby('id')['sales'].shift(shifting+(7*i)).astype(np.float16)
    
    #creating constant shift rolling agg features
    for i in [7,14,28,35,60]:
        X_melt['rolling_mean_'+str(i)] =  X_melt.groupby(['id'])['lag_28'].transform(lambda x: x.rolling(i).mean())
        X_melt['rolling_median_'+str(i)] =  X_melt.groupby(['id'])['lag_28'].transform(lambda x: x.rolling(i).median())
        
    #calender features
    X_melt['date'] = pd.to_datetime(X_melt['date'])
    #each day of the month
    X_melt['day_of_month'] = X_melt['date'].dt.day.astype(np.int8)
    #changing year value as 0 for 2011 and 1 for 2012 .... 5 for 2016
    X_melt['year'] = (X_melt['year'] - X_melt['year'].min()).astype(np.int8)
    #week number of a day in a month ex: 29th in January corresponds to 5th week of January
    X_melt['week_no_inmonth'] = X_melt['day_of_month'].apply(lambda x: math.ceil(x/7)).astype(np.int8)
    #checking if the day is weekend or not
    X_melt['is_weekend'] = (X_melt['wday']<=2).astype(np.int8)
    
    #changing the dtype to category for these columns in order to process the columns with label encoding
    cat_cols = ['id','item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2','snap_CA','snap_TX','snap_WI']
    lenc = LabelEncoder()
    for col in cat_cols:
        X_melt[col] = X_melt[col].astype('category')
        #preprocessing the categorical columns into label encoded columns
        X_melt[col] = lenc.fit_transform(X_melt[col].astype(str))
    #splitting the values of 'd' comlumn to take only the day number    
    X_melt['d'] = X_melt['d'].apply(lambda x: x.split('_')[1]).astype(np.int16)
    #final dataframe after pre-processing and feature engineering we are taking last 2 years historical sales
    X_melt = X_melt.loc[pd.to_datetime(X_melt['date'].dt.date) >= '2014-01-02']
    
    X_pre = X_melt.drop(['sales','date','weekday','wm_yr_wk'],axis=1)
    X_pre.reset_index(drop=True,inplace=True)
    n_rows = int(len(X_pre)*0.2)
    X_pre  = X_pre.tail(n_rows)
    N = 18
    X_pre_pred = pd.DataFrame()
    predictions_df = pd.DataFrame()
    best_base_models = ['M'+str(i) for i in range(1,N+1)]
    preds_X_pre  = ['pred_'+str(i) for i in range(int(X_pre.iloc[0]['d']),int(X_pre.iloc[-1]['d'])+1)]
    features_X_pre_pred = ['X_fea_'+str(i) for i in range(1,N+1)]
    #N represents number of base models
    for i in range(N):
        #predicting for all the days of X_pre using trained N base models and using them as features
        preds_X_pre[i] = pd.DataFrame()
        file_name = best_base_models[i]+'.pkl'
        for k in range(int(X_pre.iloc[0]['d']),int(X_pre.iloc[-1]['d'])+1):
            best_base_models[i] = joblib.load(file_name)
            best_base_models[i].n_jobs = -1
            preds_X_pre[i]['d_'+str(k)] = best_base_models[i].predict(X_pre[X_pre['d']==k])
        df1 = pd.melt(preds_X_pre[i],var_name='d',value_name='sales')
        X_pre_pred[features_X_pre_pred[i]] = df1['sales'].values
    best_metaM = joblib.load('best_meta_model.pkl')
    best_metaM.n_jobs = -1
    predictions = best_metaM.predict(X_pre_pred.values)
    #slicing the predictions such that to get each day predictions of all the products of test data
    start = 0
    t = int(X_pre.iloc[0]['d'])
    while start < len(predictions):
        end = start + 1
        predictions_df['d_'+str(t)] = predictions[start:end]
        start = end
        t = t+1
    predictions_df = pd.concat([X['id'],predictions_df],axis=1,sort=False)
    predictions_df_val = predictions_df[['id']]
    #validation predictions from days 1914-1941
    for i in range(28):
        predictions_df_val['F'+str(i+1)] = predictions_df['d_'+str(1914+i)]
    predictions_df_val['id'] =  predictions_df_val['id'].apply(lambda x: x.replace('evaluation','validation'))
    predictions_df_eval = predictions_df_val.copy()
    #evaluation predictions from days 1942-1969
    for i in range(28):
        predictions_df_eval['F'+str(i+1)] = predictions_df['d_'+str(1942+i)]
    predictions_df_eval["id"] = predictions_df_eval["id"].apply(lambda x: x.replace('validation','evaluation'))
    final_predictions = predictions_df_val.append(predictions_df_eval).reset_index(drop=True)
    return final_predictions


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')
    
@app.errorhandler(406)
def resource_not_found(e):
    return jsonify(error = str(e)), 404

def data_collection_errors(df):
    error_list = {}
    
    num_l = []
    for i in range(1,1942):
        num_l.append(df['d_'+str(i)].values.reshape(-1).tolist())
    num_l = [item for sublist in num_l for item in sublist]
    bool_l = [type(x) == str for x in num_l]
    idx  = [i for i, x in enumerate(bool_l) if x]
    if df.shape[1] != 1947:
        error_list['Incorrect num of columns'] = 'Expected a csv file of '+ str(1947) +' columns but got ' + str(df.shape[1]) +' columns'
        
    if idx:
        for id in idx:
            error_list['d_'+str(id+1)] = 'The feature ' + 'd_'+str(id+1) + ' = ' +num_l[id] + ' not an integer, it should be an integer'
    
    if df.state_id.values not in sales_eval_df.state_id.unique():
        error_list['state_id'] = 'Is a categorical feature with categories: '+ str(list(sales_eval_df.state_id.unique())) + ' but entered incorrect value is '+ str(df.state_id.values)
        
    if df.store_id.values not in sales_eval_df.store_id.unique():
        error_list['store_id'] = 'Is a categorical feature with categories: '+ str(list(sales_eval_df.store_id.unique())) + ' but entered incorrect value is '+ str(df.store_id.values)
        
    if df.dept_id.values not in sales_eval_df.dept_id.unique():
        error_list['dept_id'] = 'Is a categorical feature with categories: '+ str(list(sales_eval_df.dept_id.unique())) + ' but entered incorrect value is '+ str(df.dept_id.values)
        
    if df.cat_id.values not in sales_eval_df.cat_id.unique():
        error_list['cat_id'] = 'Is a categorical feature with categories: '+ str(list(sales_eval_df.cat_id.unique())) + ' but entered incorrect value is '+ str(df.cat_id.values)
            
    if df.item_id.values not in sales_eval_df.item_id.unique():
        error_list['item_id'] = 'Entered item_id value is '+ str(df.item_id.values) + ' is incorrect'
    
    if df.id.values not in sales_eval_df.id.unique():
        error_list['id'] = 'Entered id value is '+ str(df.id.values) + ' is incorrect'
    
        
    if  bool(error_list):
        abort(406,description='Please! check, ' + ', '.join(f'{k} - {v}' for k, v in error_list.items()) + ' all these are entered incorrectly update accordingly')
    return jsonify(error_list)

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        my_uploaded_file = request.files['my_uploaded_file'] # get the uploaded file
        # do something with the file
        # and return the result
        df = pd.read_csv(my_uploaded_file)
        data_collection_errors(df)
        forecasted_sales = final_func_1(df)
        forecasted_sales_list = forecasted_sales.iloc[[1]].values.reshape(-1).tolist()[1:]
        forecasted_sales_list = np.around(forecasted_sales_list,2)
        Day_list = ['Day_'+str(i) for i in range(1,29)]
        for i in range(28):
            Day_list[i] = forecasted_sales_list[i]
        return flask.render_template('forecast.html',Day_1=Day_list[0],Day_2=Day_list[1],Day_3=Day_list[2],Day_4=Day_list[3],Day_5=Day_list[4],Day_6=Day_list[5],Day_7=Day_list[6],Day_8=Day_list[7],Day_9=Day_list[8],Day_10=Day_list[9],Day_11=Day_list[10],Day_12=Day_list[11],Day_13=Day_list[12],Day_14=Day_list[13],Day_15=Day_list[14],Day_16=Day_list[15],Day_17=Day_list[16],Day_18=Day_list[17],Day_19=Day_list[18],Day_20=Day_list[19],Day_21=Day_list[20],Day_22=Day_list[21],Day_23=Day_list[22],Day_24=Day_list[23],Day_25=Day_list[24],Day_26=Day_list[25],Day_27=Day_list[26],Day_28=Day_list[27])
        return str(df.id.values)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)
