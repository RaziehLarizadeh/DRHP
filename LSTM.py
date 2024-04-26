import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             mean_absolute_percentage_error,
                             r2_score)
from sklearn.utils import shuffle
import pickle

#================================================================
time_steps = 2 # Rolling duration (tau)
scenario = 1 # Scenario number of market demand/recycling rate
data_type = ['demand','recycling'][1] # Selecting the type of data

# Model setup
lstm_units = 512
batch_size = 16
adam = 0.0001
epochs = 70
fine_tune = [True, False][0]
spliter_point = 180
train_test_ration = 0.85

# Figure setup
rcParams['figure.figsize'] = 2, 1.25
dpi = 600
fontsize = 9
fontname = 'Times New Roman'
RANDOM_SEED = 7
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Files and Data
if "outputs" not in os.listdir("./"): os.mkdir("outputs")
if data_type not in os.listdir("outputs"): os.mkdir(f"outputs/{data_type}")
if str(scenario) not in os.listdir(f"outputs/{data_type}"): os.mkdir(f"outputs/{data_type}/{scenario}")
if str(time_steps) not in os.listdir(f"outputs/{data_type}/{scenario}"): os.mkdir(f"outputs/{data_type}/{scenario}/{time_steps}")
res_file = open(f"outputs/{data_type}/{scenario}/{time_steps}/{'Yes' if fine_tune else 'No'}_{time_steps}_{data_type}_S{scenario}.txt",'w')
res_file.write(f"fine_tune: {'Yes' if fine_tune else 'No'}\tlstm_units:{lstm_units}\tspliter_point:{spliter_point}\ttrain_test_ration:{train_test_ration}\tbatch_size:{batch_size}\n")

# funtions
def acc_func(y_true, y_pred, name,file= res_file):
    # Assuming y_true is the true target values and y_pred is the predicted values
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    txt = f'\n{name.upper()}:\n\n'

    txt += f'Mean Squared Error: {mse}\n' \
           f'Mean Absolute Error: {mae}\n' \
           f'Root Mean Squared Error: {rmse}\n'\
           f'Mean Absolute Percentage Error: {mape}\n'
    print('Mean Squared Error:', mse)
    print('Mean Absolute Error:', mae)
    print('Root Mean Squared Error:', rmse)
    print('Mean Absolute Percentage Error:', mape)

    file.write(txt)
    return mse, mae, rmse,mape

def create_dataset(X, y, steps=1):
  Xs, ys, indices = [], [], []
  for i in range(len(X) - 2*steps):
      v = X.iloc[i:(i + steps)].values
      Xs.append(v)
      ys.append(y.iloc[i + steps:i + 2*steps ].values)
      indices.append(i + steps)  # Store the original index

  return np.array(Xs), np.array(ys), np.array(indices)


def create_dataset_horiz(X, y, steps=1):
  Xs, ys, indices = [], [], []
  for i in range(0,len(X) - 2*steps,steps):
      v = X.iloc[i:(i + steps)].values
      Xs.append(v)
      ys.append(y.iloc[i + steps:i + 2*steps ].values)
      indices.append(i + steps)  # Store the original index

  return np.array(Xs), np.array(ys), np.array(indices)

def make_legend(acc_index):
    text_legend = []
    text_legend.append("RMSE = {}".format(np.around(acc_index, decimals=2)))
    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                     lw=0, alpha=0)]
    leg1 = plt.legend(handles, text_legend, loc='best',prop={'family': fontname,'size': fontsize - 3 },
                      # prop={'weight':'bold'},shadow=True,
                      fancybox=True, framealpha=0.3,
                      handlelength=0, handletextpad=0)

    plt.gca().add_artist(leg1)


if __name__=="__main__":

    df = pd.DataFrame(pd.read_excel(f"Synthetic_{data_type}.xlsx"), columns=[f'Scenario {scenario}'])

    df.columns= ['dt']
    dfmin, dfmax = df.min(), df.max()
    df['dt'] = df['dt'].apply(lambda x: (x - df['dt'].min())/(df['dt'].max()-df['dt'].min()))

    # Plot data
    df['tmp'] = range(len(df))
    df.plot.scatter(x='tmp',
                          y='dt',
                          colormap='viridis')
    df.drop(columns = ['tmp'],inplace = True)

    df_org = df.copy()
    df = df.iloc[:spliter_point].copy()
    X_df, y_df, indices  = create_dataset(df, df.dt, time_steps)

    X, y, shuffled_indices = shuffle(X_df, y_df, indices, random_state=42) # shuffle the data before spliting the training and testing sets


    # Split the shuffled dataset into training and testing sets
    train_size , test_size= int(train_test_ration* len(X)), len(X)- int(train_test_ration* len(X))
    X_train, X_test, train_indices, test_indices = X[:train_size], X[train_size:train_size + test_size], shuffled_indices[:train_size], shuffled_indices[train_size:train_size + test_size]
    y_train, y_test = y[:train_size], y[train_size:train_size + test_size]


    # Configure model
    model = keras.Sequential()
    model.add(keras.layers.LSTM(
      units=lstm_units,
      input_shape=(X_train.shape[1], X_train.shape[2])
    ))

    model.add(keras.layers.Dense(units=time_steps))
    model.compile(
        loss=  'mean_squared_error', # 'mean_squared_error',
        optimizer=keras.optimizers.Adam(adam),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )

    # fit the model on training data
    history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    verbose=1,
    shuffle=False,
    callbacks=EarlyStopping(
        monitor='val_rmse',
        patience=10,
        restore_best_weights=True,
    )
    )
    pd.DataFrame(history.history).to_csv(f"outputs/{data_type}/{scenario}/{time_steps}/{'Yes' if fine_tune else 'No'}_{time_steps}_{data_type}_S{scenario}_logs.csv")
    res_file.write(f"\nNumber of run epochs on training: {str(len(history.history['loss']))}\n")

    # Get the training accuracy
    y_pred_train = model.predict(X_train)
    train_res = acc_func(y_train, y_pred_train, name = 'Training')

    # Get the testing accuracy
    y_pred = model.predict(X_test)
    test_res = acc_func(y_test, y_pred, name = 'Testing')

    # predict the Horizon phase data
    actual = []
    pred = []
    df_horiz = df_org.iloc[(spliter_point-time_steps): ].copy()
    X_horiz, y_horiz, indices_horiz  = create_dataset_horiz(df_horiz, df_horiz.dt, time_steps)

    if fine_tune==False:
        with open(f"outputs/{data_type}/{scenario}/{time_steps}/{'Yes' if fine_tune else 'No'}_{time_steps}_{data_type}_S{scenario}.pkl", "wb") as f:
            pickle.dump([model], f)

        y_pred_horiz = model.predict(X_horiz)
    else:
        with open(
                f"outputs/{data_type}/{scenario}/{time_steps}/{'Yes' if fine_tune else 'No'}_before_{time_steps}_{data_type}_S{scenario}.pkl",
                "wb") as f:
            pickle.dump([model], f)

        for i in range(len(X_horiz)):

            single_sample = np.expand_dims(X_horiz[i], axis=0)
            single_label = np.expand_dims(y_horiz[i], axis=0)
            single_sample_pred = model.predict(single_sample)

            model.train_on_batch(single_sample, single_label)
            if i ==0: y_pred_horiz = single_sample_pred.copy()
            else: y_pred_horiz = np.vstack((y_pred_horiz,single_sample_pred))

        with open(
                f"outputs/{data_type}/{scenario}/{time_steps}/{'Yes' if fine_tune else 'No'}_after_{time_steps}_{data_type}_S{scenario}.pkl",
                "wb") as f:
            pickle.dump([model], f)

    for x in y_pred_horiz: pred.extend(x)
    for x in y_horiz: actual.extend(x)
    horiz_res = acc_func(y_horiz, y_pred_horiz, name = 'Horizon')

    plt.close()
    plt.plot( range(spliter_point + len(pred)), df_org['dt'].apply(lambda x: x*(dfmax-dfmin)+dfmin).iloc[:spliter_point + len(pred)].values , linestyle ='--', linewidth= 0.45 , c='teal')

    plt.plot( range(spliter_point, spliter_point+len(pred)),  [x*(dfmax-dfmin)+dfmin for x in  pred ] , linestyle ='solid', linewidth= 0.45, c='orangered') #mediumvioletred

    plt.title(f'Scenario {scenario}\n'
              f'RMSE Train: {np.around(train_res[2],2)} R2: {np.around(train_res[-1],2)}\n'
              f'RMSE Test: {np.around(test_res[2],2)} R2: {np.around(test_res[-1],2)}\n'
              f'RMSE Horizon: {np.around(horiz_res[2],2)} R2: {np.around(horiz_res[-1],2)}')
    plt.xlabel('Days', fontname= fontname,fontsize = fontsize)
    plt.ylabel(f'{data_type[0].upper() + data_type[1:]}{" Rate" if data_type=="recycling" else ""}', fontname= fontname,fontsize = fontsize)
    plt.xticks(fontname= fontname,fontsize = fontsize-2)
    plt.yticks(fontname=fontname, fontsize=fontsize - 2)
    plt.savefig(f"outputs/{data_type}/{scenario}/{time_steps}/{'Yes' if fine_tune else 'No'}_{time_steps}_{data_type}_S{scenario}.jpeg",dpi=dpi, bbox_inches = 'tight')
    plt.title(f'Scenario {scenario}',fontname=fontname, fontsize=fontsize)
    plt.savefig(f"outputs/{data_type}/{scenario}/{time_steps}/{'Yes' if fine_tune else 'No'}_{time_steps}_{data_type}_S{scenario}_notitle.jpeg",dpi=dpi, bbox_inches = 'tight')
    plt.savefig(f"outputs/{data_type}/{scenario}/{time_steps}/{'Yes' if fine_tune else 'No'}_{time_steps}_{data_type}_S{scenario}_notitle.pdf",dpi=dpi, bbox_inches = 'tight')
    make_legend(horiz_res[2])
    plt.savefig(f"outputs/{data_type}/{scenario}/{time_steps}/{'Yes' if fine_tune else 'No'}_{time_steps}_{data_type}_S{scenario}_notitle2.jpeg",dpi=dpi, bbox_inches = 'tight')
    plt.savefig(f"outputs/{data_type}/{scenario}/{time_steps}/{'Yes' if fine_tune else 'No'}_{time_steps}_{data_type}_S{scenario}_notitle2.pdf",dpi=dpi, bbox_inches = 'tight')

    #  Save prediction data
    red_data = [(x*(dfmax-dfmin)+dfmin ).values[0] for x in pred]

    # Create a DataFrame with the red data
    red_df = pd.DataFrame({'Days': range(spliter_point+1, spliter_point + 1 +len(pred)),
                           f'Actual {data_type} Rate':[(x*(dfmax-dfmin)+dfmin).values[0] for x in actual],
                           f'{data_type}{" Rate" if data_type=="recycling" else ""}': [int(np.around(x, 0)) for x in red_data] if data_type == 'demand' else red_data })

    # Define the Excel file path and name
    excel_file_path = f"outputs/{data_type}/{scenario}/{time_steps}/{'Yes' if fine_tune else 'No'}_{time_steps}_{data_type}_S{scenario}.xlsx"
    red_df.to_excel(excel_file_path, index=False)
    print('Red data saved to', excel_file_path)

    res_file.close()
