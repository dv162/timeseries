import pandas as pd
from scipy.ndimage import label
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def create_direct_data(data, window_size=5, target_size=3):
    i = 1
    while i < window_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i+=1

    i = 0
    while i < target_size:
        data["target_{}".format(i)] = data["co2"].shift(-i-window_size)
        i += 1

    data = data.dropna(axis=0)
    return data

data = pd.read_csv("co2.csv")
data["time"] = pd.to_datetime(data["time"])
data["co2"] = data["co2"].interpolate() # nội suy fill missing values by interpolate

window_size = 5
target_size = 3
data = create_direct_data(data, 5)
x = data.drop(["time"] + ["target_{}".format(i) for i in range(target_size)], axis=1)
y = data[["target_{}".format(i) for i in range(target_size)]]
train_size = 0.8
num_samples = len(x)
x_train = x[:int(num_samples*train_size)]
y_train = y[:int(num_samples*train_size)]
x_test = x[int(num_samples*train_size):]
y_test = y[int(num_samples*train_size):]

regs = [LinearRegression() for _ in range(target_size)]
for i, reg in enumerate(regs):
    reg.fit(x_train, y_train["target_{}".format(i)])

r2= []
mae=[]
mse=[]

for i, reg in enumerate(regs):
    y_predict = reg.predict(x_test)
    r2.append(r2_score(y_test[["target_{}".format(i)]], y_predict))
    mae.append(mean_absolute_error(y_test[["target_{}".format(i)]], y_predict))
    mse.append(mean_squared_error(y_test[["target_{}".format(i)]], y_predict))

print("R2 Score: {}".format(r2))
print("MAE: {}".format(mae))
print("MSE: {}".format(mse))
# fig, ax = plt.subplots()
# ax.plot(data["time"] [:int(num_samples*train_size)],data["co2"][:int(num_samples*train_size)], label="train")
# ax.plot(data["time"] [int(num_samples*train_size):],data["co2"][int(num_samples*train_size):], label="test")
# ax.plot(data["time"] [int(num_samples*train_size):],y_predict, label="predict")
# ax.set_xlabel("Time")
# ax.set_ylabel("Co2 Consumption")
# plt.show()
fig, axs = plt.subplots(target_size, 1, figsize=(12, 4 * target_size), sharex=True)

for i, reg in enumerate(regs):
    y_pred = reg.predict(x_test)
    true = y_test["target_{}".format(i)].values
    time_axis = data["time"][int(num_samples * train_size):]

    axs[i].plot(time_axis, true, label="True target_{}".format(i))
    axs[i].plot(time_axis, y_pred, label="Predicted target_{}".format(i))
    axs[i].set_title("Prediction vs True for target_{}".format(i))
    axs[i].set_ylabel("CO2 Level")
    axs[i].legend()
    axs[i].grid(True)

    # Thêm text hiển thị R2, MAE, MSE
    textstr = '\n'.join((
        f"R² Score: {r2[i]:.4f}",
        f"MAE: {mae[i]:.4f}",
        f"MSE: {mse[i]:.4f}",
    ))
    axs[i].text(0.01, 0.95, textstr, transform=axs[i].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

axs[-1].set_xlabel("Time")
plt.tight_layout()
plt.show()

# time series