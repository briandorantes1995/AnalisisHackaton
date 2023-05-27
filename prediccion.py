# Tratamiento de datos
import pandas as pd

import matplotlib.pyplot as plt

# Modelado y Forecasting
from sklearn.ensemble import RandomForestRegressor

from skforecast.ForecasterAutoreg import ForecasterAutoreg

plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 10

datos = pd.read_csv("./Datos SINAICA - San Nicolás - PM2.5 - 2023-04-01 - 2023-05-01.csv", sep=',')
datos[['HoraInicial', 'HoraFinal']] = datos['Hora'].str.split(' ', n=1, expand=True)
datos = datos[['Fecha', 'Parámetro', 'Valor', 'Unidad']]
datos['Fecha'] = pd.to_datetime(datos['Fecha'], format='%Y-%m-%d')
datos = datos.groupby(['Fecha', 'Unidad', 'Parámetro'], as_index=False).mean()
datos = datos.set_index('Fecha')
datos = datos.asfreq('D')
datos = datos.sort_index()
print(" \nInformacion Procesada \n \n")
print(datos)

steps = 15
datos_train = datos[:-steps]
datos_test = datos[-steps:]

print(f" \nFechas train : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)}) \n")
print(f"Fechas test  : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)}) \n \n")
fig, ax = plt.subplots(figsize=(7, 2.5))
datos_train['Valor'].plot(ax=ax, label='train')
datos_test['Valor'].plot(ax=ax, label='test')
ax.legend()

forecaster = ForecasterAutoreg(
                regressor=RandomForestRegressor(random_state=123),
                lags=6
             )

forecaster.fit(y=datos_train['Valor'])

#Modificar steps para obtener mas predicciones
steps2 = 50
predicciones = forecaster.predict(steps=steps2)
print(predicciones)
