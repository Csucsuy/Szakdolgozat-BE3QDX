# Tőzsdei előrejelzés mesterséges intelligencián alapuló rendszerrel

Ez a repozitórium a Dunaújvárosi Egyetemen készült mérnökinformatikus (BSc) szakdolgozatom forráskódját tartalmazza. A projekt egy automatizált, adatvezérelt szimulációs keretrendszer, amely az amerikai részvénypiacot reprezentáló S&P 500 határidős index adatain hasonlítja össze a klasszikus statisztikai (ARIMA) és a modern gépi tanulási (XGBoost) modellek prediktív teljesítményét.

## A projekt célja
A kutatás elsődleges fókuszában a Hatékony Piacok Elméletének (EMH) és a Véletlen bolyongás (Random Walk) hipotézisének empirikus, szoftveres vizsgálata áll. A projekt arra keresi a választ, hogy a technikai indikátorok és viselkedési pénzügyi változók (naptári anomáliák) felhasználásával a fejlett gépi tanulás képes-e stabil információs előnyt (prediktív élt) kovácsolni. 

Ezen felül a projekt gyakorlati megoldást nyújt a döntési fákon alapuló algoritmusok új történelmi csúcsokon tapasztalható extrapolációs problémájára.

## Főbb funkciók és technológiák
* **Nagy teljesítményű Data Pipeline:** A több százezer soros, 5 perces felbontású historikus tick-adatok napi szintre történő memóriahatékony aggregálása a Polars keretrendszer lusta kiértékelésével (lazy evaluation).
* **Feature Engineering:** Trendkövető (SMA_20, EMA_20) és momentum (RSI_14) indikátorok generálása, kiegészítve a holidays csomagra épülő naptári anomália-változókkal.
* **Idősoros prediktív modellezés:** XGBoost regresszor (finomhangolt hiperparaméterekkel) és ARIMA(5,1,0) (Statsmodels) modell iteratív versenyeztetése.
* **Szigorú validáció:** 1-lépéses gördülő előrejelzés (1-step ahead rolling forecast) out-of-sample adatokon, adatszivárgás (data leakage) teljes kizárásával.

## Projekt struktúra
* data_loader.py: Az ETL folyamatért felelős modul. Elvégzi a magas frekvenciájú adatok betöltését, a napi aggregációt, a technikai indikátorok generálását, valamint a tanító/teszt halmazok kronológikus szétválasztását.
* models.py: A fő futtató modul. Elvégzi a célváltozók loghozam-transzformációját, betanítja a modelleket, kiszámítja az értékelési metrikákat (RMSE, Hit Rate), és legenerálja a vizuális összehasonlító grafikonokat.
* data/: (A repozitóriumban nem szerepel) Ide kell elhelyezni az 5 perces felbontású S&P 500 CSV adatbázist (ES_5Years_8_11_2024.csv).

## Telepítés és futtatás

A projekt futtatásához Python 3.9 (vagy újabb) verzió ajánlott.

**1. A repozitórium klónozása:**
```bash
git clone https://github.com/Csucsuy/Szakdolgozat-BE3QDX.git
cd Szakdolgozat-BE3QDX
```

**2. Virtuális környezet létrehozása és aktiválása (ajánlott):**
```bash
python -m venv venv
# Windows esetén:
venv\Scripts\activate
# Linux/macOS esetén:
source venv/bin/activate
```

**3. Függőségek telepítése:**
```bash
pip install -r requirements.txt
```
*(Szükséges főbb csomagok: polars, pandas, xgboost, scikit-learn, statsmodels, matplotlib, holidays)*

**4. A szimuláció elindítása:**
```bash
python models.py
```
A futtatás eredményeként a konzolon megjelennek a modellek metrikái (Hit Rate, RMSE, Feature Importance), a gyökérkönyvtárban pedig létrejön az 'eredmenyek_grafikon.png' fájl.

## Főbb kutatási eredmények
* **Extrapolációs korlát:** A nominális árak predikciójakor az XGBoost új történelmi csúcsokon horizontális platóba ütközött. A logaritmikus hozamtranszformáció ezt a strukturális hibát sikeresen feloldotta, így a modell RMSE-je tizedespontossággal összezárt az ARIMA modellével (48.58 pont vs. 48.55 pont).
* **Piaci hatékonyság (EMH):** Az extrapolációs korlátok elhárítása után mindkét modell iránytalálati aránya (Hit Rate) ~50-54% körül maradt. Ez empirikusan igazolja, hogy az S&P 500 memóriamentes, véletlen bolyongást (Random Walk) követ, így tisztán historikus adatokból az irány tartós előrejelzése statisztikai korlátokba ütközik.

## Szerző
**Grasics Patrik**
Mérnökinformatikus (BSc) hallgató  
Dunaújvárosi Egyetem
