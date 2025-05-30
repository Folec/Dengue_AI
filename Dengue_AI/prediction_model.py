import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

class linear_regression:    
    def __init__(self) :
        self.model = None
        self.X_final = None
        self.feature_names = None

    def create_X_final():
        np.random.seed(42)
        n_samples = 500

        model = LinearRegression()

        villes = ['Bruxelles', 'Gand', 'Anvers', 'Liège', 'Namur', 'Louvain']
        heures = np.random.randint(0, 24, n_samples)
        jours = np.random.choice(['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'], n_samples)
        villes_sample = np.random.choice(villes, n_samples)
        temperature = np.random.uniform(15, 30, n_samples)
        humidite = np.random.uniform(30, 90, n_samples)
        precipitations = np.random.uniform(0, 10, n_samples)
        vent = np.random.uniform(0, 20, n_samples)
        pression = np.random.uniform(980, 1050, n_samples)
        ensoleillement = np.random.uniform(0, 12, n_samples)
        visibilite = np.random.uniform(1, 10, n_samples)
        pollution = np.random.uniform(0, 100, n_samples)
        nebulosite = np.random.uniform(0, 100, n_samples)
        neige = np.random.uniform(0, 50, n_samples)
        orage = np.random.choice([0, 1], n_samples)

        valeur = (
            heures * 2.5 +
            np.array([villes.index(v) for v in villes_sample]) * 5 +
            np.random.normal(0, 10, n_samples)
        )

        df = pd.DataFrame({
            'ville': villes_sample,
            'heure': heures,
            'jour': jours,
            'temperature': temperature,
            'humidite': humidite,
            'precipitations': precipitations,
            'vent': vent,
            'pression': pression,
            'ensoleillement': ensoleillement,
            'visibilite': visibilite,
            'pollution': pollution,
            'nebulosite': nebulosite,
            'neige': neige,
            'orage': orage,
            'valeur': valeur
        })

        X = df[['ville', 'heure', 'jour', 'temperature', 'humidite', 'precipitations', 'vent', 'pression',
                'ensoleillement', 'visibilite', 'pollution', 'nebulosite', 'neige', 'orage']]
        y = df['valeur']

        encoder = OneHotEncoder(sparse_output=False)
        X_encoded = encoder.fit_transform(X[['ville', 'jour']])
        X_final = np.concatenate([X[['heure']].values, X_encoded], axis=1)

        # Generate feature names for encoded data
        categorical_feature_names = encoder.get_feature_names_out(['ville', 'jour'])
        feature_names = ['heure'] + list(categorical_feature_names)

        model = LinearRegression()
        model.fit(X_final, y)

        # 4. Prédiction
        df['prediction'] = model.predict(X_final)

        return X_final, feature_names, model, df