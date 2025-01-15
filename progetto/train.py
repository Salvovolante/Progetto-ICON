import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

# Dizionario degli iperparametri per RandomForest
RandomForestHyperparameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
}

def load_and_preprocess_data(file_path):
    # Carica il dataset
    df = pd.read_csv(file_path)

    # Verifica se la colonna 'AdoptionLikelihood' esiste nel dataset
    if 'AdoptionLikelihood' not in df.columns:
        raise ValueError("La colonna 'AdoptionLikelihood' non è presente nel dataset. Verifica il file CSV.")
    
    # Usa 'AdoptionLikelihood' come target per la predizione
    target_column = 'AdoptionLikelihood'  # Cambia con la colonna corretta
    
    # Definisci le caratteristiche (features) e il target
    X = df.drop(columns=[target_column])  # Tutte le colonne tranne il target
    y = df[target_column]  # La colonna target

    # Preprocessing per le variabili categoriche
    label_encoder = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = label_encoder.fit_transform(X[col])

    # Dividi i dati in training e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, df


def optimize_random_forest_hyperparameters(X_train, y_train):
    # Esegui la ricerca degli iperparametri per Random Forest
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                               param_grid=RandomForestHyperparameters,
                               cv=3,
                               scoring='accuracy',
                               return_train_score=True)
    
    grid_search.fit(X_train, y_train)
    
    # Crea un DataFrame con i risultati della GridSearchCV
    results = pd.DataFrame(grid_search.cv_results_)
    print("Tabella degli iperparametri ottimali per Random Forest:")
    print(results[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']])

    # Richiama la funzione per tracciare la tabella
    plot_best_params_table(results)

    return grid_search.best_params_

def plot_best_params_table(results):
    # Filtra i migliori parametri
    best_params = results.loc[results['rank_test_score'] == 1].iloc[0]

    # Prepara i dati per la tabella
    params = {
        'Iperparametro': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'],
        'Valore': [best_params['params'][key] for key in best_params['params']]
    }

    params_df = pd.DataFrame(params)

    # Crea una figura per la tabella
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('tight')
    ax.axis('off')

    # Crea la tabella
    table = ax.table(cellText=params_df.values,
                     colLabels=params_df.columns,
                     cellLoc='center',
                     loc='center')

    # Formatta la tabella
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.title("Migliori Iperparametri del Random Forest", fontsize=14)
    plt.show()


def train_random_forest(X_train, X_test, y_train, y_test, df):
    # Step 1: Ottimizzazione degli iperparametri
    print("Ricerca degli iperparametri migliori per Random Forest...")
    best_params = optimize_random_forest_hyperparameters(X_train, y_train)

    # Step 2: Addestra il modello Random Forest con i migliori iperparametri
    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    
    # Predizione
    y_pred = model.predict(X_test)
    print(f"Accuratezza del modello Random Forest: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    
    # Calcolo del rischio: Probabilità predetta per la classe "0" (non adottato)
    if hasattr(model, "predict_proba"):
        risk_scores = model.predict_proba(X_test)[:, 0]  # Probabilità della classe 0
        X_test_with_risk = X_test.copy()
        X_test_with_risk['Rischio'] = risk_scores
        
        # Diagramma a barre: distribuzione del rischio
        plt.figure(figsize=(10, 6))
        sns.histplot(risk_scores, bins=20, kde=True, color='blue')
        plt.title('Distribuzione del rischio predetto')
        plt.xlabel('Rischio')
        plt.ylabel('Frequenza')
        plt.show()

        # Aggiungi 'TimeInShelterDays' al dataset di test
        if 'TimeInShelterDays' in df.columns:
            X_test_with_risk['TimeInShelterDays'] = df.loc[X_test.index, 'TimeInShelterDays']
            
            # Calcola il rischio medio per giorni nel rifugio
            avg_risk_per_time = X_test_with_risk.groupby('TimeInShelterDays')['Rischio'].mean().reset_index()
            
            # Diagramma a barre: rischio medio per il tempo nel rifugio
            plt.figure(figsize=(10, 6))
            sns.barplot(data=avg_risk_per_time, x='TimeInShelterDays', y='Rischio', palette='viridis')
            plt.title('Rischio medio per giorni nel rifugio')
            plt.xlabel('Giorni nel rifugio')
            plt.ylabel('Rischio medio')
            plt.xticks(rotation=45)
            plt.show()
    
    return model

def train_svm(X_train, X_test, y_train, y_test):
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuratezza del modello SVM: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    return model


def train_knn(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuratezza del modello KNN: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    return model


def train_decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuratezza del modello Decision Tree: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    return model
