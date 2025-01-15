import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Dizionario degli iperparametri
DecisionTreeHyperparameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5, 10],
}

class DecisionTree:
    def __init__(self, file_path, hyperparameters=None):
        # Carica il dataset per l'adozione degli animali
        self.dataset = pd.read_csv(file_path)

        if not isinstance(self.dataset, pd.DataFrame):
            raise ValueError("Il dataset deve essere un DataFrame di Pandas.")

        print("Distribuzione delle classi nel dataset:")
        print(self.dataset['PetType'].value_counts())  # Usa 'PetType' come target

        class_counts = self.dataset['PetType'].value_counts()
        classes_to_keep = class_counts[class_counts >= 2].index
        self.dataset = self.dataset[self.dataset['PetType'].isin(classes_to_keep)]

        if hyperparameters is None:
            hyperparameters = {}

        self.model = DecisionTreeClassifier(**hyperparameters)

    def preprocess_data(self):
        # Seleziona le caratteristiche rilevanti e la variabile target
        X = self.dataset[['AgeMonths', 'WeightKg', 'AdoptionFee']]  # Usa le caratteristiche degli animali
        y = self.dataset['PetType']  # Target: tipo di animale
        return X, y

    def optimize_hyperparameters(self):
        X, y = self.preprocess_data()

        # Suddividi i dati in training e test senza stratificazione
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Esegui la ricerca degli iperparametri
        grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), 
                                   param_grid=DecisionTreeHyperparameters, 
                                   cv=3,  
                                   scoring='accuracy',
                                   return_train_score=True)

        grid_search.fit(X_train, y_train)

        results = pd.DataFrame(grid_search.cv_results_)
        print("Tabella degli iperparametri ottimali:")
        print(results[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']])

        # Richiama la funzione per tracciare la tabella
        self.plot_best_params_table(results)

        return grid_search.best_params_

    def plot_best_params_table(self, results):
        # Filtra i migliori parametri
        best_params = results.loc[results['rank_test_score'] == 1].iloc[0]

        # Prepara i dati per la tabella
        params = {
            'Iperparametro': ['criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf'],
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

        plt.title("Migliori Iperparametri del Decision Tree", fontsize=14)
        plt.show()

    def train_model(self):
        X, y = self.preprocess_data()
        # Suddividi i dati in training e test senza stratificazione
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Addestra il modello
        self.model.fit(X_train, y_train)

        # Fai previsioni e calcola l'accuratezza
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

    def plot_decision_tree(self):
        # Assicurati che il modello sia stato addestrato
        if self.model is None:
            raise ValueError("Il modello non è stato ancora addestrato.")
        
        # Verifica che ci siano più di una classe nel target
        _, y = self.preprocess_data()  # Ottieni il target
        if len(y.unique()) <= 1:
            raise ValueError("Il target ha solo una classe, non è possibile visualizzare l'albero decisionale.")

        # Limita la profondità dell'albero per evitare sovraccarico
        max_depth = 3  # Scegli una profondità inferiore per ridurre la complessità
        plt.figure(figsize=(12, 8))  # Imposta una dimensione adeguata della figura

        # Plot dell'albero decisionale
        plot_tree(self.model, 
                 filled=True, 
                 feature_names=['AgeMonths', 'WeightKg', 'AdoptionFee'],  # Nome delle feature
                 class_names=list(self.model.classes_),  # Usa le classi ottenute dal modello
                 rounded=True, 
                 fontsize=10,
                 max_depth=max_depth)  # Limita la profondità dell'albero

        plt.title("Albero Decisionale - Decision Tree")
        plt.show()

# Esempio di utilizzo della classe DecisionTree
if __name__ == "__main__":
    file_path = 'pet_adoption_data_cleaned.csv'  # Cambia il percorso con il tuo dataset
    decision_tree = DecisionTree(file_path)

    # Ottimizzazione degli iperparametri
    print("Ricerca degli iperparametri migliori...")
    best_params = decision_tree.optimize_hyperparameters()

    # Addestramento del modello con i migliori iperparametri
    print("Addestramento del modello con i migliori iperparametri...")
    decision_tree.train_model()

    # Visualizzazione dell'albero decisionale
    print("Visualizzazione dell'albero decisionale...")
    decision_tree.plot_decision_tree()
