import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

class KNN:
    def __init__(self, dataset):
        # Controlla che il dataset sia un DataFrame
        if isinstance(dataset, pd.DataFrame):
            self.dataset = dataset
        else:
            raise ValueError("Il dataset deve essere un DataFrame di Pandas.")

        self.model = KNeighborsClassifier(n_neighbors=3)

    def preprocess_data(self):
        # Codifica variabili categoriali (ad esempio, PetType, Breed, Color, Size)
        label_encoders = {}
        categorical_columns = ["PetType", "Breed", "Color", "Size"]
        
        for col in categorical_columns:
            le = LabelEncoder()
            self.dataset[col] = le.fit_transform(self.dataset[col])
            label_encoders[col] = le
        
        # Normalizza le variabili numeriche
        scaler = StandardScaler()
        numeric_columns = ["AgeMonths", "WeightKg", "TimeInShelterDays", "AdoptionFee"]
        self.dataset[numeric_columns] = scaler.fit_transform(self.dataset[numeric_columns])
        
        # Seleziona le colonne desiderate per le feature (X)
        X = self.dataset[['PetType', 'Breed', 'AgeMonths', 'WeightKg', 'Size', 'Vaccinated', 'TimeInShelterDays', 'AdoptionFee']]
        
        # Definisci la colonna target (y)
        y = self.dataset['AdoptionLikelihood']  # La probabilità di adozione
        
        return X, y

    def train_model(self):
        X, y = self.preprocess_data()
        
        # Suddividi i dati in training e test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Addestra il modello
        self.model.fit(X_train, y_train)
        
        # Fai previsioni e calcola l'accuratezza
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("Accuracy:", accuracy)

# Carica il tuo dataset (sostituisci il percorso con quello corretto)
dataset = pd.read_csv('C:/Users/Notebook Dell/Desktop/progetto/pet_adoption_data_cleaned.csv')

# Crea un'istanza di KNNAnimalAdoption e addestra il modello
knn = KNN(dataset)
knn.train_model()
