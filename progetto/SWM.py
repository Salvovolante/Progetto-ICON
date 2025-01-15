import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

class SVMAnimalAdoption:
    def __init__(self, data_path):
        # Carica il dataset
        self.dataset = pd.read_csv(data_path)
        self.model = SVC(kernel='linear')  # Puoi modificare il kernel a 'rbf', 'poly', ecc.

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
        X = self.dataset[['PetType', 'Breed', 'AgeMonths', 'WeightKg', 'Size', 'Vaccinated', 'HealthCondition', 'TimeInShelterDays', 'AdoptionFee']]
        
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

# Esempio di utilizzo della classe SVMAnimalAdoption
if __name__ == "__main__":
    svm_adoption = SVMAnimalAdoption('C:/Users/Notebook Dell/Desktop/progetto/pet_adoption_data_cleaned.csv')
    svm_adoption.train_model()
