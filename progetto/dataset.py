import pandas as pd
import os

def get_dataset(file_path):
    """Carica il dataset dal file CSV e prepara i dati per l'elaborazione."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Il file {file_path} non esiste nella directory.")
    
    print(f"Caricamento dataset da {file_path}...")
    dataset = pd.read_csv(file_path)  # Carica il dataset dal percorso fornito
    print("Dataset caricato con successo!")

    # Rimozione dei record che contengono valori nulli in colonne critiche
    dataset = dataset.dropna(subset=['PetType', 'Breed', 'AgeMonths', 'Size', 'WeightKg'])
    print(f"Righe dopo la rimozione dei nulli: {dataset.shape[0]}")

    # Uniformiamo le maiuscole per la colonna 'PetType' (se necessario)
    dataset['PetType'] = dataset['PetType'].str.title()
    print(f"PetType dopo la normalizzazione: {dataset['PetType'].unique()}")

    return dataset

def refine_dataset(dataset):
    """Raffina il dataset mantenendo solo le colonne necessarie e rimuovendo i duplicati."""
    print("Raffinamento del dataset...")
    # Definizione delle colonne che si desidera mantenere nel dataset
    columns_to_keep = [
        'PetID', 'PetType', 'Breed', 'AgeMonths', 'Color', 'Size', 'WeightKg', 
        'Vaccinated', 'HealthCondition', 'TimeInShelterDays', 'AdoptionFee', 
        'PreviousOwner', 'AdoptionLikelihood'
    ]

    # Verifica se tutte le colonne esistono nel dataset
    missing_cols = [col for col in columns_to_keep if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Colonne mancanti nel dataset: {', '.join(missing_cols)}")

    dataset = dataset[columns_to_keep]
    print(f"Righe dopo il raffinamento: {dataset.shape[0]}")

    # Rimozione righe duplicate
    dataset = dataset.drop_duplicates()
    print(f"Righe dopo la rimozione dei duplicati: {dataset.shape[0]}")

    return dataset

def generate_dictionary(dataset):
    """Genera un dizionario per mappare i valori univoci della colonna 'PetType' in valori numerici."""
    print("Generazione del dizionario per 'PetType'...")
    pet_types = dataset['PetType'].unique()
    dic_pet_types = {pet_type: idx for idx, pet_type in enumerate(pet_types)}
    print(f"Dizionario generato per 'PetType': {dic_pet_types}")
    return dic_pet_types

def create_data_frame(dataset, dictionaries):
    """Crea un dataframe mappando i valori delle colonne in base ai dizionari."""
    print("Creazione del dataframe con i valori mappati...")
    # Mappatura dei tipi di animali con i valori interi
    dataset["PetType"] = dataset["PetType"].map(dictionaries)

    # Controllo dei valori non mappati
    unmapped_values = dataset.isnull().sum()
    if unmapped_values.any():
        print(f"Attenzione: ci sono valori non mappati in alcune colonne:\n{unmapped_values}")
    
    print("Dataframe creato con successo.")
    return dataset
