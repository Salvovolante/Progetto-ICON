import os
import pandas as pd
from owlready2 import *
import dataset as ds

def load_onto():
    file_path = "./archivio/animal_adoption_ontology.rdf"
    if not os.path.exists(file_path):
        print("File non trovato. Creazione di una nuova ontologia...")
        return create_ontology()
    else:
        return get_ontology(file_path).load()
    

def create_ontology():
    # Crea un'ontologia vuota
    onto = get_ontology("")

    with onto:
        # Classi per l'ontologia
        class Animal(Thing):
            pass

        class Adoption(Thing):
            pass

        # Proprietà
        class is_adopted(ObjectProperty):
            domain = [Animal]
            range = [Adoption]

        class pet_type(DataProperty):
            domain = [Animal]
            range = [str]

        class age_months(DataProperty):
            domain = [Animal]
            range = [int]

        class weight_kg(DataProperty):
            domain = [Animal]
            range = [float]

        class adoption_fee(DataProperty):
            domain = [Adoption]
            range = [float]

        class adoption_likelihood(DataProperty):
            domain = [Adoption]
            range = [int]

    # Carica il dataset
    file_path = "C:/Users/Notebook Dell/Desktop/progetto/pet_adoption_data_cleaned.csv"
    dataset = ds.get_dataset(file_path)

    # Controllo colonne richieste
    required_columns = ['PetID', 'PetType', 'AgeMonths', 'WeightKg', 'AdoptionFee', 'AdoptionLikelihood']
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Il file CSV manca delle seguenti colonne richieste: {missing_columns}")

    # Popolamento delle classi
    with onto:
        for _, row in dataset.iterrows():
            # Crea un'istanza di Animal
            animal = Animal(f"Animal_{row['PetID']}")
            animal.pet_type = [row['PetType']]
            animal.age_months = [row['AgeMonths']]
            animal.weight_kg = [row['WeightKg']]

            # Crea un'istanza di Adoption
            adoption = Adoption(f"Adoption_{row['PetID']}")
            adoption.adoption_fee = [row['AdoptionFee']]
            adoption.adoption_likelihood = [row['AdoptionLikelihood']]

            # Collega l'animale all'adozione
            animal.is_adopted = [adoption]

    #per controllare se esiste la directory "archivio"
    os.makedirs("./archivio", exist_ok=True)

    # Salva l'ontologia in formato OWL
    onto.save(file="./archivio/animal_adoption_ontology.rdf", format="rdfxml")
    print("Ontologia salvata con successo come animal_adoption_ontology.rdf")
    return onto


# Query 1: Animali adottati con un costo maggiore a una certa soglia
def animals_high_cost(cost_threshold, csv_file='progetto/pet_adoption_data_cleaned.csv'):
    # Carica il file CSV in un DataFrame
    df = pd.read_csv(csv_file)
    
    # Filtra i dati per il costo inferiore alla soglia (AdoptionFee rappresenta il costo di adozione)
    animals_high_threshold = df[df['AdoptionFee'] >= cost_threshold]
    
    # Visualizza gli animali con un costo inferiore alla soglia
    if animals_high_threshold.empty:
        print(f"Nessun animale trovato con un costo inferiore a {cost_threshold} nel file.")
    else:
        print(f"Animali con un costo inferiore a {cost_threshold}:")
        for _, row in animals_high_threshold.iterrows():
            print(f"ID Animale: {row['PetID']}, Tipo: {row['PetType']}, Razza: {row['Breed']}, Età (mesi): {row['AgeMonths']}, "
                  f"Colore: {row['Color']}, Taglia: {row['Size']}, Peso (kg): {row['WeightKg']}, "
                  f"Vaccinato: {row['Vaccinated']}, Stato di salute: {row['HealthCondition']}, "
                  f"Giorni in rifugio: {row['TimeInShelterDays']}, Costo Adozione: {row['AdoptionFee']}, "
                  f"Proprietario precedente: {row['PreviousOwner']}, Probabilità di adozione: {row['AdoptionLikelihood']}")

# Query 2: Animali adottati con un costo inferiore a una certa soglia
def animals_under_cost(cost_threshold, csv_file='progetto/pet_adoption_data_cleaned.csv'):
    # Carica il file CSV in un DataFrame
    df = pd.read_csv(csv_file)
    
    # Filtra i dati per il costo inferiore alla soglia (AdoptionFee rappresenta il costo di adozione)
    animals_under_threshold = df[df['AdoptionFee'] <= cost_threshold]
    
    # Visualizza gli animali con un costo inferiore alla soglia
    if animals_under_threshold.empty:
        print(f"Nessun animale trovato con un costo inferiore a {cost_threshold} nel file.")
    else:
        print(f"Animali con un costo inferiore a {cost_threshold}:")
        for _, row in animals_under_threshold.iterrows():
            print(f"ID Animale: {row['PetID']}, Tipo: {row['PetType']}, Razza: {row['Breed']}, Età (mesi): {row['AgeMonths']}, "
                  f"Colore: {row['Color']}, Taglia: {row['Size']}, Peso (kg): {row['WeightKg']}, "
                  f"Vaccinato: {row['Vaccinated']}, Stato di salute: {row['HealthCondition']}, "
                  f"Giorni in rifugio: {row['TimeInShelterDays']}, Costo Adozione: {row['AdoptionFee']}, "
                  f"Proprietario precedente: {row['PreviousOwner']}, Probabilità di adozione: {row['AdoptionLikelihood']}")
