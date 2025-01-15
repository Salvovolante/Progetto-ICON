import pandas as pd
from owlready2 import *

def create_rdf(csv_file_path):
    # Crea l'ontologia
    onto = get_ontology("http://example.org/animal_adoption_ontology")  # Un URI generico, non caricheremo da un URL

    with onto:
        # Classi per l'ontologia
        class Animal(Thing):
            pass

        class Shelter(Thing):
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
    dataset = pd.read_csv(csv_file_path)

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

    # Salva l'ontologia in formato RDF (senza il file .owl)
    onto.save(file="animal_adoption_ontology.rdf", format="rdfxml")
    print("Ontologia salvata con successo come animal_adoption_ontology.rdf")

    return onto

# Percorso del file CSV
csv_file_path = "pet_adoption_data_cleaned.csv"

# Creazione dell'ontologia e salvataggio come RDF
create_rdf(csv_file_path)
