import pandas as pd
import train as tv  # Importa il modulo train
import ontologie as ot
import dataset as ds

from Non_supervisionato import UnsupervisedLearning  # Importa la classe per il K-Means
from KNN import KNN  # Importa la classe KNN
from SWM import SVMAnimalAdoption  # Importa la classe SVM
from decision_tree import DecisionTree  # Importa la nuova classe Decision Tree


def chiedi_scelta(options):
    """Chiede all'utente di fare una scelta tra le opzioni disponibili."""
    scelta = None
    while scelta not in options:
        scelta = input(f"Inserisci il numero della tua scelta ({', '.join(options)}): ")
        if scelta not in options:
            print(f"Scelta non valida. Per favore, scegli {', '.join(options)}.")
    return scelta


def main():
    # Percorso del file CSV corretto
    file_path = "C:/Users/Notebook Dell/Desktop/progetto/pet_adoption_data_cleaned.csv"

    # Carica e pulisci i dati relativi alle adozioni
    print("Caricando il dataset...")
    dataset = ds.get_dataset(file_path)
    dataset = ds.refine_dataset(dataset)
    dictionaries = ds.generate_dictionary(dataset)
    dataset = ds.create_data_frame(dataset, dictionaries)

    # Mostra le prime righe del dataset finale per verifica
    print("Dataset caricato e raffinato per le adozioni:")
    print(dataset.head())

    while True:
        print("\nScegli una delle seguenti opzioni:")
        print("1. Fai una predizione (Random Forest)")
        print("2. Fai una predizione (SVM)")
        print("3. Fai una predizione (KNN)")
        print("4. Fai una predizione (Decision Tree)")
        print("5. Apprendimento non supervisionato (K-Means Clustering)")
        print("6. Esegui query SPARQL")
        print("7. Esci")

        scelta = chiedi_scelta(['1', '2', '3', '4', '5', '6', '7'])
        print(f"Hai scelto l'opzione {scelta}")

        if scelta in ['1', '2', '3', '4', '5']:
            # Carica e pre-processa i dati
            try:
                X_train, X_test, y_train, y_test, _ = tv.load_and_preprocess_data(file_path)
            except ValueError as e:
                print(f"Errore durante il caricamento e preprocessamento dei dati: {e}")
                continue

        if scelta == '1':
            # Random Forest
            print("Eseguiamo una predizione con Random Forest...")
            tv.train_random_forest(X_train, X_test, y_train, y_test, dataset)

        elif scelta == '2':
            # SVM
            print("Eseguiamo una predizione con SVM...")
            svm = SVMAnimalAdoption(file_path)
            svm.train_model()

        elif scelta == '3':
            # KNN
            print("Eseguiamo una predizione con KNN...")
            knn = KNN(dataset)
            knn.train_model()

        elif scelta == '4':
            # Decision Tree
            print("Eseguiamo una predizione con Decision Tree...")
            decision_tree = DecisionTree(file_path)
            best_params = decision_tree.optimize_hyperparameters()
            print("I migliori iperparametri trovati:", best_params)
            decision_tree.train_model()
            decision_tree.plot_decision_tree()

        elif scelta == '5':
            # K-Means Clustering
            print("Eseguiamo l'apprendimento non supervisionato (K-Means Clustering)...")
            unsupervised = UnsupervisedLearning(file_path)
            unsupervised.perform_kmeans_clustering()

        elif scelta == '6':
            print("\nQueries disponibili:")
            print("1. Mostra gli animali adottati con un costo maggiore a una certa soglia")
            print("2. Mostra gli animali adottati con un costo minore o uguale a una certa soglia")

            scelta_q = chiedi_scelta(['1', '2'])
            print(f"Hai scelto la query {scelta_q}")

            if scelta_q == '1':
                cost_threshold = float(input("Inserisci la soglia di costo per visualizzare le adozioni: "))
                ot.animals_under_cost(cost_threshold, csv_file='C:/Users/Notebook Dell/Desktop/progetto/pet_adoption_data_cleaned.csv')

            elif scelta_q == '2':
                cost_threshold = float(input("Inserisci la soglia di costo per visualizzare le adozioni: "))
                ot.animals_high_cost(cost_threshold,csv_file='C:/Users/Notebook Dell/Desktop/progetto/pet_adoption_data_cleaned.csv')
               

        elif scelta == '7':
            print("Uscita dal programma. Arrivederci!")
            break

        # Chiedi se l'utente vuole fare un'altra operazione
        exit_choice = input("\nVuoi fare un'altra operazione? (Sì/No): ").strip().lower()
        if exit_choice not in ['sì', 'si', 'yes']:
            break


if __name__ == "__main__":
    main()
