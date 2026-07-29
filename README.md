# Reassessing the Additivity Assumption: Computational Prediction of Hammett Constants in Mono- and Polysubstituted Systems

### Laura-Charlotte Underwood Krogstad and Robert S. Paton

This GitHub repository contains all data and code used in the development of this manuscript. It includes the following:
* conda_environments: contains yaml files for the program versions and dependencies used for the analysis in this work.
* csv_files: contains csvs for the data used/generated for this work.
* notebooks: contains the Jupyter notebooks containing the workflows for data preparation and analysis as well as figure generation; these are meant to be run in the same folder as their csv dependencies (contained in csv_files).
* monosubstituted_structures: xyz files containing the geometries of the DFT-optimized lowest energy conformer for the monosubstituted benzonitriles used in this work; corresponding experimental and predicted Hammett parameters can be found in csv_files/hammett_predictions.csv.
* new_substituent_structures: xyz files containing the geometries of the DFT-optimized lowest energy conformer for the substituted benzonitriles from the DrugBank database; corresponding predicted Hammett parameters can be found in csv_files/new_predictions.csv.
* structures_3_4_5_trisubstituted: xyz files containing the geometries of the DFT-optimized lowest energy conformer for the trisubstituted benzonitriles used in this work; corresponding experimental Hammett sums and predicted Hammett parameters can be found in csv_files/hammett_predictions_3_4_5.csv. For those structures which had experimentally observed polysubstituted Hammett parameters, the experimentally observed parameters can be found in csv_files/experimental_values_3_4_5.csv.
* structures_3_4_disubstituted: xyz files containing the geometries of the DFT-optimized lowest energy conformer for the 3,4-disubstituted benzonitriles used in this work; corresponding experimental Hammett sums and predicted Hammett parameters can be found in csv_files/hammett_predictions_3_4.csv. For those structures which had experimentally observed polysubstituted Hammett parameters, the experimentally observed parameters can be found in csv_files/experimental_values_3_4.csv.
* structures_3_5_disubstituted: xyz files containing the geometries of the DFT-optimized lowest energy conformer for the 3,5-disubstituted benzonitriles used in this work; corresponding experimental Hammett sums and predicted Hammett parameters can be found in csv_files/hammett_predictions_3_5.csv. For those structures which had experimentally observed polysubstituted Hammett parameters, the experimentally observed parameters can be found in csv_files/experimental_values_3_5.csv.
  
