from mordred import Calculator, descriptors
from mordred.error import Missing
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import numpy as np
import pandas as pd
import argparse
import os
import joblib
import json
import warnings
import re

default_temp = 298.15
default_smiles_col = "SMILES"
default_mole_fraction = "Mole_fraction"
default_model_dir = "density_water_ensemble_model"
default_output_csv = "predictions.csv"

def standardize_molecule(smi):
    changes = []
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi, "Invalid SMILES"
    
    try:
        parts = smi.split('.')
        reionized_parts = []
        for part in parts:
            part_mol = Chem.MolFromSmiles(part)
            if part_mol:
                reionized = rdMolStandardize.Reionize(part_mol)
                if Chem.MolToSmiles(reionized) != Chem.MolToSmiles(part_mol):
                    changes.append("Reionized")
                reionized_parts.append(Chem.MolToSmiles(reionized))
            else:
                reionized_parts.append(part)
        reionized_smi = '.'.join(reionized_parts)
        if reionized_smi != smi:
            mol = Chem.MolFromSmiles(reionized_smi)
        else:
            mol = Chem.MolFromSmiles(smi)
    except Exception as e:
        return smi, f"Reionization Error: {e}"

    try:
        normalizer = rdMolStandardize.Normalizer()
        normalized = normalizer.normalize(mol)
        if Chem.MolToSmiles(normalized) != Chem.MolToSmiles(mol):
            changes.append("Functional groups normalized")
        mol = normalized
    except Exception as e:
        return smi, f"Normalization Error: {e}"

    try:
        smi_before_stereo = Chem.MolToSmiles(mol, isomericSmiles=True)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
        smi_after_stereo = Chem.MolToSmiles(mol, isomericSmiles=True)
        if smi_before_stereo != smi_after_stereo:
            changes.append("Stereo removed")
    except Exception as e:
        return smi, f"Stereo Cleanup Error: {e}"

    standardized_smi = Chem.MolToSmiles(mol)
    change_summary = ", ".join(changes) if changes else "No changes"
    return standardized_smi, change_summary

def reorder_charged_species(df, smiles_col='Standardized_IL_SMILES'):
    def reorder_smiles(smi):
        parts = smi.split('.')
        cations = []
        anions = []
        for part in parts:
            if re.search(r'\+[0-9]*', part):
                cations.append(part)
            elif re.search(r'\-[0-9]*', part):
                anions.append(part)
        reordered = cations + anions
        return '.'.join(reordered)
    df[smiles_col] = df[smiles_col].apply(reorder_smiles)
    return df

def load_models_and_metadata(model_dir):
    models = []
    model_metadata = []
    for i in range(1, 6):  # 5 models
        model_path = os.path.join(model_dir, f"model_{i}.joblib")
        metadata_path = os.path.join(model_dir, f"metadata_{i}.json")
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model or metadata file for model {i} not found in {model_dir}.")
        model = joblib.load(model_path)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        models.append(model)
        model_metadata.append(metadata)
    return models, model_metadata

def calculate_descriptors(smiles, required_descriptors):
    smiles_list = smiles.split('.')
    calc = Calculator(descriptors, ignore_3D=True)
    calc.descriptors = [d for d in calc.descriptors if str(d) in required_descriptors]
    descriptor_vectors = []
    descriptor_names = None
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            desc = calc(mol)
            if descriptor_names is None:
                descriptor_names = [str(d) for d in desc.keys()]
            desc_vector = []
            for name, value in desc.items():
                if isinstance(value, Missing):
                    desc_vector.append(np.nan)
                else:
                    desc_vector.append(value)
            descriptor_vectors.append(desc_vector)
        else:
            print(f"Invalid SMILES: {smi}")
    if descriptor_vectors:
        descriptor_vectors = np.array(descriptor_vectors, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
            mean_descriptor_vector = np.nanmean(descriptor_vectors, axis=0)
        return mean_descriptor_vector, descriptor_names
    else:
        print(f"No valid molecules found for SMILES: {smiles}")
        return None, None
    
def process_il_smiles_list(smiles_list, required_descriptors):
    descriptor_vectors = []
    for smiles in smiles_list:
        mean_descriptor_vector, descriptor_names = calculate_descriptors(smiles, required_descriptors)
        if mean_descriptor_vector is not None:
            descriptor_vectors.append(mean_descriptor_vector)
        else:
            descriptor_vectors.append([np.nan] * len(required_descriptors))
    return np.array(descriptor_vectors)

def main(args):
    # Apply defaults if user did not specify
    if args.input_csv is None:
        print(f"Info: No input file specified.")
    if args.smiles_col is None or args.smiles_col.strip() == "":
        args.smiles_col = default_smiles_col
        print(f"Info: No SMILES column specified. Using default: {default_smiles_col}")
    if args.mole_fraction_col is None or args.mole_fraction_col.strip() == "":
        args.mole_fraction_col = default_mole_fraction
        print(f"Info: No mole fraction column specified. Using default: {default_mole_fraction}")
    if args.model_dir is None or args.model_dir.strip() == "":
        args.model_dir = default_model_dir
        print(f"Info: No model directory specified. Using default: {default_model_dir}")
    if args.output_csv is None or args.output_csv.strip() == "":
        args.output_csv = default_output_csv
        print(f"Info: No output file specified. Using default: {default_output_csv}")

    if not os.path.exists(args.input_csv):
        print(f"Error: Input file '{args.input_csv}' does not exist.")
        return
    
    data = pd.read_csv(args.input_csv)
    if args.smiles_col not in data.columns or args.mole_fraction_col not in data.columns:
        print(f"Error: Input file must contain '{args.smiles_col}' and '{args.mole_fraction_col}' columns.")
        return

    standardized_smiles = []
    for smi in data[args.smiles_col]:
        standardized_smi, change_summary = standardize_molecule(smi)
        standardized_smiles.append(standardized_smi)
    
    data["Standardized_IL_SMILES"] = standardized_smiles
    print("Molecules standardized...")
    data = reorder_charged_species(data, smiles_col="Standardized_IL_SMILES")

    data.rename(columns={args.mole_fraction_col: 'Mole_fraction'}, inplace=True)
    if args.temp_col is None or args.temp_col not in data.columns:
        data['Temperature'] = default_temp
        print(f"Info: 'Temperature' column not provided. Using default: {default_temp} K.")
    elif data[args.temp_col].isna().any():
        data['Temperature'] = data[args.temp_col].fillna(default_temp)
        print(f"Info: Missing values in 'Temperature' column. Defaulting missing values to {default_temp} K.")
    else:
        data.rename(columns={args.temp_col: 'Temperature'}, inplace=True)

    try:
        models, model_metadata = load_models_and_metadata(args.model_dir)
    except Exception as e:
        print(f"Error loading models or metadata: {str(e)}")
        return
    
    all_smiles_list = data["Standardized_IL_SMILES"].tolist()
    predictions = []

    for i, (model, metadata) in enumerate(zip(models, model_metadata), 1):
        print(f"Calculating descriptors for Model {i}...")
        try:
            required_descriptors_model = metadata['descriptors']
            descriptor_array = process_il_smiles_list(all_smiles_list, required_descriptors_model)
            temperature_array = data['Temperature'].values.reshape(-1, 1)
            mole_fraction_array = data['Mole_fraction'].values.reshape(-1, 1)
            descriptor_array = np.hstack((descriptor_array, temperature_array, mole_fraction_array))
        except Exception as e:
            print(f"Error calculating descriptors for Model {i}: {str(e)}")
        
        try:
            print(f"Applying Model {i}...")
            preds = model.predict(descriptor_array)
            predictions.append(pd.Series(preds))
        except Exception as e:
            print(f"Error processing Model {i}: {str(e)}")
            predictions.append(pd.Series([np.nan] * len(all_smiles_list)))
    
    data['prediction_mean'] = pd.concat(predictions, axis=1).mean(axis=1)
    data['prediction_std'] = pd.concat(predictions, axis=1).std(axis=1)
    data['prediction_mean'] = data['prediction_mean'].round(4)
    data['prediction_std'] = data['prediction_std'].round(4)
    data.to_csv(args.output_csv, index=False)
    print(f'Predictions saved to {args.output_csv}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply the consensus model to predict density of IL in water.")
    parser.add_argument("--input_csv", required=True, help="[REQUIRED] Path to the input CSV file (expected comma delimiter).")
    parser.add_argument("--smiles_col", help="Name of the SMILES column in the input CSV (default: SMILES).")
    parser.add_argument("--mole_fraction_col", help="Name of the mole fraction column in the input CSV (default: Mole_fraction).")
    parser.add_argument("--temp_col", help="Name of the temperature column in the input CSV (default: Temperature).")
    parser.add_argument("--model_dir", help="Directory containing the ensemble of models and metadata (default: density_water_ensemble_model).")
    parser.add_argument("--output_csv", help="Path to save the output CSV with predictions (default: predictions.csv).")
    args = parser.parse_args()
    main(args)