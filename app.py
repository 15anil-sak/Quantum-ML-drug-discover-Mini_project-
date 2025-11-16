from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import io
import base64
import json
import logging
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Sample dataset path
DATASET_PATH = os.path.join('static', 'data', 'quantum_ml_drug_discovery_dataset.csv')

# Ensure the data directory exists
os.makedirs(os.path.join('static', 'data'), exist_ok=True)

# Create a sample dataset if it doesn't exist
if not os.path.exists(DATASET_PATH):
    logger.info("Creating sample dataset...")
    # Create sample data with SMILES strings and properties
    smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC(=O)NC1=CC=C(C=C1)O',  # Acetaminophen
        'C1=CC=C2C(=C1)C(=O)OC2=O',  # Coumarin
        'C1=CC=C(C=C1)C(=O)O',  # Benzoic acid
        'COC1=CC=C(C=C1)CCN',  # 4-Methoxyphenethylamine
        'C1=CC=C(C=C1)C(=O)NC2=CC=CC=C2',  # Benzanilide
        'CC1=CC=CC=C1NC(=O)C',  # Acetanilide
        'C1=CC=C(C=C1)C(=O)NCCN',  # N-(2-Aminoethyl)benzamide
        'CC(C)(C)C1=CC=C(C=C1)O',  # 4-tert-Butylphenol
        'C1=CC=C(C=C1)C(=O)NN',  # Benzoic hydrazide
        'C1=CC=C(C=C1)C(=O)NO',  # Benzohydroxamic acid
        'C1=CC=C(C=C1)C(=O)OC2=CC=CC=C2',  # Phenyl benzoate
        'C1=CC=C(C=C1)C(=O)OCC2=CC=CC=C2',  # Benzyl benzoate
        'C1=CC=C(C=C1)C(=O)C2=CC=CC=C2',  # Benzophenone
        'C1=CC=C(C=C1)C(=O)CCC2=CC=CC=C2',  # 1,3-Diphenylpropan-1-one
        'C1=CC=C(C=C1)C(=O)C=CC2=CC=CC=C2',  # Chalcone
        'C1=CC=C(C=C1)C(=O)C(=O)C2=CC=CC=C2',  # Benzil
        'C1=CC=C(C=C1)C(=O)OC(=O)C2=CC=CC=C2'   # Benzoic anhydride
    ]
    
    # Generate random properties
    np.random.seed(42)
    molecular_weight = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in smiles]
    logp = [Descriptors.MolLogP(Chem.MolFromSmiles(s)) for s in smiles]
    tpsa = [Descriptors.TPSA(Chem.MolFromSmiles(s)) for s in smiles]
    h_donors = [Descriptors.NumHDonors(Chem.MolFromSmiles(s)) for s in smiles]
    h_acceptors = [Descriptors.NumHAcceptors(Chem.MolFromSmiles(s)) for s in smiles]
    rotatable_bonds = [Descriptors.NumRotatableBonds(Chem.MolFromSmiles(s)) for s in smiles]
    
    # Add some quantum properties (simulated)
    quantum_property1 = np.random.normal(0.5, 0.2, len(smiles))
    quantum_property2 = np.random.normal(0.7, 0.15, len(smiles))
    binding_affinity = np.random.normal(-8.5, 2.0, len(smiles))
    
    # Create DataFrame
    df = pd.DataFrame({
        'SMILES': smiles,
        'MolecularWeight': molecular_weight,
        'LogP': logp,
        'TPSA': tpsa,
        'HBondDonors': h_donors,
        'HBondAcceptors': h_acceptors,
        'RotatableBonds': rotatable_bonds,
        'QuantumProperty1': quantum_property1,
        'QuantumProperty2': quantum_property2,
        'BindingAffinity': binding_affinity
    })
    
    # Save to CSV
    df.to_csv(DATASET_PATH, index=False)
    logger.info(f"Sample dataset created at {DATASET_PATH}")

# Load the dataset
def load_dataset():
    try:
        return pd.read_csv(DATASET_PATH)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        # Return empty DataFrame if file doesn't exist
        return pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    try:
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 10))
        search = request.args.get('search', '')
        
        df = load_dataset()
        
        # Apply search filter if provided
        if search:
            df = df[df['SMILES'].str.contains(search, case=False)]
        
        # Calculate total pages
        total_records = len(df)
        total_pages = (total_records + page_size - 1) // page_size
        
        # Paginate
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_records)
        
        # Get page data
        page_data = df.iloc[start_idx:end_idx].to_dict('records')
        
        return jsonify({
            'data': page_data,
            'pagination': {
                'page': page,
                'pageSize': page_size,
                'totalRecords': total_records,
                'totalPages': total_pages
            }
        })
    except Exception as e:
        logger.error(f"Error in get_dataset: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/molecule/image', methods=['GET'])
def get_molecule_image():
    try:
        smiles = request.args.get('smiles', '')
        if not smiles:
            return jsonify({'error': 'SMILES parameter is required'}), 400
        
        # Generate molecule from SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return jsonify({'error': 'Invalid SMILES string'}), 400
        
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        
        # Draw molecule to image
        img = Draw.MolToImage(mol, size=(300, 300))
        
        # Convert image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({'image': f'data:image/png;base64,{img_str}'})
    except Exception as e:
        logger.error(f"Error generating molecule image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    try:
        df = load_dataset()
        
        if df.empty:
            return jsonify({'error': 'Dataset is empty'}), 404
        
        # Calculate basic statistics
        stats = {
            'totalMolecules': len(df),
            'avgMolecularWeight': df['MolecularWeight'].mean(),
            'avgLogP': df['LogP'].mean(),
            'avgTPSA': df['TPSA'].mean(),
            'propertyRanges': {
                'MolecularWeight': {
                    'min': df['MolecularWeight'].min(),
                    'max': df['MolecularWeight'].max()
                },
                'LogP': {
                    'min': df['LogP'].min(),
                    'max': df['LogP'].max()
                },
                'TPSA': {
                    'min': df['TPSA'].min(),
                    'max': df['TPSA'].max()
                },
                'BindingAffinity': {
                    'min': df['BindingAffinity'].min(),
                    'max': df['BindingAffinity'].max()
                }
            },
            'distributions': {
                'MolecularWeight': df['MolecularWeight'].tolist(),
                'LogP': df['LogP'].tolist(),
                'TPSA': df['TPSA'].tolist(),
                'BindingAffinity': df['BindingAffinity'].tolist()
            }
        }
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/start', methods=['POST'])
def start_training():
    try:
        data = request.json
        logger.info(f"Starting training with parameters: {data}")
        
        # Simulate training process
        # In a real application, this would be a background task
        
        # Return a training ID
        training_id = datetime.now().strftime('%Y%m%d%H%M%S')
        
        return jsonify({
            'trainingId': training_id,
            'status': 'started',
            'message': 'Training started successfully'
        })
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    try:
        training_id = request.args.get('trainingId', '')
        
        if not training_id:
            return jsonify({'error': 'Training ID is required'}), 400
        
        # Simulate training progress
        # In a real application, this would check the actual training status
        
        # Generate random progress between 0 and 100
        progress = min(100, int(training_id[-2:]) % 100)
        
        # Generate fake loss history
        epochs = max(1, progress // 10)
        loss_history = [1.0 - 0.8 * (i / epochs) + 0.1 * np.random.random() for i in range(epochs)]
        
        return jsonify({
            'trainingId': training_id,
            'progress': progress,
            'status': 'completed' if progress == 100 else 'in_progress',
            'metrics': {
                'loss': loss_history[-1] if loss_history else 1.0,
                'accuracy': min(0.99, 0.5 + progress / 200),
                'epochs': epochs
            },
            'history': {
                'loss': loss_history,
                'epochs': list(range(1, epochs + 1))
            }
        })
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/prediction', methods=['POST'])
def make_prediction():
    try:
        data = request.json
        smiles = data.get('smiles', '')
        model_id = data.get('modelId', '')
        
        if not smiles:
            return jsonify({'error': 'SMILES parameter is required'}), 400
        
        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return jsonify({'error': 'Invalid SMILES string'}), 400
        
        # Calculate molecular properties
        mol_weight = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        
        # Simulate quantum properties and prediction
        # In a real application, this would use the actual trained model
        np.random.seed(int(hash(smiles) % 2**32))
        quantum_property1 = np.random.normal(0.5, 0.2)
        quantum_property2 = np.random.normal(0.7, 0.15)
        binding_affinity = np.random.normal(-8.5, 2.0)
        
        # Generate confidence score
        confidence = np.random.uniform(0.7, 0.95)
        
        return jsonify({
            'smiles': smiles,
            'properties': {
                'MolecularWeight': mol_weight,
                'LogP': logp,
                'TPSA': tpsa,
                'HBondDonors': h_donors,
                'HBondAcceptors': h_acceptors,
                'RotatableBonds': rotatable_bonds
            },
            'quantumProperties': {
                'QuantumProperty1': quantum_property1,
                'QuantumProperty2': quantum_property2
            },
            'prediction': {
                'BindingAffinity': binding_affinity,
                'confidence': confidence
            }
        })
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        # Simulate list of trained models
        # In a real application, this would retrieve actual models
        models = [
            {
                'id': 'model_20230601',
                'name': 'Quantum ML Model v1',
                'created': '2023-06-01',
                'accuracy': 0.89,
                'description': 'Basic quantum ML model for binding affinity prediction'
            },
            {
                'id': 'model_20230715',
                'name': 'Quantum ML Model v2',
                'created': '2023-07-15',
                'accuracy': 0.92,
                'description': 'Improved quantum ML model with better feature extraction'
            },
            {
                'id': 'model_20230901',
                'name': 'Hybrid Quantum-Classical Model',
                'created': '2023-09-01',
                'accuracy': 0.94,
                'description': 'Hybrid model combining quantum and classical ML techniques'
            }
        ]
        
        return jsonify(models)
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
