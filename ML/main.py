import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib  # For saving scalers

class RTRWHDataset(Dataset):
    """Custom Dataset for RTRWH and AR assessment data"""
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class RTRWHModel(nn.Module):
    """Neural network model for RTRWH and AR assessment"""
    def __init__(self, input_dim):
        super(RTRWHModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4 outputs: recharge potential, suitability score, ratio, cost
        )
    
    def forward(self, x):
        return self.layers(x)

class RTRWHPredictor:
    """Complete pipeline for RTRWH prediction"""
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.target_scaler = StandardScaler()
        self.categorical_cols = ['roof_material', 'soil_type']
        self.numerical_cols = [
            'rooftop_area', 'dwellers', 'annual_rainfall', 
            'aquifer_depth', 'slope', 'drainage_density', 'runoff_coefficient'
        ]
        
    def prepare_data(self, df):
        """Prepare the dataset for training"""
        # Separate features and targets
        X = df.drop(['recharge_potential', 'suitability_score', 
                    'harvest_demand_ratio', 'cost_estimation'], axis=1)
        y = df[['recharge_potential', 'suitability_score', 
               'harvest_demand_ratio', 'cost_estimation']]
        
        y_scaled = self.target_scaler.fit_transform(y)
        
        # Encode categorical features
        X_categorical = self.encoder.fit_transform(X[self.categorical_cols]).toarray()
        
        # Scale numerical features
        X_numerical = self.scaler.fit_transform(X[self.numerical_cols])
        
        # Combine features
        X_processed = np.concatenate([X_numerical, X_categorical], axis=1)
        
        return X_processed, y_scaled
    
    def train(self, df, epochs=100, batch_size=32, learning_rate=0.001):
        """Train the model"""
        self.target_scaler = StandardScaler()
        X, y = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create datasets and dataloaders
        train_dataset = RTRWHDataset(X_train, y_train)
        test_dataset = RTRWHDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = RTRWHModel(X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()
            
            # Record losses
            train_losses.append(train_loss / len(train_loader))
            test_losses.append(test_loss / len(test_loader))
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')
        
        return train_losses, test_losses
    
    def predict(self, input_data):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Process input data correctly
        numerical_data = np.array([[input_data[col] for col in self.numerical_cols]])
        numerical_data = self.scaler.transform(numerical_data)
        
        categorical_data = self.encoder.transform([[input_data[col] for col in self.categorical_cols]]).toarray()
        processed_input = np.concatenate([numerical_data, categorical_data], axis=1)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(processed_input)
            prediction = self.model(input_tensor)
            prediction = self.target_scaler.inverse_transform(prediction.numpy())
        
        return {
            'recharge_potential': prediction[0][0],
            'suitability_score': prediction[0][1],
            'harvest_demand_ratio': prediction[0][2],
            'cost_estimation': prediction[0][3]
        }
    
    def save_model(self, path):
        """Save the trained model and preprocessing objects"""
        torch.save(self.model.state_dict(), f"{path}/model.pth")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        joblib.dump(self.encoder, f"{path}/encoder.pkl")
        joblib.dump(self.target_scaler, f"{path}/target_scaler.pkl")
    
    def load_model(self, path):
        """Load a trained model and preprocessing objects"""
        self.model = RTRWHModel(len(self.numerical_cols) + len(self.encoder.get_feature_names_out()))
        self.model.load_state_dict(torch.load(f"{path}/model.pth"))
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        self.encoder = joblib.load(f"{path}/encoder.pkl")
        self.target_scaler = joblib.load(f"{path}/target_scaler.pkl")

# Example usage
import os

if __name__ == "__main__":
    # Create predictor instance
    predictor = RTRWHPredictor()
    model_dir = "."
    model_files = [
        os.path.join(model_dir, "model.pth"),
        os.path.join(model_dir, "scaler.pkl"),
        os.path.join(model_dir, "encoder.pkl")
    ]
    # Check if all model files exist
    if all(os.path.exists(f) for f in model_files):
        # Load model and preprocessing objects
        # Load encoder first to get correct input dimension
        predictor.encoder = joblib.load(os.path.join(model_dir, "encoder.pkl"))
        predictor.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        predictor.load_model(model_dir)
        print("Loaded existing model and preprocessing objects.")
    else:
        # Train model
        df = pd.read_csv("/Users/shantanutapole/Documents/Projects/AquaSpatial/ML/synthetic_dataset.csv")
        predictor.train(df)
        predictor.save_model(model_dir)
        print("Trained new model and saved preprocessing objects.")

    # Example prediction (after loading/training)
    sample_input = {
        'rooftop_area': 90,
        'dwellers': 2,
        'roof_material': 'metal',
        'annual_rainfall': 1500,
        'aquifer_depth': 20,
        'soil_type': 'loamy',
        'slope': 3,
        'drainage_density': 0.9,
        'runoff_coefficient': 0.72
    }
    prediction = predictor.predict(sample_input)
    print(prediction)