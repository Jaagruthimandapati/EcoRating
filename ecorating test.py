from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


# Assuming X_train and y_train are already defined and your model is trained
# Initialize and train model - For real use, this should be outside of the script or protected under a check
# model = RandomForestRegressor(random_state=42)
# model.fit(X_train, y_train)
# joblib.dump(model, 'model.pkl')

app = Flask(__name__)

# Load the trained model
model = joblib.load('C:\\Users\\manda\\OneDrive\\Desktop\\mini projjjj\\mini\\model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input data from the user
        product_name = request.form['product_name']  # Not used in prediction
        duriability = float(request.form['durability'])
        material = float(request.form['material'])
        packing = float(request.form['packing'])
        carbon_emissions = float(request.form['carbon_emissions'])
        water_usage = float(request.form['water_usage'])
        reusability = float(request.form['reusability'])
        dispossability = float(request.form['disposability'])
        transportation = float(request.form['transportation'])
        energy_usage = float(request.form['energy_usage'])
        type_of_energy_usage = float(request.form['type_of_energy_usage'])

        # Create input array for prediction
        input_data = np.array([[duriability, material, packing, carbon_emissions,
                                water_usage, reusability, dispossability, transportation,
                                energy_usage, type_of_energy_usage]])

        # Predict the ecoscore
        predicted_ecoscore = round(model.predict(input_data)[0],2)

        return render_template('max.html', predicted_ecoscore=predicted_ecoscore)

    return render_template('max.html')

if __name__ == '__main__':
    app.run(debug=True)