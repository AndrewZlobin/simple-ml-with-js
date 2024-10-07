// Data Converters
const fuelToNum = (fuel) => {
    switch (fuel) {
        case 'Petrol':
            return 0;
        case 'Diesel':
            return 1;
        case 'CNG':
            return 2;
        case 'LPG':
            return 3;
        case 'Electric':
            return 4;
        default:
            return 5;
    }
};

const sellerTypeToNum = (sellerType) => {
    switch (sellerType) {
        case 'Individual':
            return 0;
        case 'Dealer':
            return 1;
        case 'Trustmark Dealer':
            return 2;
        default:
            return 3;
    }
};

const transmissionToNum = (transmission) => {
    switch (transmission) {
        case 'Manual':
            return 0;
        case 'Automatic':
            return 1;
        default:
            return 2;
    }
};

const ownerToNum = (owner) => {
    switch (owner) {
        case 'First Owner':
            return 0;
        case 'Second Owner':
            return 1;
        case 'Third Owner':
            return 2;
        case 'Fourth & Above Owner':
            return 3;
        case 'Test Drive Car':
            return 4;
        default:
            return 5;
    }
};

// Set up the backend to "webgl"
ml5.setBackend('webgl');

// Initialize the neural network
const brain = ml5.neuralNetwork({
    task: 'regression',
    debug: true,
    layers: [
        { type: 'dense', units: 64, activation: 'relu' },
        { type: 'dense', units: 64, activation: 'relu' },
        { type: 'dense', units: 1, activation: 'linear' }
    ]
});

// Load and process data
async function loadAndProcessData() {
    try {
        const response = await fetch('data.json');
        const carsDetails = await response.json();
        console.log('Loaded data:', carsDetails);

        carsDetails.forEach(carDetails => {
            const input = {
                year: parseFloat(carDetails.year),
                km_driven: parseFloat(carDetails.km_driven),
                fuel: fuelToNum(carDetails.fuel),
                seller_type: sellerTypeToNum(carDetails.seller_type),
                transmission: transmissionToNum(carDetails.transmission),
                owner: ownerToNum(carDetails.owner),
            };

            const output = {
                selling_price: parseFloat(carDetails.selling_price) / 1000000
            };

            brain.addData(input, output);
        });

        console.log("Data loaded and processed");
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

// Normalize data
function normalizeData() {
    brain.normalizeData();
    console.log("Data normalized");
}

// Training options
const trainingOptions = {
    epochs: 32,
    batchSize: 512
};

// Function to monitor training
function whileTraining(epoch, loss) {
    console.log(`Epoch: ${epoch}, Loss: ${loss.loss}`);
}

// Function to handle after training is finished
function finishedTraining() {
    console.log('Model training finished');
    document.getElementById('calculateBtn').disabled = false;
}

// Train the model
async function trainModel() {
    await brain.train(trainingOptions, whileTraining, finishedTraining);
}

// Predict function
function predictScore() {
    const input = {
        year: parseFloat(document.getElementById('year').value),
        km_driven: parseFloat(document.getElementById('km_driven').value),
        fuel: fuelToNum(document.getElementById('fuel').value),
        seller_type: sellerTypeToNum(document.getElementById('seller_type').value),
        transmission: transmissionToNum(document.getElementById('transmission').value),
        owner: ownerToNum(document.getElementById('owner').value),
    };

    console.log('Input for prediction:', input);

    brain.predict(input, (results) => {
        console.log('Prediction:', results);

        // Scale the prediction back up
        const predictedPrice = results[0].value * 1000000;

        document.getElementById('calculationResultsContainer').classList.remove('d-none');
        document.getElementById('calculationResults').textContent = predictedPrice < 0
             ? 0
             : predictedPrice.toFixed(2);
    });
}

// Initialize the app
async function init() {
    try {
        await loadAndProcessData();
        normalizeData();
        await trainModel();
        document
            .getElementById('calculateBtn')
            .addEventListener('click', predictScore);
    } catch (e) {
        console.error('Error initializing:', e);
    }
}

init();