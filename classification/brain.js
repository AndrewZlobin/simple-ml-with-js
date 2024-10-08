// Data Converters
const cloudCoverToNum = (cloudCover) => {
    switch (cloudCover) {
        case 'overcast':
            return 0;
        case 'partly cloudy':
            return 1;
        case 'clear':
            return 2;
        case 'cloudy':
            return 3;
        default:
            return 4;
    }
};

const seasonToNum = (season) => {
    switch (season) {
        case 'Winter':
            return 0;
        case 'Spring':
            return 1;
        case 'Autumn':
            return 2;
        case 'Summer':
            return 3;
        default:
            return 4;
    }
};

const locationToNum = (location) => {
    switch (location) {
        case 'inland':
            return 0;
        case 'mountain':
            return 1;
        case 'coastal':
            return 2;
        default:
            return 3;
    }
};

//const weatherTypeToNum = (weatherType) => {
//    switch (weatherType) {
//        case 'Rainy':
//            return 0;
//        case 'Cloudy':
//            return 1;
//        case 'Sunny':
//            return 2;
//        case 'Snowy':
//            return 3;
//        default:
//            return 4;
//    }
//};

const convertInput = (item) => ({
        temperature: item['Temperature'],
        humidity: item['Humidity'],
        wind_speed: item['Wind Speed'],
        precipitation: item['Precipitation (%)'],
        cloud_cover: cloudCoverToNum(item['Cloud Cover']),
        atmospheric_pressure: item['Atmospheric Pressure'],
        uv_index: item['UV Index'],
        season: seasonToNum(item['Season']),
        visibility: item['Visibility (km)'],
        location: locationToNum(item['Location']),
});

const convertOutput = (item) => ({
    weather_type: item['Weather Type'],
});

// First of all, we create a variable classifier to store the neural network model.
let classifier;


// Set up the backend to "webgl"
ml5.setBackend('webgl');

// Initialize the neural network
classifier = ml5.neuralNetwork({
    task: 'classification',
    debug: true,
});

// Load and process data
async function loadAndProcessData() {
    try {
        const response = await fetch('data.json');
        const weatherDetails = await response.json();
        console.log('Loaded data:', weatherDetails.length);

        weatherDetails.forEach(item => {
            const input = convertInput(item);
            const output = convertOutput(item);

            classifier.addData(input, output);
        });

        console.log("Data loaded and processed");
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

// Normalize data
function normalizeData() {
    classifier.normalizeData();
    console.log("Data normalized");
}

// Training options
const trainingOptions = {
    epochs: 64,
    batchSize: 2048
};

// Function to monitor training
function whileTraining(epoch, loss) {
    console.log(`Epoch: ${epoch}, Loss: ${loss.loss}`);
}

// Function to handle after training is finished
function finishedTraining() {
    console.log('Model training finished');
    document.getElementById('classifyBtn').disabled = false;
}

// Train the model
async function trainModel() {
    await classifier.train(trainingOptions, whileTraining, finishedTraining);
}

// Classify function
function predictScore() {
    const input = {
        temperature: parseFloat(document.getElementById('temperature').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        wind_speed: parseFloat(document.getElementById('wind_speed').value),
        precipitation: parseFloat(document.getElementById('precipitation').value),
        cloud_cover: parseFloat(document.getElementById('cloud_cover').value),
        atmospheric_pressure: parseFloat(document.getElementById('atmospheric_pressure').value),
        uv_index: parseFloat(document.getElementById('uv_index').value),
        season: parseFloat(document.getElementById('season').value),
        visibility: parseFloat(document.getElementById('visibility').value),
        location: parseFloat(document.getElementById('location').value),
}

    console.log('Input for prediction:', input);

    classifier.classify(input, (results) => {
        const result = results[0];
        console.log(results);

        const resultsContainer = document.getElementById('classifyResult');
        resultsContainer.textContent = `The weather would be ${result.label} (with confidence ${result.confidence})`;
        resultsContainer.classList.remove('d-none');
    });
}

// Initialize the app
async function init() {
    try {
        await loadAndProcessData();
        normalizeData();
        await trainModel();
        document
            .getElementById('classifyBtn')
            .addEventListener('click', predictScore);
    } catch (e) {
        console.error('Error initializing:', e);
    }
}

init();