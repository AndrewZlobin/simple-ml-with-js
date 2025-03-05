// URL to Trained Cloud model from Teachable Machines
const modelUrl = 'https://teachablemachine.withgoogle.com/models/WxAOawMrn/';
// We create a variable classifier to store the neural network model.
let classifier;
// Action buttons and inputs
const uploadButton = document.getElementById('uploadImage');
const classifyButton = document.getElementById('classifyDogEmotion');
// Card elements and variables
const imageContainer = document.getElementById('imageContainer');
const resultContainer = document.getElementById('resultContainer');
// Uploaded image
let uploadedImage;

function preload() {
    classifier = ml5.imageClassifier(`${modelUrl}model.json`, {
        flipped: true,
    });
}

function uploadImage(event) {
    const files = event.target.files;

    if (files.length === 0) {
        return;
    }

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const img = document.createElement("img");
        img.src = URL.createObjectURL(file);
        img.onload = () => URL.revokeObjectURL(img.src);

        imageContainer.src = img.src;
        imageContainer.classList.remove('d-none');

        classifyButton.disabled = false;

        uploadedImage = img;
    }
}

function classifyDogEmotion() {
    if (uploadedImage) {
        classifier.classify(uploadedImage, (output, error) => {
            if (!!error) {
                resultContainer.innerText = error?.toString();
            } else {
                const first = output?.at(0);
                const label = first?.label?.toString() || '';
                const confidence = first?.confidence?.toPrecision(2) || '';

                resultContainer.innerText = `This dog looks ${label} with confidence ${confidence * 100}%`;
            }
        });
    }
}

preload();
uploadButton.addEventListener('change', uploadImage);
classifyButton.addEventListener('click', classifyDogEmotion);
