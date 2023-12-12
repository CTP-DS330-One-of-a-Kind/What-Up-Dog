
function uploadImage() {
    var formData = new FormData(document.getElementById('upload-form'));
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('prediction-result').innerText = data.prediction;
        if (data.filename) {
            var img = document.getElementById('uploaded-image');
            img.src = '/uploads/' + data.filename;
            img.style.display = 'block';
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}