const video = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const emotionText = document.getElementById("emotion");

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
    })
    .catch((err) => {
        console.error("Error accessing webcam:", err);
    });

async function captureAndSend() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert frame to Blob
    canvas.toBlob(async (blob) => {
        let formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        try {
            // Send frame to FastAPI for prediction
            const response = await fetch("http://127.0.0.1:8000/predict/", {
                method: "POST",
                body: formData
            });

            const result = await response.json();
            emotionText.innerText = "Emotion: " + result.emotion;

            // Draw emotion label on canvas
            ctx.font = "20px Arial";
            ctx.fillStyle = "red";
            ctx.fillText(result.emotion, 10, 50);

        } catch (error) {
            console.error("Error fetching prediction:", error);
        }
    }, "image/jpeg");
}

// Capture and send frame every second
setInterval(captureAndSend, 1000);
