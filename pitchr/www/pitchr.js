let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

const recordBtn = document.getElementById("record-btn");
const statusText = document.getElementById("status-text");
const result = document.getElementById("result");
const noMatch = document.getElementById("no-match");
const errorMsg = document.getElementById("error-msg");
const songName = document.getElementById("song-name");
const scoreText = document.getElementById("score-text");

recordBtn.addEventListener("click", async () => {
	if (!isRecording) {
		await startRecording();
	} else {
		stopRecording();
	}
});

async function startRecording() {
	try {
		const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
		mediaRecorder = new MediaRecorder(stream);
		audioChunks = [];

		mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
		mediaRecorder.onstop = handleStop;
		mediaRecorder.start();

		isRecording = true;
		recordBtn.textContent = "⏹";
		recordBtn.classList.replace("btn-primary", "btn-danger");
		statusText.textContent = "Humming... press to stop";
		result.style.display = "none";
		noMatch.style.display = "none";
		errorMsg.style.display = "none";
	} catch (e) {
		showError("Microphone access denied.");
	}
}

function stopRecording() {
	mediaRecorder.stop();
	mediaRecorder.stream.getTracks().forEach((t) => t.stop());
	isRecording = false;
	recordBtn.textContent = "🎤";
	recordBtn.classList.replace("btn-danger", "btn-primary");
	statusText.textContent = "Processing...";
}

async function handleStop() {
	const blob = new Blob(audioChunks, { type: "audio/webm" });
	const arrayBuffer = await blob.arrayBuffer();
	const uint8 = new Uint8Array(arrayBuffer);
	const base64 = btoa(String.fromCharCode(...uint8));

	try {
		const res = await fetch("/api/method/pitchr.pitchr.api.recognize", {
			method: "POST",
			headers: { "Content-Type": "application/json", "X-Frappe-CSRF-Token": "Guest" },
			body: JSON.stringify({ audio_b64: base64 }),
		});
		const data = await res.json();
		const msg = data.message;

		if (msg.matched) {
			songName.textContent = "🎵 " + msg.song_name;
			scoreText.textContent = "Match score: " + msg.score;
			result.style.display = "block";
			statusText.textContent = "Press to hum again";
		} else {
			noMatch.style.display = "block";
			statusText.textContent = "Press to try again";
		}
	} catch (e) {
		showError("Something went wrong. Try again.");
	}
}

function showError(msg) {
	errorMsg.textContent = msg;
	errorMsg.style.display = "block";
	statusText.textContent = "Press to try again";
}
