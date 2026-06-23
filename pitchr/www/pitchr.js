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

// Pipeline sample rate; must match pitchr.pitch.pipeline.SAMPLE_RATE.
const TARGET_SAMPLE_RATE = 22050;

// Decode the recorded (compressed) audio and resample it to mono float32 PCM at
// TARGET_SAMPLE_RATE. The backend expects raw float32 samples, so the codec is
// handled entirely in the browser via the Web Audio API.
async function decodeToPCM(blob) {
	const arrayBuffer = await blob.arrayBuffer();
	const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
	const decoded = await decodeCtx.decodeAudioData(arrayBuffer);
	decodeCtx.close();

	const frameCount = Math.ceil(decoded.duration * TARGET_SAMPLE_RATE);
	const offline = new OfflineAudioContext(1, frameCount, TARGET_SAMPLE_RATE);
	const source = offline.createBufferSource();
	source.buffer = decoded;
	source.connect(offline.destination);
	source.start();
	const rendered = await offline.startRendering();
	return rendered.getChannelData(0); // Float32Array, mono, 22050 Hz
}

// Base64-encode a Float32Array's raw bytes without blowing the call stack on
// large buffers (String.fromCharCode(...hugeArray) overflows).
function float32ToBase64(float32) {
	const bytes = new Uint8Array(float32.buffer);
	let binary = "";
	const chunk = 0x8000;
	for (let i = 0; i < bytes.length; i += chunk) {
		binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
	}
	return btoa(binary);
}

async function handleStop() {
	const blob = new Blob(audioChunks, { type: "audio/webm" });
	let base64;
	try {
		const pcm = await decodeToPCM(blob);
		base64 = float32ToBase64(pcm);
	} catch (e) {
		showError("Could not process the recording. Try again.");
		return;
	}

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
