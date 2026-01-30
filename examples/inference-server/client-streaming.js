/* Client for streaming ASR - this is for TESTING PURPOSES */
import WebSocket from "ws";
import fs from "fs";

const args = process.argv.slice(2);
let endpoint = "";
let filePath = "";

for (let i = 0; i < args.length; i++) {
    if (args[i] === "-e" && args[i + 1]) {
        endpoint = args[i + 1];
        i++;
    } else if (args[i] === "-f" && args[i + 1]) {
        filePath = args[i + 1];
        i++;
    }
}

if (!endpoint || !filePath) {
    console.error("Usage: node client-streaming.js -e <ws_url> -f <audio_file>");
    process.exit(1);
}

const CHUNK_BYTES = 640;
const ws = new WebSocket(endpoint);

console.log(`Connecting to ${endpoint}...`);

const stats = fs.statSync(filePath);
const duration = stats.size / 32000.0;
console.log(`Audio Duration: ${duration.toFixed(2)}s`);

const startTime = Date.now();

ws.on("open", () => {
    console.log("Connected.");
    ws.send(JSON.stringify({
        type: "start",
        format: "pcm_s16le",
        sample_rate_hz: 16000,
        channels: 1
    }));

    try {
        const fd = fs.openSync(filePath, "r");
        const buf = Buffer.alloc(CHUNK_BYTES);

        const sendChunk = () => {
            const n = fs.readSync(fd, buf, 0, CHUNK_BYTES, null);
            if (n > 0) {
                // Must copy buffer because ws.send is async and we reuse buf immediately
                ws.send(Buffer.from(buf.subarray(0, n)));
                setImmediate(sendChunk);
            } else {
                fs.closeSync(fd);
                ws.send(JSON.stringify({ type: "stop" }));
                console.log("Finished sending audio.");
            }
        };

        sendChunk();
    } catch (err) {
        console.error(`Error reading file: ${err.message}`);
        ws.close();
    }
});

ws.on("message", (data) => {
    try {
        const timestamp = new Date().toISOString().split('T')[1].slice(0, -1);
        const evt = JSON.parse(data.toString());
        if (evt.type === 'partial') {
            process.stdout.write(`\r[${timestamp}] [Partial] ${evt.text}`);
        } else if (evt.type === 'final') {
            console.log(`\n[${timestamp}] [Final] ${evt.text}`);
        } else {
            console.log(`\n[${timestamp}] [${evt.type}] ${JSON.stringify(evt)}`);
        }
    } catch {
        console.log("\n[Non-JSON]", data.toString());
    }
});

ws.on("close", () => {
    const endTime = Date.now();
    const processTime = (endTime - startTime) / 1000.0;
    const rtf = processTime / duration;
    console.log(`\nProcessing Time: ${processTime.toFixed(2)}s`);
    console.log(`Real-Time Factor (RTF): ${rtf.toFixed(4)}`);
    console.log("\nDisconnected.");
});

ws.on("error", (err) => {
    console.error(`WebSocket error: ${err.message}`);
});
