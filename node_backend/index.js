const express = require("express");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");
const cors = require("cors");

const app = express();
const PORT = 3000;

// Enable CORS for all origins (React frontend)
app.use(cors());
app.use(express.json());

// Use multer to handle file uploads in memory
const upload = multer({ storage: multer.memoryStorage() });

// Simple GET route to check server
app.get("/", (req, res) => {
  res.send("Node server running!");
});

// POST /predict route
app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    console.log("Received file:", req.file.originalname, req.file.size);

    // Prepare FormData to send to Flask
    const formData = new FormData();
    formData.append("image", req.file.buffer, req.file.originalname);

    // Send POST request to Flask
    const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
      headers: formData.getHeaders(),
    });

    console.log("Flask response:", response.data);

    res.json(response.data); // Forward Flask response to React
  } catch (err) {
    console.error("Error calling Flask:", err.response?.data || err.message);
    res.status(500).json({ error: "Prediction failed" });
  }
});

app.post("/rag", async (req, res) => {
  try {
    const { disease, question } = req.body;
    const response = await axios.post("http://127.0.0.1:5000/rag", { disease, question });
    res.json(response.data);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "RAG request failed" });
  }
});


// Start Node server
app.listen(PORT, () => {
  console.log(`Node server running on http://localhost:${PORT}`);
});






// const express = require("express");
// const multer = require("multer");
// const axios = require("axios");
// const FormData = require("form-data");

// const app = express();
// const upload = multer({ storage: multer.memoryStorage() }); // store file in memory

// app.post("/predict", upload.single("image"), async (req, res) => {
//   try {
//     if (!req.file) return res.status(400).json({ error: "No file uploaded" });

//     const formData = new FormData();
//     formData.append("image", req.file.buffer, req.file.originalname);

//     const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
//       headers: formData.getHeaders(),
//     });

//     res.json(response.data);
//   } catch (err) {
//     console.error(err.message);
//     res.status(500).json({ error: "Prediction failed" });
//   }
// });

// app.listen(3000, () => console.log("Node server running on port 3000"));
