

import React, { useState } from "react";
import axios from "axios";
import "./App.css";
import ReactMarkdown from "react-markdown";


function App() {
  const [user, setUser] = useState(null);
  const [isRegistering, setIsRegistering] = useState(false);
  const [username, setUsername] = useState(""); // <-- add username
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [latestUploadId, setLatestUploadId] = useState(null);

  const [userQuestion, setUserQuestion] = useState("");
  const [ragAnswer, setRagAnswer] = useState(null);

  const handleLogin = async () => {
    if (!email || !password) return alert("Enter email and password!");
    try {
      const res = await axios.post("http://localhost:5000/login", { email, password });
      if (res.data.user_id) {
        setUser({ id: res.data.user_id, username: res.data.username });
        alert("Login successful!");
      } else {
        alert(res.data.error || "Login failed");
      }
    } catch (err) {
      console.error(err);
      alert("Login failed!");
    }
  };

  const handleRegister = async () => {
    if (!username || !email || !password) return alert("Fill all fields!");
    try {
      const res = await axios.post("http://localhost:5000/register", { username, email, password }); // <-- send username
      alert(res.data.message || "Registration successful!");
      setIsRegistering(false);
    } catch (err) {
      console.error(err);
      alert(err.response?.data?.error || "Registration failed!");
    }
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select an image!");
    if (!user) return alert("Please log in first!");

    setLoading(true);
    const formData = new FormData();
    formData.append("image", file);
    formData.append("user_id", user.id);

    try {
      // const res = await axios.post("http://localhost:3000/predict", formData, {
      const res = await axios.post("http://localhost:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      setPrediction(res.data);
      setLatestUploadId(res.data.upload_id);
      setRagAnswer(null); // clear previous RAG
    } catch (err) {
      console.error(err);
      alert("Prediction failed!");
    } finally {
      setLoading(false);
    }
  };

 const handleQuestion = async () => {
  if (!userQuestion) return alert("Please type a question!");
  if (!prediction) return alert("Predict first!");
  try {
    // const res = await axios.post("http://localhost:3000/rag", {
    const res = await axios.post("http://localhost:5000/rag", {
      disease: prediction.label,
      question: userQuestion,
      user_id: user.id,
      upload_id: latestUploadId
    });
 
    setRagAnswer(res.data);
  } catch (err) {
    console.error(err);
    alert("RAG request failed!");
  }
};


  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    setFile(uploadedFile);
    if (uploadedFile) setPreviewUrl(URL.createObjectURL(uploadedFile));
    setPrediction(null);
    setRagAnswer(null);
  };

  return (
    <div className="app-container">
      <h2 className="app-title">DL Model Prediction + RAG</h2>

      {!user && (
        <div className="login-box">
          <h3>{isRegistering ? "Register" : "Login"}</h3>

          {isRegistering && (
            <input
              type="text"
              placeholder="Username"
              value={username}
              onChange={e => setUsername(e.target.value)}
            />
          )}

          <input type="email" placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} />
          <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} />

          <button onClick={isRegistering ? handleRegister : handleLogin} className="btn btn-login">
            {isRegistering ? "Register" : "Login"}
          </button>

          <p onClick={() => setIsRegistering(!isRegistering)} className="switch-form">
            {isRegistering ? "Already have an account? Login" : "Don't have an account? Register"}
          </p>
        </div>
      )}

      {user && (
        <>
          {!prediction ? (
            <div className="centered-predict">
              <h3>Predict</h3>
              <div className="upload-section">
                <input type="file" onChange={handleFileChange} />
                <button onClick={handleUpload} className="btn">{loading ? "Predicting..." : "Predict"}</button>
              </div>
              {previewUrl && <img src={previewUrl} alt="Uploaded" className="uploaded-image" />}
            </div>
          ) : (
            <div className="columns-container">
              <div className="left-column">
                <h3 className="subheading">Predict</h3>
                <div className="prediction-box">
                  <img src={previewUrl} alt="Uploaded" className="uploaded-image" />
                  <div>
                    <strong>Label:</strong> {prediction.label} <br />
                    <strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%
                  </div>
                </div>
              </div>

              <div className="right-column">
                <h3 className="subheading">Ask a question</h3>
                <input
                  type="text"
                  placeholder="Type your question..."
                  value={userQuestion}
                  onChange={(e) => setUserQuestion(e.target.value)}
                />
                <button onClick={handleQuestion}>Get Answer</button>

                {ragAnswer && (
  <div className="rag-answer-box">
    <h4>AI’s Response:</h4>
    <ReactMarkdown>{ragAnswer.answer}</ReactMarkdown>

    {ragAnswer.summary && (
      <div className="summary-box">
        <h4>Summary:</h4>
        <p>{ragAnswer.summary}</p>
      </div>
    )}
  </div>
)}

              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default App;



 