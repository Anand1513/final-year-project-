document.addEventListener('DOMContentLoaded', () => {
    
    // --- Scroll Timeline Line Logic ---
    const scrollLine = document.getElementById('scroll-line');
    
    window.addEventListener('scroll', () => {
        // Calculate scroll percentage
        const scrollTop = window.scrollY;
        const docHeight = document.body.scrollHeight - window.innerHeight;
        const scrollPercent = scrollTop / docHeight;
        
        // Update SVG line dashoffset
        // The stroke-dasharray is 1000, so setting offset from 1000 down to 0
        if (scrollLine) {
            const offset = 1000 - (scrollPercent * 1000);
            scrollLine.style.strokeDashoffset = offset;
        }

        // Sticky Navbar
        const navbar = document.getElementById('navbar');
        if (scrollTop > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // --- Custom Masking Reveal Animation ---
    const revealElements = document.querySelectorAll('.reveal-mask');
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.15
    };

    const revealObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
                // Unobserve after revealing for performance and static feel
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    revealElements.forEach(el => {
        revealObserver.observe(el);
    });

    // --- Presentation (Viva) Mode Toggle ---
    const vivaBtn = document.getElementById('viva-mode-btn');
    let isVivaMode = false;
    
    const vivaControls = document.createElement('div');
    vivaControls.className = 'viva-controls hidden';
    vivaControls.innerHTML = `<button id="exit-viva" class="btn btn-outline" style="background: #fff;"><i class="fa-solid fa-xmark"></i> Exit Presentation</button>`;
    document.body.appendChild(vivaControls);

    const toggleVivaMode = () => {
        isVivaMode = !isVivaMode;
        if (isVivaMode) {
            document.body.classList.add('viva-mode');
            vivaControls.classList.remove('hidden');
            window.scrollTo({ top: 0, behavior: 'smooth' });
        } else {
            document.body.classList.remove('viva-mode');
            vivaControls.classList.add('hidden');
        }
    };

    vivaBtn.addEventListener('click', (e) => {
        e.preventDefault();
        toggleVivaMode();
    });
    
    document.getElementById('exit-viva').addEventListener('click', toggleVivaMode);

    // --- Dual Mode Toggle Logic ---
    const modeNpkBtn = document.getElementById('mode-npk');
    const modeRegionalBtn = document.getElementById('mode-regional');
    const formNpk = document.getElementById('prediction-form');
    const formRegional = document.getElementById('regional-form');
    const terminalOutput = document.getElementById('terminal-output');

    modeNpkBtn.addEventListener('click', () => {
        modeNpkBtn.classList.add('active');
        modeRegionalBtn.classList.remove('active');
        formNpk.style.display = 'block';
        formRegional.style.display = 'none';
        terminalOutput.classList.add('hidden');
    });

    modeRegionalBtn.addEventListener('click', () => {
        modeRegionalBtn.classList.add('active');
        modeNpkBtn.classList.remove('active');
        formRegional.style.display = 'block';
        formNpk.style.display = 'none';
        terminalOutput.classList.add('hidden');
    });

    // --- Terminal Form Submission (7-Parameter Model) ---
    formNpk.addEventListener('submit', (e) => {
        e.preventDefault();
        executeInference();
    });

    // --- Regional Form Submission (Hybrid Model) ---
    formRegional.addEventListener('submit', (e) => {
        e.preventDefault();
        executeRegionalInference();
    });
});

function executeInference() {
    // UI Elements
    const terminalOutput = document.getElementById('terminal-output');
    const submitBtn = document.getElementById('submit-btn');

    // Get 7 Parameters
    const data = {
        nitrogen: Number(document.getElementById("nitrogen").value),
        phosphorous: Number(document.getElementById("phosphorous").value),
        potassium: Number(document.getElementById("potassium").value),
        temperature: Number(document.getElementById("temperature").value),
        humidity: Number(document.getElementById("humidity").value),
        ph: Number(document.getElementById("ph").value),
        rainfall: Number(document.getElementById("rainfall").value)
    };

    // UI Loading State
    terminalOutput.classList.remove('hidden');
    terminalOutput.innerHTML = `> EXECUTING NPK INFERENCE PIPELINE...<br>> TRANSMITTING DATA ARRAY... <span class="blink">_</span>`;
    if(submitBtn) submitBtn.disabled = true;

    fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        body: JSON.stringify(data),
    })
    .then((response) => {
        if (!response.ok) return response.json().then(err => { throw new Error(err.detail || "Server exception") });
        return response.json();
    })
    .then((responseData) => {
        const predictedCrops = responseData["result"];
        const cropsHtml = predictedCrops.map((c, i) => `#${i+1} ${c.crop.toUpperCase()} ${c.confidence}%`).join('<br>');
        setTimeout(() => {
            terminalOutput.innerHTML = `
<div style="color: #8b949e; margin-bottom: 5px;">7-PARAMETER INPUT</div>
N: ${data.nitrogen}<br>
P: ${data.phosphorous}<br>
K: ${data.potassium}<br>
Temperature: ${data.temperature}°C<br>
Humidity: ${data.humidity}%<br>
pH: ${data.ph}<br>
Rainfall: ${data.rainfall} mm<br><br>

<div style="color: #8b949e; margin-bottom: 5px;">RANDOM FOREST OUTPUT</div>
<strong style="color: #4ade80; font-size: 1.2rem;">${cropsHtml}</strong><br><br>
<div id="ai-explanation" style="color: #ffbd2e; font-style: italic;">> [AI ASSISTANT GENERATING EXPLANATION...]</div>
<div id="xai-plots" style="color: #ffbd2e; font-style: italic;">> [GENERATING EXPLAINABLE AI PLOTS...]</div>
            `;
            
            // Fetch AI Explanation
            fetch("http://127.0.0.1:8000/explain/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    crop: predictedCrops[0].crop,
                    nitrogen: data.nitrogen, phosphorous: data.phosphorous, potassium: data.potassium,
                    temperature: data.temperature, humidity: data.humidity, ph: data.ph, rainfall: data.rainfall
                })
            })
            .then(res => res.json())
            .then(explainData => {
                const aiDiv = document.getElementById("ai-explanation");
                if(aiDiv) {
                    aiDiv.innerHTML = `<div style="color: #8b949e; margin-bottom: 5px; margin-top: 10px;">GEMINI AI ASSISTANT</div><span style="color: #58a6ff;">${explainData.explanation}</span>`;
                }
            })
            .catch(err => {
                const aiDiv = document.getElementById("ai-explanation");
                if(aiDiv) aiDiv.innerHTML = `<span style="color: #f85149;">> [AI EXPLANATION FAILED]</span>`;
            });
            
            // Fetch XAI Plots
            fetch("http://127.0.0.1:8000/explain_plots/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    crop: predictedCrops[0].crop,
                    nitrogen: data.nitrogen, phosphorous: data.phosphorous, potassium: data.potassium,
                    temperature: data.temperature, humidity: data.humidity, ph: data.ph, rainfall: data.rainfall
                })
            })
            .then(res => res.json())
            .then(plotData => {
                const xaiDiv = document.getElementById("xai-plots");
                if(xaiDiv && plotData.shap_base64) {
                    xaiDiv.innerHTML = `<div style="color: #8b949e; margin-bottom: 5px; margin-top: 10px;">EXPLAINABLE AI (SHAP)</div>
<img src="data:image/png;base64,${plotData.shap_base64}" style="max-width: 100%; border: 1px solid #30363d; border-radius: 6px; margin-top: 10px;">`;
                } else if(xaiDiv) {
                    xaiDiv.innerHTML = "";
                }
            })
            .catch(err => {
                const xaiDiv = document.getElementById("xai-plots");
                if(xaiDiv) xaiDiv.innerHTML = `<span style="color: #f85149;">> [XAI PLOTS FAILED]</span>`;
            });
            
        }, 800);
    })
    .catch((error) => {
        terminalOutput.innerHTML = `> EXCEPTION RAISED.<br>> ERR_DESC: ${error.message}`;
    })
    .finally(() => {
        setTimeout(() => { if(submitBtn) submitBtn.disabled = false; }, 800);
    });
}

function executeRegionalInference() {
    const terminalOutput = document.getElementById('terminal-output');
    const submitBtn = document.getElementById('submit-regional-btn');

    const data = {
        state: document.getElementById("state").value,
        district: document.getElementById("district").value
    };

    terminalOutput.classList.remove('hidden');
    terminalOutput.innerHTML = `> EXECUTING REGIONAL HYBRID PIPELINE...<br>> CALLING OPEN-METEO API FOR LIVE WEATHER... <span class="blink">_</span>`;
    if(submitBtn) submitBtn.disabled = true;

    fetch("http://127.0.0.1:8000/predict_hybrid/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        body: JSON.stringify(data),
    })
    .then((response) => {
        if (!response.ok) return response.json().then(err => { throw new Error(err.detail || "Server exception") });
        return response.json();
    })
    .then((responseData) => {
        const crops = responseData["crop"];
        const validated = responseData["validated"];
        const temp = responseData["live_temp"];
        const hum = responseData["live_humidity"];
        const soil = responseData["soil_data"];
        
        setTimeout(() => {
            let validationMsg = validated ? 
                `<span style="color: #4ade80;">> [REGION VALIDATED - GROWN HISTORICALLY]</span>` : 
                `<span style="color: #ffbd2e;">> [MODEL PREDICTION - MAY NOT BE LOCALLY TRADITIONAL]</span>`;

            const cropsHtml = crops.map((c, i) => `#${i+1} ${c.crop.toUpperCase()} ${c.confidence}%`).join('<br>');

            terminalOutput.innerHTML = `
<div style="color: #8b949e; margin-bottom: 5px;">LOCATION</div>
<strong style="color: #fff;">${data.district.toUpperCase()}, ${data.state.toUpperCase()}</strong><br><br>

<div style="color: #8b949e; margin-bottom: 5px;">7-PARAMETER INPUT</div>
N: ${soil.N}<br>
P: ${soil.P}<br>
K: ${soil.K}<br>
Temperature: ${temp}°C<br>
Humidity: ${hum}%<br>
pH: ${soil.pH}<br>
Rainfall: ${soil.Rainfall} mm<br><br>

<div style="color: #8b949e; margin-bottom: 5px;">RANDOM FOREST OUTPUT</div>
<strong style="color: #4ade80; font-size: 1.2rem;">${cropsHtml}</strong><br><br>
${validationMsg}<br><br>
<div id="ai-explanation" style="color: #ffbd2e; font-style: italic;">> [AI ASSISTANT GENERATING EXPLANATION...]</div>
<div id="xai-plots" style="color: #ffbd2e; font-style: italic;">> [GENERATING EXPLAINABLE AI PLOTS...]</div>
            `;
            
            // Fetch AI Explanation
            fetch("http://127.0.0.1:8000/explain/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    crop: crops[0].crop,
                    nitrogen: soil.N, phosphorous: soil.P, potassium: soil.K,
                    temperature: temp, humidity: hum, ph: soil.pH, rainfall: soil.Rainfall
                })
            })
            .then(res => res.json())
            .then(explainData => {
                const aiDiv = document.getElementById("ai-explanation");
                if(aiDiv) {
                    aiDiv.innerHTML = `<div style="color: #8b949e; margin-bottom: 5px; margin-top: 10px;">GEMINI AI ASSISTANT</div><span style="color: #58a6ff;">${explainData.explanation}</span>`;
                }
            })
            .catch(err => {
                const aiDiv = document.getElementById("ai-explanation");
                if(aiDiv) aiDiv.innerHTML = `<span style="color: #f85149;">> [AI EXPLANATION FAILED]</span>`;
            });
            
            // Fetch XAI Plots
            fetch("http://127.0.0.1:8000/explain_plots/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    crop: crops[0].crop,
                    nitrogen: soil.N, phosphorous: soil.P, potassium: soil.K,
                    temperature: temp, humidity: hum, ph: soil.pH, rainfall: soil.Rainfall
                })
            })
            .then(res => res.json())
            .then(plotData => {
                const xaiDiv = document.getElementById("xai-plots");
                if(xaiDiv && plotData.shap_base64) {
                    xaiDiv.innerHTML = `<div style="color: #8b949e; margin-bottom: 5px; margin-top: 10px;">EXPLAINABLE AI (SHAP)</div>
<img src="data:image/png;base64,${plotData.shap_base64}" style="max-width: 100%; border: 1px solid #30363d; border-radius: 6px; margin-top: 10px;">`;
                } else if(xaiDiv) {
                    xaiDiv.innerHTML = "";
                }
            })
            .catch(err => {
                const xaiDiv = document.getElementById("xai-plots");
                if(xaiDiv) xaiDiv.innerHTML = `<span style="color: #f85149;">> [XAI PLOTS FAILED]</span>`;
            });

        }, 800);
    })
    .catch((error) => {
        terminalOutput.innerHTML = `> EXCEPTION RAISED.<br>> ERR_DESC: ${error.message}`;
    })
    .finally(() => {
        setTimeout(() => { if(submitBtn) submitBtn.disabled = false; }, 800);
    });
}
