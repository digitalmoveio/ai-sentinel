/**
 * AI Sentinel - Frontend Application
 */

document.addEventListener("DOMContentLoaded", () => {
    const uploadArea = document.getElementById("uploadArea");
    const fileInput = document.getElementById("fileInput");
    const filePreview = document.getElementById("filePreview");
    const imagePreview = document.getElementById("imagePreview");
    const videoPreview = document.getElementById("videoPreview");
    const fileName = document.getElementById("fileName");
    const fileSize = document.getElementById("fileSize");
    const removeFile = document.getElementById("removeFile");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const resultsSection = document.getElementById("resultsSection");
    const newAnalysis = document.getElementById("newAnalysis");

    let selectedFile = null;

    // ---- Upload Area Events ----

    uploadArea.addEventListener("click", () => fileInput.click());

    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.classList.add("dragover");
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("dragover");
    });

    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    removeFile.addEventListener("click", resetUpload);

    analyzeBtn.addEventListener("click", analyzeFile);

    newAnalysis.addEventListener("click", () => {
        resetUpload();
        resultsSection.style.display = "none";
        window.scrollTo({ top: 0, behavior: "smooth" });
    });

    // ---- File Handling ----

    function handleFile(file) {
        const maxSize = 100 * 1024 * 1024; // 100MB
        if (file.size > maxSize) {
            alert("File too large. Maximum size is 100MB.");
            return;
        }

        selectedFile = file;

        // Show preview
        uploadArea.style.display = "none";
        filePreview.style.display = "block";
        fileName.textContent = file.name;
        fileSize.textContent = formatSize(file.size);

        if (file.type.startsWith("image/")) {
            imagePreview.src = URL.createObjectURL(file);
            imagePreview.style.display = "block";
            videoPreview.style.display = "none";
        } else if (file.type.startsWith("video/")) {
            videoPreview.src = URL.createObjectURL(file);
            videoPreview.style.display = "block";
            imagePreview.style.display = "none";
        }

        analyzeBtn.disabled = false;
        resultsSection.style.display = "none";
    }

    function resetUpload() {
        selectedFile = null;
        fileInput.value = "";
        uploadArea.style.display = "block";
        filePreview.style.display = "none";
        imagePreview.style.display = "none";
        imagePreview.src = "";
        videoPreview.style.display = "none";
        videoPreview.src = "";
        analyzeBtn.disabled = true;
        showBtnText(true);
    }

    // ---- Analysis ----

    async function analyzeFile() {
        if (!selectedFile) return;

        showBtnText(false);
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await fetch("/api/analyze", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (data.error) {
                alert("Error: " + data.error);
                return;
            }

            displayResults(data);
        } catch (err) {
            alert("Analysis failed. Please try again.");
            console.error(err);
        } finally {
            showBtnText(true);
            analyzeBtn.disabled = false;
        }
    }

    // ---- Display Results ----

    function displayResults(data) {
        const result = data.result;
        resultsSection.style.display = "block";

        // Animate gauge
        const aiProb = result.ai_probability;
        const realProb = result.real_probability;
        const confidence = result.confidence;

        // Update gauge
        const gaugeFill = document.getElementById("gaugeFill");
        const gaugeValue = document.getElementById("gaugeValue");
        const gaugeLabel = document.getElementById("gaugeLabel");

        const circumference = 2 * Math.PI * 85; // ~534
        const offset = circumference * (1 - aiProb / 100);

        // Color based on AI probability
        let gaugeColor;
        if (aiProb > 70) gaugeColor = "var(--color-ai)";
        else if (aiProb > 40) gaugeColor = "var(--color-warning)";
        else gaugeColor = "var(--color-real)";

        setTimeout(() => {
            gaugeFill.style.strokeDashoffset = offset;
            gaugeFill.style.stroke = gaugeColor;
        }, 100);

        gaugeValue.textContent = aiProb.toFixed(1) + "%";
        gaugeValue.style.color = gaugeColor;
        gaugeLabel.textContent = aiProb > 50 ? "AI Generated" : "Likely Real";

        // Update bars
        document.getElementById("aiPercent").textContent = aiProb.toFixed(1) + "%";
        document.getElementById("realPercent").textContent = realProb.toFixed(1) + "%";

        setTimeout(() => {
            document.getElementById("aiBar").style.width = aiProb + "%";
            document.getElementById("realBar").style.width = realProb + "%";
        }, 100);

        // Confidence
        document.getElementById("confidenceValue").textContent = confidence.toFixed(1) + "%";

        // Analysis details
        const grid = document.getElementById("analysisGrid");
        grid.innerHTML = "";

        const details = result.analysis_details || {};
        const methodNames = {
            error_level_analysis: "Error Level Analysis",
            frequency_analysis: "Frequency Domain",
            statistical_analysis: "Statistical Analysis",
            metadata_analysis: "Metadata Analysis",
            texture_analysis: "Texture Analysis",
            temporal_analysis: "Temporal Consistency",
            motion_analysis: "Motion Analysis",
        };

        for (const [key, detail] of Object.entries(details)) {
            const card = document.createElement("div");
            card.className = "analysis-card";

            const score = detail.score || 0;
            const scoreClass = score > 0.6 ? "high" : score > 0.35 ? "medium" : "low";
            const scorePercent = (score * 100).toFixed(1);

            card.innerHTML = `
                <h4><span class="score-dot ${scoreClass}"></span> ${methodNames[key] || key}</h4>
                <div class="card-score">${scorePercent}%</div>
                <div class="card-interpretation">${detail.interpretation || ""}</div>
            `;
            grid.appendChild(card);
        }

        // Video-specific (temporal + motion at top level)
        if (result.temporal_analysis && result.temporal_analysis.score !== undefined) {
            const card = document.createElement("div");
            card.className = "analysis-card";
            const s = result.temporal_analysis.score;
            const sc = s > 0.6 ? "high" : s > 0.35 ? "medium" : "low";
            card.innerHTML = `
                <h4><span class="score-dot ${sc}"></span> Temporal Consistency</h4>
                <div class="card-score">${(s * 100).toFixed(1)}%</div>
                <div class="card-interpretation">${result.temporal_analysis.interpretation || ""}</div>
            `;
            grid.appendChild(card);
        }

        if (result.motion_analysis && result.motion_analysis.score !== undefined) {
            const card = document.createElement("div");
            card.className = "analysis-card";
            const s = result.motion_analysis.score;
            const sc = s > 0.6 ? "high" : s > 0.35 ? "medium" : "low";
            card.innerHTML = `
                <h4><span class="score-dot ${sc}"></span> Motion Analysis</h4>
                <div class="card-score">${(s * 100).toFixed(1)}%</div>
                <div class="card-interpretation">${result.motion_analysis.interpretation || ""}</div>
            `;
            grid.appendChild(card);
        }

        // Warnings
        const warnings = result.warnings || [];
        const warningsArea = document.getElementById("warningsArea");
        const warningsList = document.getElementById("warningsList");

        if (warnings.length > 0) {
            warningsArea.style.display = "block";
            warningsList.innerHTML = warnings.map((w) => `<li>${w}</li>`).join("");
        } else {
            warningsArea.style.display = "none";
        }

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // ---- Utilities ----

    function showBtnText(showText) {
        document.querySelector(".btn-text").style.display = showText ? "inline" : "none";
        document.querySelector(".btn-loading").style.display = showText ? "none" : "inline-flex";
    }

    function formatSize(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }
});
