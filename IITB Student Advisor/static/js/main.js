document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("metricsForm");
    const errBanner = document.getElementById("errorBanner");
    const outputCard = document.getElementById("outputCard");
    const waitingCard = document.getElementById("waitingCard");
    
    const timeFields = ["SleepHoursPerNight", "StudyHoursPerDay", "PassiveEntertainmentHrs", "SocialMediaHours", "TotalScreenTime"];

    function checkTimeLimits() {
        let sleep = parseFloat(document.getElementById("SleepHoursPerNight").value) || 0;
        let study = parseFloat(document.getElementById("StudyHoursPerDay").value) || 0;
        let passive = parseFloat(document.getElementById("PassiveEntertainmentHrs").value) || 0;
        let social = parseFloat(document.getElementById("SocialMediaHours").value) || 0;
        let screen = parseFloat(document.getElementById("TotalScreenTime").value) || 0;

        let err = null;
        if (sleep + study + passive > 24) err = "Sleep + Study + Passive Entertainment cannot exceed 24 hrs";
        else if (sleep + social + study > 24) err = "Sleep + Social Media + Study cannot exceed 24 hrs";
        else if (sleep + screen > 24) err = "Sleep + Total Screen Time cannot exceed 24 hrs";

        timeFields.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                if (err && el.value !== "") el.classList.add("invalid");
                else el.classList.remove("invalid");
            }
        });
        return err;
    }

    // Check custom 24-hour equations dynamically as the user types
    document.querySelectorAll("input, select").forEach(el => {
        el.addEventListener("input", function() {
            if (timeFields.includes(el.id)) {
                checkTimeLimits();
            }
            errBanner.classList.add("hidden"); // clear banner on typing
        });
    });

    // The submit event ONLY fires if the native HTML5 validation (min/max/required) passes
    form.addEventListener("submit", async function(e) {
        e.preventDefault(); 
        errBanner.classList.add("hidden");
        
        // Final check on the custom 24-hour equations
        let timeErr = checkTimeLimits();
        if (timeErr) {
            errBanner.textContent = timeErr;
            errBanner.classList.remove("hidden");
            window.scrollTo(0, 0);
            return;
        }

        const btn = document.getElementById("calcBtn");
        btn.disabled = true;
        btn.textContent = "Processing...";

        // Collect all data
        const data = {};
        const inputs = form.querySelectorAll("input, select");
        inputs.forEach(input => {
            data[input.id] = input.value;
        });

        try {
            const res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.error || "Server failed");
            }

            const result = await res.json();

            // Show results UI
            waitingCard.style.display = "none";
            outputCard.style.display = "block";

            // Update grade text
            document.getElementById("finalScore").textContent = result.predicted_grade.toFixed(1);
            
            // Update risk pill
            const badge = document.getElementById("riskBadge");
            badge.textContent = result.status;
            badge.className = "badge " + result.status;

            // Update SVG gauge
            const gauge = document.getElementById("gradeBar");
            const offset = 282.7 - (282.7 * result.predicted_grade) / 100;
            gauge.style.strokeDashoffset = offset;
            
            if (result.status === "good") gauge.style.stroke = "var(--color-good)";
            else if (result.status === "average") gauge.style.stroke = "var(--color-avg)";
            else gauge.style.stroke = "var(--color-risk)";

            // Print AI Advice
            document.getElementById("aiAdvice").textContent = result.recommendation;

        } catch (err) {
            errBanner.textContent = err.message;
            errBanner.classList.remove("hidden");
            window.scrollTo(0, 0);
        } finally {
            btn.disabled = false;
            btn.textContent = "Predict Grade";
        }
    });
});