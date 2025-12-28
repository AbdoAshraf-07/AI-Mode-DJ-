// 1. وظيفة بتنادي على السيرفر كل ثانية عشان تاخد المود المكتشف
async function updateMood() {
    try {
        const response = await fetch('/get_current_mood'); // رابط الباك إند
        const data = await response.json();
        const moodText = document.getElementById('mood-text');
        
        if (data.mood === "Scanning...") {
            moodText.innerText = "جاري التعرف...";
            moodText.style.color = "#777";
        } else {
            moodText.innerText = data.mood;
            moodText.style.color = "#00f2ea";
        }
        document.getElementById('confidence-value').innerText = data.confidence.toFixed(1);
    } catch (e) {
        console.log("في انتظار تشغيل الباك إند...");
    }
}

// 2. وظيفة الزرار اللي بيبعت طلب عمل بلاي ليست
document.getElementById('generate-btn').onclick = async function() {
    const btn = this;
    btn.innerText = "جاري اختيار الأغاني...";
    
    try {
        const response = await fetch('/create_playlist_api', { method: 'POST' });
        const data = await response.json();
        
        if(data.status === "success") {
            // إظهار النتيجة
            document.getElementById('result-box').classList.remove('hidden');
            document.getElementById('target-mood').innerText = data.mood;
            document.getElementById('youtube-link').href = data.url;
            btn.innerText = "تحديث القائمة 🔄";
        }
    } catch (e) {
        alert("تأكد من تشغيل السيرفر (Back-end) أولاً");
        btn.innerText = "اقتراح موسيقى تناسب حالتي 🎵";
    }
};

// تشغيل التحديث المستمر للحالة كل ثانية
setInterval(updateMood, 1000);
