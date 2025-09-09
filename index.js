// server.js
const express = require('express');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// --- fetch data from source API ---
async function fetchData() {
    try {
        const response = await axios.get('https://ahihidonguoccut-2b5i.onrender.com/mohobomaycai');
        return response.data;
    } catch (error) {
        console.error("Error fetching data:", error.message);
        return null;
    }
}

// --- Main predict function (entry) ---
function predict(data) {
    // Expect data.Lich_su_phien (array of {Phien, Ket_qua}) OR fallback use other fields if single session
    const historyRecords = Array.isArray(data.Lich_su_phien) ? data.Lich_su_phien : (data.history || []);
    const history = historyRecords.map(h => (h.Ket_qua || h.ket_qua || h.result || '').trim());

    if (!history || history.length < 20) {
        return {
            predict_goc: "Không có đủ dữ liệu",
            tin_cay: "0%",
            tong_quat_so_do_du_doan: "Không đủ dữ liệu lịch sử (cần ít nhất 20 phiên).",
            giai_thich: "Hệ thống yêu cầu ít nhất 20 phiên lịch sử để phân tích pattern và chạy ensemble AI."
        };
    }

    // Deterministic: extract last 20 for short-term and last 50 (or all) for long-term
    const last20 = history.slice(-20);
    const last50 = history.slice(-50);

    // Generate 20 deterministic pattern samples from history
    const patternSamples = generatePatternSamplesDeterministic(history, 20);

    // Define models (each returns { prediction, explanation })
    const models = [
        { name: 'Model 1: Pattern Analysis', func: model1PatternAnalysis },
        { name: 'Model 2: Rolling Frequency', func: model2FrequencyAnalysis },
        { name: 'Model 3: Markov Chain', func: model3MarkovChain },
        { name: 'Model 4: N-gram Matching', func: model4NgramMatching },
        { name: 'Model 5: Heuristic Ensemble', func: model5Heuristic }
    ];

    const allPredictions = [];
    const explanations = [];

    // Run ensemble: each model on short-term and long-term (deterministic)
    models.forEach((model, idx) => {
        const shortPred = model.func(last20.slice(), patternSamples, `${idx+1}-S`);
        const longPred = model.func(last50.length >= 20 ? last50.slice() : history.slice(), patternSamples, `${idx+1}-L`);
        allPredictions.push(shortPred.prediction);
        allPredictions.push(longPred.prediction);
        explanations.push(shortPred.explanation);
        explanations.push(longPred.explanation);
    });

    // Voting
    const voteTai = allPredictions.filter(p => p === 'Tài').length;
    const voteXiu = allPredictions.filter(p => p === 'Xỉu').length;
    const totalVotes = allPredictions.length;
    let finalResult = 'Không xác định';
    if (voteTai > voteXiu) finalResult = 'Tài';
    else if (voteXiu > voteTai) finalResult = 'Xỉu';
    const confidence = Math.round((Math.max(voteTai, voteXiu) / totalVotes) * 100);

    // Build composite explanation (AI TỔNG HỢP)
    const aiExplanation = buildAiTongHopExplanation({
        finalResult, confidence, voteTai, voteXiu, totalVotes, patternSamples, explanations, history
    });

    return {
        predict_goc: finalResult,
        tin_cay: `${confidence}%`,
        tong_quat_so_do_du_doan: `Từ ${patternSamples.length} mẫu cầu trích xuất, ensemble ${totalVotes} AI (2 phiên bản mỗi model). Kết quả: ${voteTai}/${totalVotes} vote Tài, ${voteXiu}/${totalVotes} vote Xỉu. Dự đoán chính: ${finalResult}.`,
        giai_thich: aiExplanation
    };
}

// ----------------- Deterministic Pattern Sample Generator -----------------
function generatePatternSamplesDeterministic(fullHistory, sampleCount = 20) {
    // Build counts of sliding windows of length 3,4,5 (prefer longer)
    const windows = {};
    for (let n of [5,4,3]) {
        for (let i = 0; i <= fullHistory.length - n - 1; i++) {
            const key = fullHistory.slice(i, i + n).join('-');
            const next = fullHistory[i + n];
            if (!windows[key]) windows[key] = { total: 0, nextCounts: { 'Tài':0, 'Xỉu':0 } };
            windows[key].total++;
            if (next === 'Tài' || next === 'Xỉu') windows[key].nextCounts[next]++;
        }
    }

    // Convert windows to array, sort by (total * entropy reduction) -> prefer frequent and decisive patterns
    const windowList = Object.entries(windows).map(([pattern, info]) => {
        const taiNext = info.nextCounts['Tài'];
        const xiuNext = info.nextCounts['Xỉu'];
        const decisive = Math.abs(taiNext - xiuNext);
        return { pattern, total: info.total, taiNext, xiuNext, decisive };
    }).sort((a,b) => (b.total * b.decisive) - (a.total * a.decisive));

    // Build samples deterministically: take top patterns, and also construct complementary types (bệt, 1-1, 2-2, complex)
    const samples = [];
    // 1) Top explicit patterns
    for (let i = 0; i < windowList.length && samples.length < Math.floor(sampleCount*0.6); i++) {
        const w = windowList[i];
        const next = w.taiNext >= w.xiuNext ? 'Tài' : 'Xỉu';
        samples.push({ type: 'match', pattern: w.pattern, next });
    }

    // 2) Detect runs (bệt)
    const runs = detectRuns(fullHistory);
    for (let r of runs.slice(0, Math.floor(sampleCount*0.15))) {
        samples.push({ type: 'bệt', pattern: r.pattern, next: r.predNext });
    }

    // 3) Detect alternations (1-1)
    const alternations = detectAlternations(fullHistory);
    for (let a of alternations.slice(0, Math.floor(sampleCount*0.1))) {
        samples.push({ type: '1-1', pattern: a.pattern, next: a.predNext });
    }

    // 4) Add balanced canonical patterns if we still need to fill
    const canonical = [
        { pattern: 'Tài-Xỉu-Tài-Xỉu', next: 'Tài' },
        { pattern: 'Xỉu-Tài-Xỉu-Tài', next: 'Xỉu' },
        { pattern: 'Tài-Tài-Xỉu-Xỉu', next: 'Tài' },
        { pattern: 'Xỉu-Xỉu-Tài-Tài', next: 'Xỉu' },
        { pattern: 'Tài-Tài-Tài-Xỉu', next: 'Xỉu' },
        { pattern: 'Xỉu-Xỉu-Xỉu-Tài', next: 'Tài' },
        { pattern: 'Tài-Xỉu-Xỉu-Tài', next: 'Xỉu' },
        { pattern: 'Xỉu-Tài-Tài-Xỉu', next: 'Tài' }
    ];
    let ci = 0;
    while (samples.length < sampleCount && ci < canonical.length) {
        samples.push(Object.assign({ type: 'canonical' }, canonical[ci]));
        ci++;
    }

    // 5) If still less (rare), pad using last20 windows deterministically
    const last20 = fullHistory.slice(-20);
    let idx = 0;
    while (samples.length < sampleCount && idx < Math.max(1, last20.length-3)) {
        const pattern = last20.slice(idx, idx+4).join('-');
        const last = last20[idx+4] || last20[last20.length - 1];
        samples.push({ type: 'pad', pattern, next: (last === 'Tài' ? 'Tài' : 'Xỉu') });
        idx++;
    }

    // Trim to requested sampleCount
    const finalSamples = samples.slice(0, sampleCount);

    // Count Tài/Xỉu next to log
    const taiNextCount = finalSamples.filter(s => s.next === 'Tài').length;
    console.log(`Deterministic samples generated: ${finalSamples.length} (Tài next: ${taiNextCount}, Xỉu next: ${finalSamples.length - taiNextCount})`);
    return finalSamples;
}

function detectRuns(history) {
    const runs = [];
    let i = 0;
    while (i < history.length) {
        let j = i+1;
        while (j < history.length && history[j] === history[i]) j++;
        const length = j - i;
        if (length >= 3) {
            const pattern = history.slice(i, j).join('-');
            const predNext = (history[j] ? history[j] : (history[i] === 'Tài' ? 'Xỉu' : 'Tài')); // deterministic guess: invert next if infinite run end
            runs.push({ start: i, length, pattern, predNext });
        }
        i = j;
    }
    // sort by length desc
    return runs.sort((a,b) => b.length - a.length);
}

function detectAlternations(history) {
    // find places with strict alternation of length >=4 (T-X-T-X)
    const alts = [];
    for (let i = 0; i <= history.length - 4; i++) {
        const seq = history.slice(i, i+4);
        const isAlt = seq[0] !== seq[1] && seq[0] === seq[2] && seq[1] === seq[3];
        if (isAlt) {
            const pattern = seq.join('-');
            const predNext = seq[3] === 'Tài' ? 'Xỉu' : 'Tài';
            alts.push({ index: i, pattern, predNext });
        }
    }
    return alts;
}

// ----------------- Models -----------------

// Model 1 - Pattern Analysis (detect many bridge types)
function model1PatternAnalysis(data, samples, modelId) {
    const last6 = data.slice(-6).join('-');
    let prediction = 'Tài'; // default safe
    let explanation = '';

    // deterministic checks from highest-specificity patterns to low
    if (last6.includes('Tài-Xỉu-Tài-Xỉu')) { // 1-1
        prediction = last6.endsWith('Tài') ? 'Xỉu' : 'Tài';
        explanation = `Model ${modelId} (Pattern): Phát hiện 1-1 (so le) trong recent "${last6}". Theo quy luật đảo, dự đoán ${prediction}.`;
    } else if (last6.includes('Tài-Xỉu-Xỉu-Tài')) { // 1-2-1
        prediction = 'Xỉu';
        explanation = `Model ${modelId} (Pattern): Mẫu 1-2-1 trong "${last6}" — dự đoán gãy sang ${prediction}.`;
    } else if (last6.includes('Xỉu-Tài-Tài-Xỉu')) { // 2-1-2
        prediction = 'Tài';
        explanation = `Model ${modelId} (Pattern): Mẫu 2-1-2 => tiếp theo có xu hướng ${prediction}.`;
    } else if (last6.match(/Tài-Tài-Tài/) || last6.match(/Xỉu-Xỉu-Xỉu/)) { // bệt 3+
        prediction = data[data.length - 1]; // tiếp tục bệt
        explanation = `Model ${modelId} (Pattern): Bệt >3 phát hiện trong "${last6}". Mô hình dự báo tiếp tục bệt: ${prediction}, nhưng cảnh báo gãy nếu chuỗi càng dài.`;
    } else if (last6.match(/Tài-Tài-Xỉu-Xỉu/)) { // 2-2
        prediction = last6.endsWith('Tài') ? 'Xỉu' : 'Tài';
        explanation = `Model ${modelId} (Pattern): Cầu 2-2 trong "${last6}", thông thường đảo sau nhóm: ${prediction}.`;
    } else {
        // fallback: check samples majority
        const taiCount = samples.filter(s => s.next === 'Tài').length;
        prediction = taiCount >= samples.length/2 ? 'Tài' : 'Xỉu';
        explanation = `Model ${modelId} (Pattern): Không match cầu cụ thể trong "${last6}". Dựa vào ${samples.length} mẫu, ${taiCount} mẫu lead đến Tài => dự đoán ${prediction}.`;
    }

    // Add sample support %
    const sampleTaiPct = Math.round(samples.filter(s => s.next === 'Tài').length / samples.length * 100);
    explanation += ` Hỗ trợ từ mẫu: ${sampleTaiPct}% dẫn tới Tài.`;

    return { prediction, explanation };
}

// Model 2 - Rolling Frequency
function model2FrequencyAnalysis(data, samples, modelId) {
    if (!data.length) return { prediction: 'Tài', explanation: `Model ${modelId} (Frequency): Không có dữ liệu.` };
    const total = data.length;
    const taiCount = data.filter(x => x === 'Tài').length;
    const pctTai = Math.round(taiCount / total * 100);
    let prediction;
    let explanation = `Model ${modelId} (Frequency): Trong ${total} phiên gần nhất, Tài = ${pctTai}%. `;

    // deterministic balancing logic: if heavily skewed -> predict opposite (cân bằng),
    // if balanced -> predict according to slight trend
    if (pctTai >= 70) {
        prediction = 'Xỉu';
        explanation += `Tỷ lệ Tài >=70% → kỳ vọng cân bằng sang Xỉu.`;
    } else if (pctTai <= 30) {
        prediction = 'Tài';
        explanation += `Tỷ lệ Tài <=30% → kỳ vọng bù về Tài.`;
    } else {
        // small bias: if pctTai>50 then slightly favor continuation but we use balancing heuristic
        prediction = pctTai > 55 ? 'Xỉu' : (pctTai < 45 ? 'Tài' : (pctTai >= 50 ? 'Tài' : 'Xỉu'));
        explanation += `Tỷ lệ trung gian → dùng heuristic cân bằng/tiếp tục: dự đoán ${prediction}.`;
    }

    const sampleTaiPct = Math.round(samples.filter(s => s.next === 'Tài').length / samples.length * 100);
    explanation += ` Mẫu hỗ trợ: ${sampleTaiPct}% dẫn đến Tài.`;
    return { prediction, explanation };
}

// Model 3 - Markov Chain
function model3MarkovChain(data, samples, modelId) {
    if (data.length < 2) {
        return { prediction: 'Tài', explanation: `Model ${modelId} (Markov): Không đủ dữ liệu để tính ma trận chuyển trạng thái.` };
    }
    let TT=0, TX=0, XT=0, XX=0;
    for (let i=0;i<data.length-1;i++){
        if (data[i] === 'Tài') {
            if (data[i+1] === 'Tài') TT++; else TX++;
        } else {
            if (data[i+1] === 'Xỉu') XX++; else XT++;
        }
    }
    const pTT = TT + TX > 0 ? TT / (TT + TX) : 0.5;
    const pXX = XX + XT > 0 ? XX / (XX + XT) : 0.5;
    const last = data[data.length-1];
    let prediction = last === 'Tài' ? (pTT >= 0.5 ? 'Tài' : 'Xỉu') : (pXX >= 0.5 ? 'Xỉu' : 'Tài');
    const explanation = `Model ${modelId} (Markov): P(T→T)=${Math.round(pTT*100)}%, P(X→X)=${Math.round(pXX*100)}%. Kết quả hiện tại "${last}" → dự đoán ${prediction}.`;
    return { prediction, explanation };
}

// Model 4 - N-gram Matching (4-gram primary, fallback 3-gram)
function model4NgramMatching(data, samples, modelId) {
    const n = Math.min(4, Math.max(3, data.length - 1)); // use up to 4
    const lastN = data.slice(-n).join('-');
    // Build n-gram map from data
    const ngramMap = {};
    for (let i=0;i<=data.length - n - 1;i++) {
        const key = data.slice(i, i+n).join('-');
        const next = data[i+n];
        if (!ngramMap[key]) ngramMap[key] = { Tài:0, Xỉu:0, total:0 };
        ngramMap[key][next] = (ngramMap[key][next] || 0) + 1;
        ngramMap[key].total++;
    }

    let prediction = 'Tài';
    let explanation = `Model ${modelId} (N-gram): So khớp chuỗi length ${n} "${lastN}". `;
    if (ngramMap[lastN]) {
        const entry = ngramMap[lastN];
        prediction = entry.Tài >= entry.Xỉu ? 'Tài' : 'Xỉu';
        explanation += `Tìm thấy ${entry.total} match: Tài=${entry.Tài}, Xỉu=${entry.Xỉu}. Dự đoán ${prediction}.`;
    } else {
        // no exact match -> try shorter gram or fallback to sample majority
        const shorter = n > 3 ? data.slice(-3).join('-') : null;
        if (shorter && ngramMap[shorter]) {
            const entry = ngramMap[shorter];
            prediction = entry.Tài >= entry.Xỉu ? 'Tài' : 'Xỉu';
            explanation += `Không match exact, dùng shorter "${shorter}": Tài=${entry.Tài}, Xỉu=${entry.Xỉu} => ${prediction}.`;
        } else {
            const sampleTai = samples.filter(s=>s.next==='Tài').length;
            prediction = sampleTai >= samples.length/2 ? 'Tài' : 'Xỉu';
            explanation += `Không có match n-gram. Dùng mẫu: ${sampleTai}/${samples.length} lead Tài => ${prediction}.`;
        }
    }
    return { prediction, explanation };
}

// Model 5 - Heuristic Weighted (pattern 40%, markov 30%, freq 20%, ngram 10%)
function model5Heuristic(data, samples, modelId) {
    // patternScore: fraction of samples that predict Tài
    const patternFraction = samples.filter(s => s.next === 'Tài').length / samples.length;
    const freqFraction = data.filter(x => x === 'Tài').length / data.length;

    // Markov proxy: compute pTT and pXX average to produce a Tài preference metric
    let TT=0, TX=0, XT=0, XX=0;
    for (let i=0;i<data.length-1;i++){
        if (data[i] === 'Tài') {
            if (data[i+1] === 'Tài') TT++; else TX++;
        } else {
            if (data[i+1] === 'Xỉu') XX++; else XT++;
        }
    }
    const pTT = TT + TX > 0 ? TT / (TT + TX) : 0.5;
    const pXX = XX + XT > 0 ? XX / (XX + XT) : 0.5;
    // markovFavor: positive if markov favors Tài overall
    const markovFavor = (pTT - (1 - pXX)) / 2 + 0.5; // normalized approx

    // ngram proxy: check last 4-gram support for Tài
    const last4 = data.slice(-4).join('-');
    // compute a mini ngram vote
    let ngramFavor = 0.5;
    for (let i=0;i<=data.length-5;i++){
        const key = data.slice(i,i+4).join('-');
        const next = data[i+4];
        if (key === last4) {
            ngramFavor = (next === 'Tài') ? 1 : 0;
            break;
        }
    }

    const totalScore = 0.4 * patternFraction + 0.3 * markovFavor + 0.2 * freqFraction + 0.1 * ngramFavor;
    const prediction = totalScore > 0.5 ? 'Tài' : 'Xỉu';
    const explanation = `Model ${modelId} (Heuristic): Weights => Pattern 40% (${Math.round(patternFraction*100)}%), Markov 30% (${Math.round(markovFavor*100)}%), Freq 20% (${Math.round(freqFraction*100)}%), Ngram 10% (${Math.round(ngramFavor*100)}%). Tổng score Tài: ${Math.round(totalScore*100)}% → Dự đoán ${prediction}.`;

    return { prediction, explanation };
}

// ----------------- AI TỔNG HỢP Explanation Builder -----------------
function buildAiTongHopExplanation({ finalResult, confidence, voteTai, voteXiu, totalVotes, patternSamples, explanations, history }) {
    // Short header
    let text = `AI TỔNG HỢP: Dự đoán chính là **${finalResult}** (độ tin cậy ${confidence}%). Tổng phiếu: Tài ${voteTai}/${totalVotes}, Xỉu ${voteXiu}/${totalVotes}.\n\n`;

    // Summary of pattern samples
    const taiLead = patternSamples.filter(s => s.next === 'Tài').length;
    text += `Chi tiết mẫu (tổng ${patternSamples.length} mẫu): ${taiLead} mẫu dẫn tới Tài, ${patternSamples.length - taiLead} dẫn tới Xỉu. Những mẫu top giúp hệ thống nhận diện các loại cầu: bệt, 1-1, 2-2, 1-2-1, 2-1-2, 3-1, 1-3, 2-3, 3-2, 4-1, 1-4.\n\n`;

    // Attach model explanations (most load-bearing first: show up to 5-7)
    text += `Giải thích chi tiết từ các model (tóm tắt):\n`;
    explanations.forEach((e, i) => {
        text += `• AI #${i+1}: ${e}\n`;
    });

    // Add concrete example reference (last history snippet)
    const lastWindow = history.slice(-10).join('-');
    text += `\nDữ liệu tham chiếu (10 phiên gần nhất): ${lastWindow}.\n`;

    // Final note about limits and recommendation
    text += `\nLưu ý: Mô hình phân tích cầu dựa trên lịch sử — không thể đảm bảo chính xác 100%. Khuyến nghị: kết hợp phân tích AI với quản lý vốn (Kelly hoặc sizing) và chỉ sử dụng kết quả như 1 nguồn tham khảo.\n`;

    return text;
}

// ----------------- Express endpoints -----------------
app.get('/predict-tai-xiu', async (req, res) => {
    const rawData = await fetchData();
    if (!rawData) {
        return res.status(500).json({ error: "Failed to fetch data from the source API." });
    }

    // Try to normalize incoming rawData to provide session, dice, total, result
    const session = rawData.Phien || rawData.phien || rawData.session || null;
    const dice = (rawData.Xuc_xac_1 !== undefined && rawData.Xuc_xac_2 !== undefined && rawData.Xuc_xac_3 !== undefined)
        ? `${rawData.Xuc_xac_1}-${rawData.Xuc_xac_2}-${rawData.Xuc_xac_3}`
        : (rawData.Xuc_xac || rawData.dice || null);
    const total = rawData.Tong || rawData.tong || rawData.total || null;
    const result = rawData.Ket_qua || rawData.ket_qua || rawData.result || null;

    const p = predict(rawData);

    const fullResponse = {
        session,
        dice,
        total,
        result,
        next_session: typeof session === 'number' ? session + 1 : null,
        predict_goc: p.predict_goc,
        tin_cay: p.tin_cay,
        tong_quat_so_do_du_doan: p.tong_quat_so_do_du_doan,
        giai_thich: p.giai_thich,
        id: "@ Văn Nhật Tới Ngủ Cùng Nè 🫦"
    };

    res.json(fullResponse);
});

app.get('/', (req, res) => {
    res.send('API is running. Use /predict-tai-xiu endpoint to get enhanced predictions (AI TỔNG HỢP).');
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
